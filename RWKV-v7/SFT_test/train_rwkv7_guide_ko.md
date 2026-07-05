# train_rwkv7.py 사용 가이드

RWKV-7 (BlinkDL g1/g1a 계열) 사전학습 체크포인트를 **fp32 full fine-tuning** 하는 스크립트.
jsonl 텍스트 → max_len 고정 padding → causal LM 학습. torch.compile / FSDP / grad checkpoint / resume 지원.

---

## 1. 환경 준비

```bash
pip install torch --upgrade          # 2.1+ 권장 (compile/FSDP use_orig_params 때문), 2.4+면 안심
pip install rwkv                     # world tokenizer(TRIE_TOKENIZER)용
```

토크나이저는 `rwkv` 패키지에 들어있는 `rwkv_vocab_v20230424.txt`를 자동으로 찾아 씀.
별도 vocab 파일 다운로드 불필요.

체크포인트는 BlinkDL 원본 `.pth` 그대로 사용:

```bash
# 예: https://huggingface.co/BlinkDL/rwkv7-g1/tree/main
wget https://huggingface.co/BlinkDL/rwkv7-g1/resolve/main/rwkv7-g1a-0.1b-20250728-ctx4096.pth
```

n_layer / n_embd / vocab / low-rank 차원(d_w, d_a, d_v, d_g, d_ffn)은 전부
**체크포인트 shape에서 자동 추론**하니까 모델 크기 바뀌어도 인자 수정 필요 없음.

---

## 2. 데이터 포맷

한 줄에 JSON 하나, `text` 필드에 학습할 텍스트:

```jsonl
{"text": "첫 번째 학습 샘플..."}
{"text": "두 번째 샘플..."}
```

필드 이름이 다르면 `--text_key` 로 지정 (예: `--text_key content`).

전처리 규칙:

- 각 텍스트 끝에 eos(token 0) 하나를 붙임
- `max_len` 초과분은 **뒤가 잘림** (truncate). 잘린 샘플 수는 시작 시 로그로 출력됨
- 부족분은 token 0으로 padding, 해당 위치 label은 `-100`이라 loss에서 제외
- 빈 텍스트/빈 줄은 스킵
- 시퀀스끼리 이어붙이기(packing) 없음 — 샘플당 state가 0에서 새로 시작

전체 데이터를 시작 시 한 번에 토크나이즈해서 메모리에 올림. 실험용 수만~수십만 샘플 규모 가정.

---

## 3. 기본 실행 (single GPU)

```bash
python train_rwkv7.py \
    --model rwkv7-g1a-0.1b-20250728-ctx4096.pth \
    --data mydata.jsonl \
    --max_len 512 --batch_size 8 --grad_accum 4 \
    --lr 2e-5 --epochs 3 \
    --compile \
    --save_dir out
```

effective batch = `batch_size × grad_accum × GPU수`. 위 예시면 8×4 = 32.

출력물 (`--save_dir` 아래):

| 파일 | 내용 |
|---|---|
| `ckpt-latest.pt` | model + optimizer + epoch/step. **resume용**. `--save_every` 스텝마다 + 매 epoch 끝에 덮어씀 |
| `rwkv7-finetuned.pth` | 학습 종료 시 model state_dict만. BlinkDL 포맷 그대로라 rwkv pip 패키지 추론, numpy 레퍼런스, HF 변환 스크립트에 바로 사용 가능 |

---

## 4. 인자 정리

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model` | — | 사전학습 `.pth`. `--resume` 쓰면 생략 가능 |
| `--data` | (필수) | jsonl 경로 |
| `--text_key` | `text` | jsonl에서 읽을 필드 |
| `--max_len` | 512 | 고정 시퀀스 길이 (padding/truncate 기준) |
| `--batch_size` | 4 | GPU당 micro batch |
| `--grad_accum` | 1 | gradient accumulation 스텝 수 |
| `--epochs` | 1 | 총 epoch |
| `--lr` | 2e-5 | full FT 기준 1e-5 ~ 5e-5 권장. fp32라 bf16보다 살짝 높여도 안정적 |
| `--weight_decay` | 0.0 | 켜면 README 규칙대로 **큰 행렬에만** 적용 (emb, head, att/ffn Linear). x_r, w1/w2, r_k 같은 소형 파라미터는 항상 wd=0 |
| `--warmup_steps` | 0 | linear warmup. resume 시 global step 기준으로 이어짐 |
| `--clip` | 1.0 | grad norm clipping |
| `--save_every` | 200 | optimizer step 기준 저장 주기 |
| `--log_every` | 10 | 로그 주기 (loss, lr, tok/s) |
| `--resume` | — | `ckpt-latest.pt` 경로 |
| `--compile` | off | WKV step torch.compile (python 경로 전용) |
| `--cuda_kernel` | off | wind_backstepping fp32 CUDA 커널 사용. **속도 원하면 이걸 켤 것** |
| `--fsdp` | off | torchrun 필요 (아래 참고) |
| `--grad_ckpt` | off | block 단위 activation checkpointing |
| `--tf32` | off | matmul tf32 허용. **켜면 strict fp32가 아니게 됨** — 재현성 실험이면 끄고, 속도만 원하면 켜기 |
| `--device` | auto | `cpu` 강제 가능 (디버깅용) |

---

## 5. torch.compile 관련

- 컴파일 대상은 **WKV recurrence step 함수 하나뿐**. T 루프 전체를 compile하면 max_len만큼 unroll돼서 컴파일 시간이 폭발하기 때문에 일부러 이렇게 설계함
- 그래서 가속 폭은 "조금" 수준 — python 오버헤드와 step 내부 커널 fusion 정도
- 첫 forward가 컴파일 때문에 수십 초 걸리는 건 정상
- shape이 바뀌면 재컴파일됨. `drop_last=True`로 배치 크기를 고정해놨으니 보통 1회 컴파일로 끝나지만, **도중에 `--batch_size`나 `--max_len`을 바꿔서 resume하면 다시 컴파일**됨 (동작엔 문제 없음)

---

## 5-1. CUDA 커널 (`--cuda_kernel`) — 진짜 가속

wind_backstepping 커널(forward + backward, `typedef bf = float`라 **fp32 그대로**)을
스크립트 안에 소스 문자열로 하드코딩해놨고, `--cuda_kernel` 주면 `load_inline`으로 JIT 컴파일해서
naive python loop 대신 사용함.

```bash
python train_rwkv7.py --model ... --data ... --max_len 512 --cuda_kernel
```

- **요구사항**: CUDA GPU + nvcc(CUDA toolkit), 그리고 `--max_len`이 **16의 배수** (커널의
  `_CHUNK_LEN_=16` 때문. 어차피 512, 1024 같은 값 쓸 테니 실질 제약 없음)
- 첫 실행 시 컴파일에 1~2분 걸리고 이후엔 torch extension 캐시(`~/.cache/torch_extensions`)에서 즉시 로드
- **속도**: naive loop 대비 수십 배. `--compile`은 커널 경로에선 의미 없어서 자동 무시됨
- **메모리도 절약**: backward용 state를 매 timestep이 아니라 16스텝마다만 저장하고 나머지는
  backward에서 역산(backstepping). 8절의 메모리 걱정이 대부분 사라짐 → `--grad_ckpt` 없이도 여유로움
- **수식 동일성**: 커널은 decay를 `exp(-exp(w))`로 받는데, python 경로의
  `exp(-sigmoid(z)/√e)`와 `w = -softplus(-z) - 0.5` 변형으로 정확히 항등 (max diff ~6e-8 확인).
  두 경로 모두 같은 `w_log`에서 출발하므로 `--cuda_kernel` 켜고 끄고 결과 비교 검증도 가능
- FSDP와 같이 쓸 때는 rank0이 먼저 컴파일하고 barrier 후 나머지가 캐시 로드하게 처리해놔서
  그냥 `--fsdp --cuda_kernel` 같이 주면 됨
- 커널 입출력은 (B,T,H,64) contiguous fp32, `z = -kk`, `b = kk*a`로 매핑되어 있음

---

## 6. FSDP (multi-GPU)

```bash
torchrun --nproc_per_node=4 train_rwkv7.py \
    --model rwkv7-g1a-0.1b-....pth \
    --data mydata.jsonl \
    --max_len 512 --batch_size 4 \
    --fsdp --compile \
    --save_dir out
```

- Block 단위 auto wrap, `use_orig_params=True` (compile·이름 기반 param group 호환)
- fp32 그대로 — mixed precision 설정 없음
- `DistributedSampler` 자동 적용, epoch마다 셔플 시드 갱신
- 저장은 FULL_STATE_DICT로 rank0에 모아서 함 → **rank0 CPU RAM에 모델 전체가 올라갈 공간 필요** (0.1B~1.5B 규모면 문제 없음)
- resume도 동일하게 `--resume out/ckpt-latest.pt` 주면 됨. optimizer state는 `FSDP.optim_state_dict_to_load`로 자동 re-shard
- **GPU 개수를 바꿔서 resume해도 됨** (full state dict 저장이라)

주의: 모든 rank가 초기 로드 시 전체 `.pth`를 CPU에 올렸다가 shard함. 소형 모델 기준이라 단순하게 갔음.

---

## 7. Resume

```bash
python train_rwkv7.py --resume out/ckpt-latest.pt --data mydata.jsonl --epochs 3 ...
```

- `--model` 불필요 (줘도 무시되고 resume 쪽이 우선)
- 복원 범위: 모델, optimizer(AdamW moment 포함), global step, epoch
- epoch **단위**로 이어짐 — epoch 중간에 죽었으면 그 epoch을 처음부터 다시 돎 (데이터 위치까지 정밀 복원은 안 함, 실험용 단순화)
- `ckpt-latest.pt`는 계속 덮어쓰므로, 특정 시점을 보존하고 싶으면 수동으로 복사해둘 것
- resume 시에도 `--data`, `--max_len`, `--lr` 등은 다시 줘야 함 (ckpt에 args 저장 안 함)

---

## 8. 메모리 & 성능 감각

naive recurrence라 backward를 위해 **매 timestep의 state S (B×H×64×64)를 전부 유지**함.
대략적인 activation 메모리 스케일: `B × T × n_layer × n_embd × 64 × 4 bytes` + 부수 텐서 몇 배.

예) 0.1B (L12, D768), B=8, T=512 → state만 layer당 ~0.8GB, 12층이면 ~10GB + α.

메모리 부족하면 우선순위:

1. `--grad_ckpt` — block 경계만 저장하고 block 내부는 재계산. 메모리 대폭 절감, 속도 ~30% 손해
2. `--batch_size` 줄이고 `--grad_accum` 올리기 (effective batch 유지)
3. `--max_len` 줄이기

속도가 아쉬우면: `--compile` + (재현성 안 중요하면) `--tf32`.
그래도 느리면 그건 naive loop의 한계라, 진짜 속도가 필요해지는 시점엔 FLA(fused chunked kernel)로 갈아타는 게 맞음 — 지금 스크립트는 "수식이 레퍼런스와 1:1로 보이는" 걸 우선한 실험용 설계.

---

## 9. 학습 확인 팁

- 사전학습 가중치가 제대로 로드됐다면 **첫 loss가 보통 2~4 근처**에서 시작함 (일반 영어 텍스트 기준). 10 이상에서 시작하면 로드 실패나 토크나이저 문제 의심
- 시작 시 `[load] missing=... unexpected=...` 로그가 뜨면 체크포인트 구조가 예상과 다른 것 — missing이 있으면 assert로 죽게 해놨음
- 학습 후 `rwkv7-finetuned.pth`를 numpy 레퍼런스 스크립트에 넣어서 logits 확인하면 수식 일치 검증 가능 (네가 하던 방식 그대로)

## 10. 흔한 에러

| 증상 | 원인/해결 |
|---|---|
| `데이터가 batch_size보다 적음` | drop_last=True라 샘플 수 < batch_size면 배치 0개. 배치 줄이거나 데이터 늘리기 |
| resume 후 첫 스텝에서 재컴파일 | batch_size/max_len 변경 때문. 정상 동작 |
| FSDP에서 `nccl` 에러 | torchrun으로 실행 안 했거나 `CUDA_VISIBLE_DEVICES` 불일치 |
| loss가 NaN | lr 너무 큼 (fp32라 흔치 않지만 1e-4 이상이면 가능). lr 낮추고 clip 확인 |
| 토크나이저 import 에러 | `pip install rwkv` 안 됨. 패키지 내 vocab 파일 경로를 자동으로 잡으니 rwkv 버전만 최신이면 OK |
