# train_rwkv7.py — Usage Guide

Full **fp32 fine-tuning** of pretrained RWKV-7 checkpoints (BlinkDL g1/g1a family).
Reads a jsonl of text, pads to a fixed `max_len`, and trains causal LM.
Supports torch.compile, FSDP, gradient checkpointing, and resume.

---

## 1. Setup

```bash
pip install torch --upgrade          # 2.1+ recommended (compile / FSDP use_orig_params); 2.4+ is safest
pip install rwkv                     # provides the world tokenizer (TRIE_TOKENIZER)
```

The tokenizer file (`rwkv_vocab_v20230424.txt`) is located automatically inside the
installed `rwkv` package — no separate download needed.

Use an original BlinkDL `.pth` checkpoint directly:

```bash
# e.g. https://huggingface.co/BlinkDL/rwkv7-g1/tree/main
wget https://huggingface.co/BlinkDL/rwkv7-g1/resolve/main/rwkv7-g1a-0.1b-20250728-ctx4096.pth
```

Model dimensions (n_layer, n_embd, vocab, and the low-rank sizes d_w / d_a / d_v / d_g / d_ffn)
are **all inferred from checkpoint shapes**, so you never edit code when switching model sizes.

---

## 2. Data format

One JSON object per line, text in the `text` field:

```jsonl
{"text": "first training sample..."}
{"text": "second sample..."}
```

Use `--text_key` if your field has a different name (e.g. `--text_key content`).

Preprocessing rules:

- One eos token (id 0) is appended to each text.
- Sequences longer than `max_len` are **truncated from the end**. The number of truncated
  samples is printed at startup.
- Shorter sequences are padded with token 0; those positions get label `-100` and are
  excluded from the loss.
- Empty texts / blank lines are skipped.
- No sequence packing — each sample starts from a fresh zero state.

The whole dataset is tokenized once at startup and held in memory. Assumes an experimental
scale of tens of thousands to a few hundred thousand samples.

---

## 3. Basic run (single GPU)

```bash
python train_rwkv7.py \
    --model rwkv7-g1a-0.1b-20250728-ctx4096.pth \
    --data mydata.jsonl \
    --max_len 512 --batch_size 8 --grad_accum 4 \
    --lr 2e-5 --epochs 3 \
    --compile \
    --save_dir out
```

Effective batch = `batch_size × grad_accum × num_GPUs`. Above: 8×4 = 32.

Outputs (under `--save_dir`):

| File | Contents |
|---|---|
| `ckpt-latest.pt` | model + optimizer + epoch/step. **For resume.** Overwritten every `--save_every` steps and at each epoch end. |
| `rwkv7-finetuned.pth` | model state_dict only, written at the end. Plain BlinkDL format — works directly with the rwkv pip package, the numpy reference, or an HF conversion script. |

---

## 4. Arguments

| Arg | Default | Description |
|---|---|---|
| `--model` | — | pretrained `.pth`. Optional when `--resume` is given. |
| `--data` | (required) | jsonl path |
| `--text_key` | `text` | field to read from each jsonl line |
| `--max_len` | 512 | fixed sequence length (padding/truncation target) |
| `--batch_size` | 4 | per-GPU micro batch |
| `--grad_accum` | 1 | gradient accumulation steps |
| `--epochs` | 1 | total epochs |
| `--lr` | 2e-5 | for full FT, 1e-5 – 5e-5 is a good range. fp32 tolerates slightly higher than bf16. |
| `--weight_decay` | 0.0 | if set, applied **only to large matrices** (emb, head, att/ffn Linear), per the README rule. Small params (x_r, w1/w2, r_k, etc.) always get wd=0. |
| `--warmup_steps` | 0 | linear warmup. On resume, continues from the global step. |
| `--clip` | 1.0 | grad-norm clipping |
| `--save_every` | 200 | save interval in optimizer steps |
| `--log_every` | 10 | log interval (loss, lr, tok/s) |
| `--resume` | — | path to `ckpt-latest.pt` |
| `--compile` | off | torch.compile the WKV step |
| `--fsdp` | off | requires torchrun (see below) |
| `--grad_ckpt` | off | per-block activation checkpointing |
| `--tf32` | off | allow tf32 matmul. **Turning this on makes it no longer strict fp32** — leave off for reproducibility experiments, turn on if you only want speed. |
| `--device` | auto | can force `cpu` for debugging |

---

## 5. About torch.compile

- Only the **WKV recurrence step function** is compiled. Compiling the full time loop would
  unroll it `max_len` times and blow up compile time, so this is intentional.
- As a result the speedup is modest — mostly Python overhead reduction and some kernel fusion
  inside the step.
- The first forward taking tens of seconds due to compilation is normal.
- Recompiles on shape change. Batch size is fixed via `drop_last=True`, so you usually compile
  once — but **resuming with a different `--batch_size` or `--max_len` triggers a recompile**
  (harmless).

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

- Per-block auto wrap, `use_orig_params=True` (compatible with compile and name-based param groups).
- Stays in fp32 — no mixed precision configured.
- `DistributedSampler` applied automatically, shuffle seed refreshed each epoch.
- Saving gathers a FULL_STATE_DICT on rank0 → **rank0 needs enough CPU RAM to hold the full
  model** (fine for 0.1B–1.5B).
- Resume works the same way with `--resume out/ckpt-latest.pt`; optimizer state is re-sharded
  automatically via `FSDP.optim_state_dict_to_load`.
- **You can resume with a different number of GPUs** (full state dict on disk).

Note: every rank loads the full `.pth` to CPU at startup before sharding. Kept simple since this
targets smaller models.

---

## 7. Resume

```bash
python train_rwkv7.py --resume out/ckpt-latest.pt --data mydata.jsonl --epochs 3 ...
```

- `--model` not needed (ignored if given; resume takes priority).
- Restores: model, optimizer (AdamW moments included), global step, epoch.
- Resumes at **epoch granularity** — if it died mid-epoch, that epoch restarts from the
  beginning (no exact data-position restore; simplified for experiments).
- `ckpt-latest.pt` is continuously overwritten; copy it manually to preserve a specific point.
- On resume you still pass `--data`, `--max_len`, `--lr`, etc. (args are not stored in the ckpt).

---

## 8. Memory & performance intuition

This is a naive recurrence, so backward keeps **the state S (B×H×64×64) for every timestep**.
Rough activation-memory scaling: `B × T × n_layer × n_embd × 64 × 4 bytes`, plus a few multiples
for auxiliary tensors.

Example: 0.1B (L12, D768), B=8, T=512 → the state alone is ~0.8GB per layer, ~10GB over 12 layers plus overhead.

If you run out of memory, in priority order:

1. `--grad_ckpt` — save only block boundaries and recompute inside each block. Big memory savings,
   ~30% slower.
2. Lower `--batch_size`, raise `--grad_accum` (keeps effective batch constant).
3. Reduce `--max_len`.

If you want more speed: `--compile` + (if reproducibility isn't critical) `--tf32`.
If it's still slow, that's the limit of the naive loop — when you genuinely need speed, switch to
FLA (fused chunked kernels). This script prioritizes making the math visibly match the reference 1:1,
which is an experimental design choice.

---

## 9. Sanity checks

- If the pretrained weights loaded correctly, the **first loss usually starts around 2–4** on
  ordinary English text. Starting above ~10 suggests a load failure or tokenizer mismatch.
- A startup log `[load] missing=... unexpected=...` means the checkpoint structure differs from
  what's expected — the script asserts (dies) if anything required is missing.
- After training, feed `rwkv7-finetuned.pth` into the numpy reference script to verify logits and
  confirm the math matches — the same way you've been checking equivalence.

## 10. Common errors

| Symptom | Cause / fix |
|---|---|
| `data smaller than batch_size` | drop_last=True, so if samples < batch_size you get 0 batches. Lower the batch or add data. |
| Recompile on the first step after resume | Changed batch_size/max_len. Normal. |
| `nccl` error under FSDP | Not launched via torchrun, or `CUDA_VISIBLE_DEVICES` mismatch. |
| Loss is NaN | lr too high (rare in fp32, but possible above ~1e-4). Lower lr, check clip. |
| Tokenizer import error | `pip install rwkv` didn't take. The vocab path is auto-resolved inside the package, so just keep rwkv up to date. |
