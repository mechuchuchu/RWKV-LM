#!/usr/bin/env python
"""
RWKV-7 full fine-tuning (fp32)

- BlinkDL .pth 사전학습 가중치 로드 (rwkv7-g1 / g1a 계열, johanwind numpy 레퍼런스와 동일한 수식)
- jsonl ({"text": ...}) -> max_len 고정 padding 방식 causal LM 학습 (pad/eos = token 0, loss는 -100 마스킹)
- torch.compile: WKV recurrence step 커널만 컴파일 (T 루프 전체를 컴파일하면 unroll 지옥이라 step만)
- FSDP 옵션 (torchrun 으로 실행)
- resume 지원 (model + optimizer + step/epoch)

CUDA kernel (권장, 수십 배 빠름):
    python train_rwkv7.py --model ... --data ... --cuda_kernel

single GPU (naive python loop):
    python train_rwkv7.py --model rwkv7-g1a-0.1b-20250728-ctx4096.pth --data data.jsonl \
        --max_len 512 --batch_size 4 --compile

FSDP (N GPUs):
    torchrun --nproc_per_node=N train_rwkv7.py --model ... --data ... --fsdp

resume:
    ... --resume out/ckpt-latest.pt
"""
import argparse
import functools
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint as grad_checkpoint

HEAD_SIZE = 64
PAD_TOKEN = 0  # RWKV world vocab: 0 = <eos>/doc boundary, padding으로도 사용
CHUNK_LEN = 16  # wind_backstepping 커널이 state를 저장하는 주기 (max_len은 이 배수여야 함)

USE_CUDA_KERNEL = False  # --cuda_kernel 시 True로 전환


# ----------------------------------------------------------------------------
# wind_backstepping CUDA kernel (fp32, forward + backward) — 소스 하드코딩
# 원본: BlinkDL RWKV-LM / johanwind backstepping 커널의 fp32(typedef bf=float) 버전
# ----------------------------------------------------------------------------

WIND_CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <assert.h>

using bf = float;
__device__ inline float to_float(const bf & u) { return u; }
__device__ inline bf to_bf(const float & u) { return u; }

typedef bf * __restrict__ F_;

__global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, bf* y_, float* s_, float* sa_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    for (int t = 0; t < T; t++) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        float sa = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            sa += a[j] * state[j];
        }
        sa_[ind] = sa;

        float v = to_float(v_[ind]);
        float y = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v;
            y += s * q[j];
        }
        y_[ind] = to_bf(y);

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
#pragma unroll
            for (int j = 0; j < C; j++) {
                s_[base + j*C] = state[j];
            }
        }
    }
}

__global__ void backward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s_, float * __restrict__ sa_, bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = qi = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = to_float(k_[ind]);
        a[i] = ai = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        v[i] = to_float(v_[ind]);
        dy[i] = dyi = to_float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C;
#pragma unroll
            for (int j = 0; j < C; j++) {
                stateT[j] = s_[base + j];
            }
        }

        float dq = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dq += stateT[j]*dy[j];
        }
        dq_[ind] = to_bf(dq);

        float iwi = 1.0f/wi;
#pragma unroll
        for (int j = 0; j < C; j++) {
            stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j]*stateT[j];
            dk += dstateT[j]*v[j];
            dv += dstate[j]*k[j];
            dSb += dstate[j]*b[j];
            db += dstateT[j]*sa[j];
        }
        dw_[ind] = to_bf(dw * wi * wi_fac);
        dk_[ind] = to_bf(dk);
        dv_[ind] = to_bf(dv);
        db_[ind] = to_bf(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            da += stateT[j]*dSb_shared[j];
        }
        da_[ind] = to_bf(da);

#pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j] = dstate[j]*w[j] + dSb * a[j];
            dstateT[j] = dstateT[j]*wi + ai * dSb_shared[j];
        }
    }
}

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa) {
    forward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,y,s,sa);
}
void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da) {
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,dy,s,sa,dw,dq,dk,dv,dz,da);
}
"""

WIND_CPP_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
using bf = float;
void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa);
void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr());
}
void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da);
void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)dy.data_ptr(),
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)dz.data_ptr(), (bf*)da.data_ptr());
}
TORCH_LIBRARY(wind_backstepping, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()");
}
TORCH_LIBRARY_IMPL(wind_backstepping, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
"""

_WIND_LOADED = False


def load_wind_kernel(verbose=True):
    global _WIND_LOADED
    if _WIND_LOADED:
        return
    from torch.utils.cpp_extension import load_inline
    load_inline(
        name="wind_backstepping",
        cpp_sources=[WIND_CPP_SRC],
        cuda_sources=[WIND_CUDA_SRC],
        is_python_module=False,
        verbose=verbose,
        extra_cuda_cflags=[
            "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_C_={HEAD_SIZE}", f"-D_CHUNK_LEN_={CHUNK_LEN}",
        ],
    )
    _WIND_LOADED = True


class WindBackstepping(torch.autograd.Function):
    """y = wkv7(...) 를 CUDA 커널로. 입력 전부 (B,T,H,N) contiguous fp32.
    w는 log-log 도메인: 커널 내부에서 decay = exp(-exp(w)).
    z = -kk, b = kk*a 로 전달 (S = S*w^T - S@kk (kk*a)^T + v k^T 와 동일)."""

    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0, f"T={T}는 CHUNK_LEN={CHUNK_LEN}의 배수여야 함"
        assert all(i.dtype == torch.float32 for i in (w, q, k, v, z, b))
        assert all(i.is_contiguous() for i in (w, q, k, v, z, b))
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dy = dy.contiguous()
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in (w, q, k, v, z, b)]
        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db


# ----------------------------------------------------------------------------
# WKV-7 recurrence  (numpy 레퍼런스의 S 업데이트를 배치 버전으로 그대로 옮김)
#   S = S * w^T - S @ kk (kk*a)^T + v k^T ;  y = S @ r
# ----------------------------------------------------------------------------

def _wkv7_step(S, r, w, k, v, kk, a):
    # S: (B,H,N,N), 나머지: (B,H,N)
    S = (
        S * w.unsqueeze(-2)
        - (S @ kk.unsqueeze(-1)) @ (kk * a).unsqueeze(-2)
        + v.unsqueeze(-1) @ k.unsqueeze(-2)
    )
    y = (S @ r.unsqueeze(-1)).squeeze(-1)
    return S, y


WKV_STEP = _wkv7_step  # --compile 시 torch.compile 버전으로 교체됨


def wkv7(r, w, k, v, kk, a):
    # 입력 전부 (B,T,H,N) -> y: (B,T,H,N)
    B, T, H, N = r.shape
    S = torch.zeros(B, H, N, N, dtype=r.dtype, device=r.device)
    ys = []
    for t in range(T):
        S, y = WKV_STEP(S, r[:, t], w[:, t], k[:, t], v[:, t], kk[:, t], a[:, t])
        ys.append(y)
    return torch.stack(ys, dim=1)


# ----------------------------------------------------------------------------
# Model (파라미터 이름을 BlinkDL 체크포인트 키와 동일하게 맞춤 -> load_state_dict 그대로 사용)
# ----------------------------------------------------------------------------

class TimeMix(nn.Module):
    def __init__(self, C, layer_id, d_w, d_a, d_v, d_g):
        super().__init__()
        self.layer_id = layer_id
        H = C // HEAD_SIZE
        z = lambda *s: nn.Parameter(torch.zeros(*s))

        self.x_r = z(1, 1, C); self.x_w = z(1, 1, C); self.x_k = z(1, 1, C)
        self.x_v = z(1, 1, C); self.x_a = z(1, 1, C); self.x_g = z(1, 1, C)

        self.w0 = z(1, 1, C); self.w1 = z(C, d_w); self.w2 = z(d_w, C)
        self.a0 = z(1, 1, C); self.a1 = z(C, d_a); self.a2 = z(d_a, C)
        if layer_id > 0:  # v residual은 layer 1+ 에만 존재
            self.v0 = z(1, 1, C); self.v1 = z(C, d_v); self.v2 = z(d_v, C)
        self.g1 = z(C, d_g); self.g2 = z(d_g, C)

        self.k_k = z(1, 1, C); self.k_a = z(1, 1, C); self.r_k = z(H, HEAD_SIZE)

        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

    def forward(self, x, v_first):
        B, T, C = x.shape
        H, N = C // HEAD_SIZE, HEAD_SIZE

        xx = F.pad(x, (0, 0, 1, -1)) - x  # token shift: (last_x - x)
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        # numpy 레퍼런스: w = exp(-sigmoid(z)/e**0.5)  (clamp-w 버전)
        # 항등변형: -softplus(-z) - 0.5 = log(sigmoid(z)/sqrt(e)) 이므로
        #   exp(-exp(w_log)) == exp(-sigmoid(z)/sqrt(e))  -> 두 경로 수식 동일
        w_log = -F.softplus(-(torch.tanh(xw @ self.w1) @ self.w2 + self.w0)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(xv @ self.v1 @ self.v2 + self.v0)

        a = torch.sigmoid(xa @ self.a1 @ self.a2 + self.a0)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = (k * self.k_k).view(B, T, H, N)
        kk = F.normalize(kk, p=2.0, dim=-1, eps=1e-12)  # max(norm, 1e-12) semantics
        k = k * (1 + (a - 1) * self.k_a)

        r_, k_, v_, a_ = [i.view(B, T, H, N) for i in (r, k, v, a)]

        if USE_CUDA_KERNEL:
            y = WindBackstepping.apply(
                w_log.view(B, T, H, N).contiguous(),  # w (log-log 도메인)
                r_.contiguous(), k_.contiguous(), v_.contiguous(),
                (-kk).contiguous(),        # z = -kk
                (kk * a_).contiguous(),    # b = kk * a
            )  # (B,T,H,N)
        else:
            w_ = torch.exp(-torch.exp(w_log)).view(B, T, H, N)
            y = wkv7(r_, w_, k_, v_, kk, a_)  # (B,T,H,N)

        y = self.ln_x(y.reshape(B * T, C)).view(B, T, C)
        y = y + ((r_ * k_ * self.r_k).sum(-1, keepdim=True) * v_).view(B, T, C)
        return self.output(y * g), v_first


class ChannelMix(nn.Module):
    def __init__(self, C, d_ffn):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.key = nn.Linear(C, d_ffn, bias=False)
        self.value = nn.Linear(d_ffn, C, bias=False)

    def forward(self, x):
        xx = F.pad(x, (0, 0, 1, -1)) - x
        k = self.key(x + xx * self.x_k)
        return self.value(torch.relu(k) ** 2)


class Block(nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__()
        C = cfg["n_embd"]
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(C)
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.att = TimeMix(C, layer_id, cfg["d_w"], cfg["d_a"], cfg["d_v"], cfg["d_g"])
        self.ffn = ChannelMix(C, cfg["d_ffn"])

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        dx, v_first = self.att(self.ln1(x), v_first)
        x = x + dx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKV7(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg["n_layer"])])
        self.ln_out = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.grad_ckpt = False

    def forward(self, idx):
        x = self.emb(idx)
        v_first = None
        for block in self.blocks:
            if self.grad_ckpt and self.training:
                x, v_first = grad_checkpoint(block, x, v_first, use_reentrant=False)
            else:
                x, v_first = block(x, v_first)
        return self.head(self.ln_out(x))


def infer_config(sd):
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    vocab_size, C = sd["emb.weight"].shape
    assert C % HEAD_SIZE == 0
    return dict(
        n_layer=n_layer,
        n_embd=C,
        vocab_size=vocab_size,
        d_w=sd["blocks.0.att.w1"].shape[1],
        d_a=sd["blocks.0.att.a1"].shape[1],
        d_g=sd["blocks.0.att.g1"].shape[1],
        d_v=(sd["blocks.1.att.v1"].shape[1] if n_layer > 1 else HEAD_SIZE),
        d_ffn=sd["blocks.0.ffn.key.weight"].shape[0],
    )


# ----------------------------------------------------------------------------
# Data: jsonl -> 고정 max_len padding
# ----------------------------------------------------------------------------

def get_tokenizer():
    import rwkv  # pip install rwkv
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    vocab_file = os.path.join(os.path.dirname(rwkv.__file__), "rwkv_vocab_v20230424.txt")
    return TRIE_TOKENIZER(vocab_file)


class JsonlDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, text_key="text"):
        self.samples = []
        n_trunc = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line)[text_key]
                toks = tokenizer.encode(text)
                if not toks:
                    continue
                if len(toks) > max_len:
                    n_trunc += 1
                seq = (toks + [PAD_TOKEN])[: max_len + 1]  # 끝에 eos(0) 하나
                x, y = seq[:-1], seq[1:]
                pad = max_len - len(x)
                x = x + [PAD_TOKEN] * pad
                y = y + [-100] * pad  # padding 위치는 loss 제외
                self.samples.append(
                    (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
                )
        self.n_trunc = n_trunc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ----------------------------------------------------------------------------
# Checkpoint save / load (resume)
# ----------------------------------------------------------------------------

def save_checkpoint(path, model_for_sd, optimizer, epoch, gstep, use_fsdp, rank):
    if use_fsdp:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType, FullStateDictConfig, FullOptimStateDictConfig,
        )
        with FSDP.state_dict_type(
            model_for_sd, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            msd = model_for_sd.state_dict()
            osd = FSDP.optim_state_dict(model_for_sd, optimizer)
        if rank == 0:
            torch.save({"model": msd, "optim": osd, "epoch": epoch, "step": gstep}, path)
        dist.barrier()
    else:
        msd = {k.replace("_orig_mod.", ""): v for k, v in model_for_sd.state_dict().items()}
        torch.save({"model": msd, "optim": optimizer.state_dict(), "epoch": epoch, "step": gstep}, path)
    if rank == 0:
        print(f"[ckpt] saved -> {path} (epoch {epoch}, step {gstep})")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None, help="사전학습 BlinkDL .pth (resume 시 생략 가능)")
    p.add_argument("--data", type=str, required=True, help="학습용 jsonl 경로")
    p.add_argument("--text_key", type=str, default="text")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4, help="GPU당 micro batch size")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="README 규칙대로 큰 행렬(emb/head/att·ffn Linear)에만 적용")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="out")
    p.add_argument("--save_every", type=int, default=200, help="optimizer step 기준")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None, help="ckpt-latest.pt 경로")
    p.add_argument("--compile", action="store_true", help="WKV step torch.compile (python 경로 전용)")
    p.add_argument("--cuda_kernel", action="store_true",
                   help="wind_backstepping fp32 CUDA 커널 사용 (nvcc 필요, max_len은 16의 배수)")
    p.add_argument("--fsdp", action="store_true", help="torchrun 으로 실행")
    p.add_argument("--grad_ckpt", action="store_true", help="block 단위 activation checkpointing (메모리 절약)")
    p.add_argument("--tf32", action="store_true", help="matmul tf32 허용 (기본은 strict fp32)")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    assert args.model or args.resume, "--model 또는 --resume 중 하나는 필요"

    # ---- distributed / device ----
    use_fsdp = args.fsdp
    if use_fsdp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        rank, world = 0, 1
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(args.seed + rank)
    if args.tf32:
        torch.set_float32_matmul_precision("high")

    def log0(*a):
        if rank == 0:
            print(*a, flush=True)

    # ---- weights ----
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        sd = resume_ckpt["model"]
        log0(f"[resume] {args.resume} (epoch {resume_ckpt['epoch']}, step {resume_ckpt['step']})")
    else:
        sd = torch.load(args.model, map_location="cpu", weights_only=True)
        sd = {k: v.float() for k, v in sd.items()}  # bf16 ckpt -> fp32

    cfg = infer_config(sd)
    log0(f"[model] {cfg}")

    model = RWKV7(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        log0(f"[load] missing={missing} unexpected={unexpected}")
        assert not missing, "필수 파라미터 누락: 체크포인트 구조 확인 필요"
    del sd
    model = model.float().to(device)
    model.grad_ckpt = args.grad_ckpt

    # ---- FSDP wrap ----
    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        model = FSDP(
            model,
            auto_wrap_policy=policy,
            use_orig_params=True,  # torch.compile / 이름 기반 param group 호환
            device_id=(local_rank if device.type == "cuda" else None),
        )
    fsdp_model = model  # state_dict / clip 용 레퍼런스

    # ---- CUDA kernel (wind_backstepping, fp32 forward+backward) ----
    if args.cuda_kernel:
        global USE_CUDA_KERNEL
        assert device.type == "cuda", "--cuda_kernel 은 CUDA 필요"
        assert args.max_len % CHUNK_LEN == 0, f"--cuda_kernel 사용 시 --max_len 은 {CHUNK_LEN}의 배수여야 함"
        if use_fsdp:
            # rank0이 먼저 JIT 컴파일 -> 캐시 경쟁 방지, 나머지는 캐시에서 로드
            if rank == 0:
                load_wind_kernel(verbose=True)
            dist.barrier()
            if rank != 0:
                load_wind_kernel(verbose=False)
        else:
            load_wind_kernel(verbose=True)
        USE_CUDA_KERNEL = True
        log0("[kernel] wind_backstepping (fp32) 사용")

    # ---- compile (WKV step만: T 루프 전체 컴파일은 unroll 때문에 비현실적) ----
    if args.compile:
        if args.cuda_kernel:
            log0("[compile] --cuda_kernel 사용 중엔 python WKV 경로가 안 쓰이므로 compile 효과 없음 (무시)")
        else:
            global WKV_STEP
            WKV_STEP = torch.compile(_wkv7_step, dynamic=False)
            log0("[compile] wkv7 step compiled (첫 step은 컴파일 때문에 느림)")

    # ---- optimizer (wd는 큰 행렬만) ----
    DECAY_SUFFIX = (
        "emb.weight", "head.weight",
        "att.receptance.weight", "att.key.weight", "att.value.weight", "att.output.weight",
        "ffn.key.weight", "ffn.value.weight",
    )
    decay, no_decay = [], []
    for n, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        (decay if n.endswith(DECAY_SUFFIX) else no_decay).append(p_)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.99), eps=1e-8,
    )
    if resume_ckpt is not None:
        if use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            osd = FSDP.optim_state_dict_to_load(fsdp_model, optimizer, resume_ckpt["optim"])
            optimizer.load_state_dict(osd)
        else:
            optimizer.load_state_dict(resume_ckpt["optim"])

    # ---- data ----
    tokenizer = get_tokenizer()
    dataset = JsonlDataset(args.data, tokenizer, args.max_len, args.text_key)
    log0(f"[data] {len(dataset)} samples (truncated: {dataset.n_trunc})")
    if use_fsdp:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            drop_last=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    assert len(loader) > 0, "데이터가 batch_size보다 적음 (drop_last=True)"

    os.makedirs(args.save_dir, exist_ok=True)
    latest_path = os.path.join(args.save_dir, "ckpt-latest.pt")

    start_epoch = resume_ckpt["epoch"] if resume_ckpt else 0
    gstep = resume_ckpt["step"] if resume_ckpt else 0
    del resume_ckpt

    # ---- train ----
    model.train()
    micro = 0
    t0, tok_count = time.time(), 0
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, cfg["vocab_size"]), y.view(-1), ignore_index=-100
            )
            (loss / args.grad_accum).backward()
            micro += 1
            tok_count += x.numel()

            if micro % args.grad_accum == 0:
                # warmup
                lr_now = args.lr * (min(1.0, (gstep + 1) / args.warmup_steps) if args.warmup_steps > 0 else 1.0)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                if use_fsdp:
                    fsdp_model.clip_grad_norm_(args.clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                gstep += 1

                if gstep % args.log_every == 0:
                    dt = time.time() - t0
                    log0(f"epoch {epoch} | step {gstep} | loss {loss.item():.4f} | "
                         f"lr {lr_now:.2e} | {tok_count * world / max(dt, 1e-9):.0f} tok/s")
                    t0, tok_count = time.time(), 0
                if gstep % args.save_every == 0:
                    save_checkpoint(latest_path, fsdp_model, optimizer, epoch, gstep, use_fsdp, rank)

        # epoch 끝: 남은 grad flush + 저장
        if micro % args.grad_accum != 0:
            if use_fsdp:
                fsdp_model.clip_grad_norm_(args.clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            gstep += 1
            micro = 0
        save_checkpoint(latest_path, fsdp_model, optimizer, epoch + 1, gstep, use_fsdp, rank)

    # 최종 모델만 따로 (BlinkDL 스타일 .pth, 추론/변환에 바로 사용)
    if use_fsdp:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig,
        )
        with FSDP.state_dict_type(
            fsdp_model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            msd = fsdp_model.state_dict()
        if rank == 0:
            torch.save(msd, os.path.join(args.save_dir, "rwkv7-finetuned.pth"))
    else:
        msd = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
        torch.save(msd, os.path.join(args.save_dir, "rwkv7-finetuned.pth"))
    log0(f"[done] final model -> {os.path.join(args.save_dir, 'rwkv7-finetuned.pth')}")

    if use_fsdp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
