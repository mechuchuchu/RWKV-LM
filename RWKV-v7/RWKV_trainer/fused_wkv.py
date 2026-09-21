"""Optional RWKV-7 fused WKV kernel for the Transformers snapshot model.

The upstream training kernel is almost a drop-in match for the snapshot's
recurrent update. This adapter changes its decay input from raw W parameters to
the snapshot's already transformed ``w_log`` representation.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F


CHUNK_LEN = 16
HEAD_SIZE = 64


def _load_extension() -> None:
    if hasattr(torch.ops, "rwkv7_snapshot_wkv") and hasattr(torch.ops.rwkv7_snapshot_wkv, "forward"):
        return
    from torch.utils.cpp_extension import load

    cuda_dir = Path(__file__).resolve().parent / "cuda"
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST",
        ".".join(str(x) for x in torch.cuda.get_device_capability()),
    )
    load(
        name="rwkv7_snapshot_wkv",
        sources=[str(cuda_dir / "rwkv7_snapshot_wkv.cu"), str(cuda_dir / "rwkv7_snapshot_wkv.cpp")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-D_N_={HEAD_SIZE}", f"-D_CHUNK_LEN_={CHUNK_LEN}"],
        is_python_module=False,
        verbose=True,
    )


class _SnapshotWKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w_log, k, v, a, b):
        if r.ndim != 4 or r.shape[-1] != HEAD_SIZE:
            raise ValueError(f"fused RWKV kernel expects [B,T,H,{HEAD_SIZE}], got {tuple(r.shape)}")
        batch, seq_len, heads, _ = r.shape
        if not all(x.is_cuda and x.dtype == torch.bfloat16 for x in (r, w_log, k, v, a, b)):
            raise TypeError("fused RWKV kernel requires contiguous CUDA BF16 tensors")
        padded_len = (seq_len + CHUNK_LEN - 1) // CHUNK_LEN * CHUNK_LEN
        pad = padded_len - seq_len
        if pad:
            zeros = (0, 0, 0, 0, 0, pad)
            r, w_log, k, v, a, b = (F.pad(x, zeros) for x in (r, w_log, k, v, a, b))
        tensors = tuple(x.contiguous() for x in (r, w_log, k, v, a, b))
        r, w_log, k, v, a, b = tensors
        y = torch.empty_like(v)
        workspace = torch.empty(batch, heads, padded_len // CHUNK_LEN, HEAD_SIZE, HEAD_SIZE, device=r.device, dtype=torch.float32)
        sa = torch.empty(batch, padded_len, heads, HEAD_SIZE, device=r.device, dtype=torch.float32)
        torch.ops.rwkv7_snapshot_wkv.forward(r, w_log, k, v, a, b, y, workspace, sa)
        ctx.save_for_backward(r, w_log, k, v, a, b, workspace, sa)
        return y[:, :seq_len]

    @staticmethod
    def backward(ctx, grad_y):
        r, w_log, k, v, a, b, workspace, sa = ctx.saved_tensors
        padded_len = r.shape[1]
        grad_y_padded = F.pad(grad_y, (0, 0, 0, 0, 0, padded_len - grad_y.shape[1])).contiguous()
        grads = [torch.empty_like(x) for x in (r, w_log, k, v, a, b)]
        torch.ops.rwkv7_snapshot_wkv.backward(r, w_log, k, v, a, b, grad_y_padded, workspace, sa, *grads)
        return tuple(grad[:, : grad_y.shape[1]] for grad in grads)


def install_fused_wkv(model: torch.nn.Module) -> None:
    """Replace the snapshot's chunked WKV registry entry for zero-state training."""

    _load_extension()
    globals_dict = model.rwkv7.blocks[0].att.forward.__globals__
    registry = globals_dict["RWKV7_WKV_FUNCTIONS"]

    def fused(r, w_log, k, v, kk, a, state, cu_seq_lens=None):
        if cu_seq_lens is not None:
            raise ValueError("fused snapshot WKV does not support packed sequence boundaries")
        # The trainer sets use_cache=False, so this state is freshly allocated and zero.
        # Returning it unchanged is sufficient because the caller does not request cache.
        # Under BF16 autocast, the model's k_k parameter promotes kk to FP32. The
        # adapted upstream kernel has BF16 activation inputs, so cast inside the
        # graph; autograd still propagates the custom-op gradients through these casts.
        kk_kernel = kk.to(dtype=torch.bfloat16)
        a_kernel = a.to(dtype=torch.bfloat16)
        y = _SnapshotWKV.apply(r, w_log, k, v, -kk_kernel, kk_kernel * a_kernel)
        return y, state

    registry["chunked"] = fused
