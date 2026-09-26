"""One-batch CUDA smoke test for 100-digit RWKV7 + ROSA addition."""

import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load

from rosa_add100_toy import CONTEXT_LEN, VOCAB_SIZE, make_batch
from rosa_surrogate_toy import rosa_qkv_ref
from rosa_numba import rosa_qkv_batch_numba


DEVICE = torch.device("cuda")
WIDTH = 128
HEAD_SIZE = 16
CHUNK_LEN = 16
N_BLOCKS = 4
PROXY_WIDTH = 16
ROSA_DROPOUT = 0.1
ROSA_SIGN_FLIP_P = 0.0


def load_wkv7_kernel():
    root = Path(__file__).resolve().parent
    load(
        name="wind_backstepping_add100_smoke",
        sources=[str(root / "cuda/wkv7_op.cpp"), str(root / "cuda/wkv7_cuda.cu")],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
        ],
    )


class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        batch, length, heads, head_size = w.shape
        assert length % CHUNK_LEN == 0
        assert head_size == HEAD_SIZE
        assert all(t.dtype == torch.float32 for t in (w, q, k, v, z, b))
        tensors = [t.contiguous() for t in (w, q, k, v, z, b)]
        w, q, k, v, z, b = tensors
        y = torch.empty_like(v)
        state = torch.empty(
            batch,
            heads,
            length // CHUNK_LEN,
            head_size,
            head_size,
            dtype=torch.float32,
            device=w.device,
        )
        state_aux = torch.empty(
            batch, length, heads, head_size, dtype=torch.float32, device=w.device
        )
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, state, state_aux)
        ctx.save_for_backward(w, q, k, v, z, b, state, state_aux)
        return y

    @staticmethod
    def backward(ctx, dy):
        w, q, k, v, z, b, state, state_aux = ctx.saved_tensors
        dy = dy.contiguous()
        dw, dq, dk, dv, dz, db = [torch.empty_like(t) for t in (w, q, k, v, z, b)]
        torch.ops.wind_backstepping.backward(
            w, q, k, v, z, b, dy, state, state_aux, dw, dq, dk, dv, dz, db
        )
        return dw, dq, dk, dv, dz, db


def run_rwkv7(q, w, k, v, a, b):
    batch, length, channels = q.shape
    tensors = [t.contiguous().view(batch, length, channels // HEAD_SIZE, HEAD_SIZE) for t in (q, w, k, v, a, b)]
    return WindBackstepping.apply(*tensors).view(batch, length, channels)


class RWKV7TimeMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        channels = WIDTH
        heads = WIDTH // HEAD_SIZE
        args = SimpleNamespace(n_layer=2)
        ratio_0_to_1 = layer_id / (args.n_layer - 1)
        ratio_1_to_almost0 = 1.0 - layer_id / args.n_layer
        ddd = torch.arange(channels, dtype=torch.float32).view(1, 1, channels) / channels
        self.x_r = nn.Parameter(1.0 - ddd.pow(0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - ddd.pow(0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - ddd.pow(0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - ddd.pow(0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - ddd.pow(0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - ddd.pow(0.2 * ratio_1_to_almost0))

        def ortho_init(tensor, scale):
            gain = math.sqrt(tensor.shape[0] / tensor.shape[1]) if tensor.shape[0] > tensor.shape[1] else 1
            return nn.init.orthogonal_(tensor, gain=gain * scale)

        zigzag = torch.arange(channels, dtype=torch.float32)
        zigzag = ((zigzag % HEAD_SIZE) - ((HEAD_SIZE - 1) / 2)) / ((HEAD_SIZE - 1) / 2)
        zigzag = zigzag * zigzag.abs()
        linear = torch.arange(channels, dtype=torch.float32) / (channels - 1) - 0.5
        decay = torch.tensor(
            [-6 + 6 * (n / (channels - 1)) ** (1 + ratio_0_to_1**0.3) for n in range(channels)]
        )

        self.w1 = nn.Parameter(torch.zeros(channels, 16))
        self.w2 = nn.Parameter(ortho_init(torch.zeros(16, channels), 0.1))
        self.w0 = nn.Parameter(decay.view(1, 1, channels) + 0.5 + zigzag.view(1, 1, channels) * 2.5)
        self.a1 = nn.Parameter(torch.zeros(channels, 16))
        self.a2 = nn.Parameter(ortho_init(torch.zeros(16, channels), 0.1))
        self.a0 = nn.Parameter(torch.zeros(1, 1, channels) - 0.19 + zigzag.view(1, 1, channels) * 0.3 + linear.view(1, 1, channels) * 0.4)
        self.v1 = nn.Parameter(torch.zeros(channels, 16))
        self.v2 = nn.Parameter(ortho_init(torch.zeros(16, channels), 0.1))
        self.v0 = nn.Parameter(torch.zeros(1, 1, channels) + 0.73 - linear.view(1, 1, channels) * 0.4)
        self.g1 = nn.Parameter(torch.zeros(channels, 16))
        self.g2 = nn.Parameter(ortho_init(torch.zeros(16, channels), 0.1))
        self.k_k = nn.Parameter(torch.zeros(1, 1, channels) + 0.71 - linear.view(1, 1, channels) * 0.1)
        self.k_a = nn.Parameter(torch.zeros(1, 1, channels) + 1.02)
        self.r_k = nn.Parameter(torch.zeros(heads, HEAD_SIZE) - 0.04)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        self.ln_x = nn.GroupNorm(heads, channels, eps=64e-5)
        self.receptance.weight.data.uniform_(-0.5 / channels**0.5, 0.5 / channels**0.5)
        self.key.weight.data.uniform_(-0.05 / channels**0.5, 0.05 / channels**0.5)
        self.value.weight.data.uniform_(-0.5 / channels**0.5, 0.5 / channels**0.5)
        self.output.weight.data.zero_()

    def forward(self, x, v_first):
        batch, length, channels = x.shape
        heads = channels // HEAD_SIZE
        xx = self.time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if v_first is None:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(batch, length, heads, HEAD_SIZE), dim=-1, p=2.0).view(batch, length, channels)
        k = k * (1 + (a - 1) * self.k_a)
        y = run_rwkv7(r, w, k, v, -kk, kk * a)
        y = self.ln_x(y.reshape(batch * length, channels)).view(batch, length, channels)
        y = y + (
            (r.view(batch, length, heads, HEAD_SIZE) * k.view(batch, length, heads, HEAD_SIZE) * self.r_k)
            .sum(dim=-1, keepdim=True)
            * v.view(batch, length, heads, HEAD_SIZE)
        ).view(batch, length, channels)
        return self.output(y * g), v_first


class ROSAProxy(nn.Module):
    def __init__(self):
        super().__init__()
        width = PROXY_WIDTH
        self.input = nn.Linear(3, width)
        self.position = nn.Embedding(CONTEXT_LEN, width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=2,
            dim_feedforward=width * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=1, enable_nested_tensor=False)
        self.output = nn.Linear(width, 2)

    def forward(self, q, k, v):
        batch, length, channels = q.shape
        x = torch.stack((q, k, v), dim=-1).transpose(1, 2).reshape(batch * channels, length, 3)
        positions = torch.arange(length, device=x.device)
        x = self.input(x) + self.position(positions)
        causal = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.transformer(x, mask=causal)
        return self.output(x).view(batch, channels, length, 2).permute(0, 2, 1, 3)


def exact_rosa_targets(q, k, v):
    batch, length, channels = q.shape
    qb = (q.detach() > 0).to(torch.uint8).transpose(1, 2).contiguous().cpu().numpy().reshape(-1, length)
    kb = (k.detach() > 0).to(torch.uint8).transpose(1, 2).contiguous().cpu().numpy().reshape(-1, length)
    vb = (v.detach() > 0).to(torch.uint8).transpose(1, 2).contiguous().cpu().numpy().reshape(-1, length)
    target = rosa_qkv_batch_numba(qb, kb, vb)
    return torch.from_numpy(target).view(batch, channels, length).transpose(1, 2).long().to(q.device)


class ROSAQKV(nn.Module):
    def __init__(
        self,
        channels,
        dropout=ROSA_DROPOUT,
        sign_flip_p=ROSA_SIGN_FLIP_P,
        ste_gradient_scale=1.0,
    ):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_q = nn.Parameter(torch.zeros(1, 1, channels))
        self.x_k = nn.Parameter(torch.zeros(1, 1, channels))
        self.x_v = nn.Parameter(torch.zeros(1, 1, channels))
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proxy = ROSAProxy()
        self.emb = nn.Parameter(torch.ones(1, 1, channels))
        self.output = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.sign_flip_p = sign_flip_p
        self.ste_gradient_scale = ste_gradient_scale

    def forward(self, x):
        xx = self.time_shift(x) - x
        q = self.q(x + xx * self.x_q)
        k = self.k(x + xx * self.x_k)
        v = self.v(x + xx * self.x_v)

        target = exact_rosa_targets(q, k, v)
        proxy_logits = self.proxy(q, k, v)
        proxy_probs = proxy_logits.softmax(dim=-1)
        proxy_signal = proxy_probs[..., 1] - proxy_probs[..., 0]
        exact_signal = 2 * target.to(proxy_signal.dtype) - 1
        # Subtract the identical proxy tensor before adding the exact value.
        # This preserves the exact forward bits while keeping proxy gradients.
        rosa_output = exact_signal + self.ste_gradient_scale * (
            proxy_signal - proxy_signal.detach()
        )
        if self.training and self.sign_flip_p > 0:
            flip = torch.rand_like(rosa_output) < self.sign_flip_p
            sign = torch.where(flip, -1.0, 1.0)
            rosa_output = rosa_output * sign
        distill_loss = F.cross_entropy(proxy_logits.flatten(0, 2), target.flatten())
        return self.dropout(self.output(rosa_output * self.emb)), distill_loss


class FeedForward(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, channels))
        self.key = nn.Linear(channels, channels * 4, bias=False)
        self.value = nn.Linear(channels * 4, channels, bias=False)
        self.value.weight.data.zero_()
        nn.init.orthogonal_(self.key.weight.data, gain=2.0)

    def forward(self, x):
        xx = self.time_shift(x) - x
        x = x + xx * self.x_k
        return self.value(F.relu(self.key(x)) ** 2)


class Block(nn.Module):
    def __init__(
        self,
        block_id,
        rosa_dropout=ROSA_DROPOUT,
        rosa_sign_flip_p=ROSA_SIGN_FLIP_P,
        ste_gradient_scale=1.0,
    ):
        super().__init__()
        self.ln_rwkv = nn.LayerNorm(WIDTH)
        self.ln_ffn = nn.LayerNorm(WIDTH)
        self.ln_rosa = nn.LayerNorm(WIDTH)
        self.rwkv = RWKV7TimeMix(0 if block_id == 0 else 1)
        self.rosa = ROSAQKV(
            WIDTH,
            dropout=rosa_dropout,
            sign_flip_p=rosa_sign_flip_p,
            ste_gradient_scale=ste_gradient_scale,
        )
        self.ffn = FeedForward(WIDTH)

    def forward(self, x, v_first):
        rosa_out, aux_loss = self.rosa(self.ln_rosa(x))
        rwkv_out, v_first = self.rwkv(self.ln_rwkv(x), v_first)
        x = x + rwkv_out + rosa_out
        x = x + self.ffn(self.ln_ffn(x))
        return x, v_first, aux_loss


class AdditionModel(nn.Module):
    def __init__(
        self,
        rosa_dropout=ROSA_DROPOUT,
        rosa_sign_flip_p=ROSA_SIGN_FLIP_P,
        ste_gradient_scale=1.0,
    ):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, WIDTH)
        self.blocks = nn.ModuleList(
            [
                Block(
                    i,
                    rosa_dropout=rosa_dropout,
                    rosa_sign_flip_p=rosa_sign_flip_p,
                    ste_gradient_scale=ste_gradient_scale,
                )
                for i in range(N_BLOCKS)
            ]
        )
        self.ln_out = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, VOCAB_SIZE)

    def forward(self, ids):
        x = self.emb(ids)
        v_first = None
        aux_losses = []
        for block in self.blocks:
            x, v_first, aux_loss = block(x, v_first)
            aux_losses.append(aux_loss)
        return self.head(self.ln_out(x)), torch.stack(aux_losses).mean()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test needs a CUDA GPU for the RWKV7 kernel.")
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    load_wkv7_kernel()
    # Exclude Numba's one-time JIT cost from steady-state model timing.
    warmup = torch.zeros((1, 1, 1), device=DEVICE)
    exact_rosa_targets(warmup, warmup, warmup)
    batch = make_batch(1, device=DEVICE, rng=random.Random(123))
    model = AdditionModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.eval()
    with torch.no_grad():
        model(batch["input_ids"])
    torch.cuda.synchronize()
    model.train()

    start = time.perf_counter()
    logits, surrogate_loss = model(batch["input_ids"])
    token_loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), batch["target_ids"].reshape(-1), reduction="none"
    ).view_as(batch["target_ids"])
    task_loss = (token_loss * batch["loss_mask"]).sum() / batch["loss_mask"].sum()
    loss = task_loss + 0.1 * surrogate_loss
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - start

    optimizer.zero_grad(set_to_none=True)
    backward_start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - backward_start

    q_grad = model.blocks[0].rosa.q.weight.grad
    rwkv_internal_grad = model.blocks[0].rwkv.receptance.weight.grad
    rwkv_output_grad = model.blocks[0].rwkv.output.weight.grad
    proxy_grad = model.blocks[0].rosa.proxy.output.weight.grad
    assert torch.isfinite(loss).item()
    assert q_grad is not None and torch.isfinite(q_grad).all().item() and q_grad.norm().item() > 0
    assert rwkv_internal_grad is not None and torch.isfinite(rwkv_internal_grad).all().item()
    assert rwkv_output_grad is not None and torch.isfinite(rwkv_output_grad).all().item() and rwkv_output_grad.norm().item() > 0
    assert proxy_grad is not None and torch.isfinite(proxy_grad).all().item() and proxy_grad.norm().item() > 0

    optimizer.step()
    print(
        f"OK: exact-100-digit batch; context={CONTEXT_LEN}; "
        f"task_loss={task_loss.item():.4f}; surrogate_loss={surrogate_loss.item():.4f}; "
        f"total_loss={loss.item():.4f}"
    )
    print(
        f"grad_norms: q_proj={q_grad.norm().item():.4e}; "
        f"rwkv_output={rwkv_output_grad.norm().item():.4e}; "
        f"rwkv_receptance(first-step zero-init)={rwkv_internal_grad.norm().item():.4e}; "
        f"surrogate_head={proxy_grad.norm().item():.4e}"
    )
    print(f"timing: forward={forward_seconds:.2f}s; backward={backward_seconds:.2f}s")
    print("one optimizer step completed")


if __name__ == "__main__":
    main()
