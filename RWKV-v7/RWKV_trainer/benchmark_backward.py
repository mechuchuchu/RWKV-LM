#!/usr/bin/env python3
"""Benchmark RWKV-7 backward time across model, batch, and sequence sizes.

Examples:

    python benchmark_backward.py \
        --model /path/to/model \
        --batch-sizes 1 2 \
        --seq-lens 64 128 256 \
        --mode lora \
        --wkv chunked

The default loss is ``last_token``.  It keeps the benchmark usable on smaller
GPUs while still backpropagating through the whole recurrent sequence.  Use
``--loss-mode causal`` for a closer approximation to a normal language-model
training step; it retains the full vocabulary logits and uses more memory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


LOCAL_DEFAULT = Path(
    "/workspace/.hf_home/hub/models--RWKV--RWKV7-G1j-1.5B-20260831/"
    "snapshots/2c18b29ab7fbece25ff6112281eea0fa41fcb30f"
)
REMOTE_DEFAULT = "RWKV/RWKV7-G1j-1.5B-20260831"


def parse_args() -> argparse.Namespace:
    default_model = str(LOCAL_DEFAULT) if LOCAL_DEFAULT.is_dir() else REMOTE_DEFAULT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        dest="models",
        nargs="+",
        default=[default_model],
        help="Local snapshot path or Hugging Face model id. Pass several to compare sizes.",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--wkv", choices=["eager", "chunked", "fused"], default="chunked")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument(
        "--loss-mode",
        choices=["last_token", "causal"],
        default="last_token",
        help="last_token reduces memory; causal retains full logits and uses normal LM loss.",
    )
    parser.add_argument(
        "--lora-targets",
        nargs="+",
        default=["receptance", "key", "value", "output"],
        help="Linear module names used in --mode lora.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def add_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.lora_targets,
    )
    return get_peft_model(model, config)


def install_wkv_backend(model: torch.nn.Module, backend: str) -> None:
    # The remote RWKV-7 implementation owns this registry.  Keep the benchmark
    # independent of its source path and only use the public config/registry
    # contract exposed by modeling_rwkv7.py.
    model.config.wkv_implementation = "chunked" if backend == "fused" else backend
    if backend != "fused":
        return

    if not torch.cuda.is_available():
        raise RuntimeError("The fused WKV backend requires CUDA")
    trainer_dir = Path(__file__).resolve().parent
    if str(trainer_dir) not in sys.path:
        sys.path.insert(0, str(trainer_dir))
    from fused_wkv import install_fused_wkv

    install_fused_wkv(model)


def load_model(model_ref: str, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
    }
    model = AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
    model.config.use_cache = False
    install_wkv_backend(model, args.wkv)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.mode == "lora":
        model = add_lora(model, args)
    model.to(device)
    model.train()
    return model


def parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def make_batch(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    loss_mode: str,
) -> dict[str, torch.Tensor]:
    vocab_size = int(model.config.vocab_size)
    # Avoid the pad/eos token so every position is a real recurrent step.
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    batch = {"input_ids": input_ids}
    if loss_mode == "causal":
        batch["labels"] = input_ids.clone()
    return batch


def forward_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    loss_mode: str,
) -> torch.Tensor:
    if loss_mode == "causal":
        output = model(**batch, use_cache=False)
        return output.loss

    output = model(
        input_ids=batch["input_ids"],
        use_cache=False,
        logits_to_keep=1,
    )
    # Only the final logit is materialized, but the recurrent backward still
    # traverses all seq_len positions.
    return output.logits.float().mean()


def is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def benchmark_shape(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    batch = make_batch(model, batch_size, seq_len, device, args.loss_mode)
    model.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        loss = forward_loss(model, batch, args.loss_mode)
        loss.backward()
        model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    forward_ms: list[float] = []
    backward_ms: list[float] = []
    losses: list[float] = []
    for _ in range(args.steps):
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            forward_start = torch.cuda.Event(enable_timing=True)
            backward_start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            forward_start.record()
            loss = forward_loss(model, batch, args.loss_mode)
            backward_start.record()
            loss.backward()
            end.record()
            end.synchronize()
            forward_ms.append(forward_start.elapsed_time(backward_start))
            backward_ms.append(backward_start.elapsed_time(end))
        else:
            start = time.perf_counter()
            loss = forward_loss(model, batch, args.loss_mode)
            before_backward = time.perf_counter()
            loss.backward()
            end = time.perf_counter()
            forward_ms.append((before_backward - start) * 1000.0)
            backward_ms.append((end - before_backward) * 1000.0)
        losses.append(float(loss.detach().cpu()))

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 2**20

    median_forward = statistics.median(forward_ms)
    median_backward = statistics.median(backward_ms)
    median_total = median_forward + median_backward
    tokens = batch_size * seq_len
    return {
        "status": "ok",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "forward_ms": round(median_forward, 3),
        "backward_ms": round(median_backward, 3),
        "forward_backward_ms": round(median_total, 3),
        "backward_tokens_per_sec": round(tokens / (median_backward / 1000.0), 2),
        "forward_backward_tokens_per_sec": round(tokens / (median_total / 1000.0), 2),
        "loss": round(statistics.mean(losses), 6),
        "peak_memory_mb": round(peak_memory_mb, 1) if peak_memory_mb is not None else None,
    }


def print_result(row: dict[str, Any]) -> None:
    if row["status"] == "oom":
        print(
            f"B={row['batch_size']:>3} T={row['seq_len']:>5}  OOM"
            f"  ({row.get('error', 'CUDA out of memory')})"
        )
        return
    print(
        f"B={row['batch_size']:>3} T={row['seq_len']:>5}  "
        f"fwd={row['forward_ms']:>9.2f} ms  "
        f"bwd={row['backward_ms']:>9.2f} ms  "
        f"total={row['forward_backward_ms']:>9.2f} ms  "
        f"bwd_tok/s={row['backward_tokens_per_sec']:>10.1f}  "
        f"peak={row['peak_memory_mb'] or 0:>8.1f} MB"
    )


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.steps < 1:
        raise ValueError("--warmup must be non-negative and --steps must be positive")
    if any(value < 1 for value in [*args.batch_sizes, *args.seq_lens]):
        raise ValueError("batch sizes and sequence lengths must be positive")

    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    amp_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    if args.wkv == "fused" and dtype != torch.bfloat16:
        raise ValueError("The supplied fused RWKV-7 snapshot kernel expects BF16 activations")

    all_rows: list[dict[str, Any]] = []
    for model_ref in args.models:
        print(f"\nmodel={model_ref}")
        print(f"device={device} dtype={dtype} mode={args.mode} wkv={args.wkv} loss={args.loss_mode}")
        model = None
        try:
            # The autocast context belongs around the actual calls.  Loading is
            # deliberately outside it so model construction is not benchmarked.
            model = load_model(model_ref, args, device, dtype)
            total_params, trainable_params = parameter_count(model)
            print(f"parameters={total_params:,} trainable={trainable_params:,}")
            for batch_size in args.batch_sizes:
                for seq_len in args.seq_lens:
                    try:
                        if amp_enabled:
                            with torch.autocast(device_type=device.type, dtype=dtype):
                                result = benchmark_shape(model, batch_size, seq_len, args, device)
                        else:
                            result = benchmark_shape(model, batch_size, seq_len, args, device)
                        result.update(
                            {
                                "model": model_ref,
                                "dtype": str(dtype).removeprefix("torch."),
                                "device": str(device),
                                "mode": args.mode,
                                "wkv": args.wkv,
                                "loss_mode": args.loss_mode,
                                "total_params": total_params,
                                "trainable_params": trainable_params,
                            }
                        )
                    except Exception as error:  # keep the shape sweep going after OOM
                        if not is_oom(error):
                            raise
                        result = {
                            "model": model_ref,
                            "dtype": str(dtype).removeprefix("torch."),
                            "device": str(device),
                            "mode": args.mode,
                            "wkv": args.wkv,
                            "loss_mode": args.loss_mode,
                            "total_params": total_params,
                            "trainable_params": trainable_params,
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "status": "oom",
                            "error": str(error).splitlines()[0][:160],
                        }
                        model.zero_grad(set_to_none=True)
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    all_rows.append(result)
                    print_result(result)
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in all_rows for key in row})
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"saved={args.output}")


if __name__ == "__main__":
    main()
