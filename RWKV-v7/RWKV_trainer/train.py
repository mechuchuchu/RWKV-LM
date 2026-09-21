#!/usr/bin/env python3
"""Fine-tune an RWKV-7 Transformers checkpoint.

The upstream RWKV-v7/train_temp trainer is intended for pretraining from the
native .pth format. This script is for the newer Transformers/safetensors
checkpoint format and saves PEFT adapters that can be loaded with
PeftModel.from_pretrained().
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/full fine-tuning for an RWKV-7 HF checkpoint")
    parser.add_argument("--model_dir", required=True, help="Local HF model directory or model id")
    parser.add_argument("--train_file", required=True, help="JSONL or UTF-8 text training file")
    parser.add_argument("--output_dir", required=True, help="Directory for the adapter and checkpoints")
    parser.add_argument("--text_key", default="text", help="JSONL field containing text")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1, help="Overrides epochs when positive")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--wkv_implementation", choices=["chunked", "eager"], default="chunked")
    parser.add_argument("--fused_wkv", action="store_true", help="Use the adapted RWKV-LM CUDA WKV kernel for training")
    parser.add_argument("--full_finetune", action="store_true", help="Train all model weights; likely OOM on a 12 GB GPU")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_targets",
        nargs="+",
        default=["receptance", "key", "value", "output"],
        help="Linear submodule names to receive LoRA adapters",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Do not contact Hugging Face Hub while loading the model/tokenizer",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
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


def load_record_text(record: Any, tokenizer: Any, text_key: str) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        raise ValueError("Each JSONL record must be a string or an object")

    messages = record.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            raise ValueError("The 'messages' field must be a list")
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except (AttributeError, TypeError, ValueError):
            # Keep the trainer useful with a tokenizer that has no chat template.
            return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)

    value = record.get(text_key)
    if value is None:
        raise ValueError(f"JSONL record has neither 'messages' nor {text_key!r}")
    if not isinstance(value, str):
        raise ValueError(f"JSONL field {text_key!r} must contain a string")
    return value


class TokenizedTextDataset(Dataset[list[int]]):
    def __init__(self, path: str, tokenizer: Any, max_length: int, text_key: str):
        self.examples: list[list[int]] = []
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)

        if source.suffix.lower() == ".json":
            parsed = json.loads(source.read_text(encoding="utf-8"))
            records: Iterable[Any] = parsed if isinstance(parsed, list) else [parsed]
        elif source.suffix.lower() == ".jsonl":
            records = (
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        else:
            records = source.read_text(encoding="utf-8").splitlines()

        for record in records:
            text = load_record_text(record, tokenizer, text_key).strip()
            if not text:
                continue
            token_ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            if len(token_ids) >= 2:
                self.examples.append(token_ids)

        if not self.examples:
            raise ValueError(f"No usable examples found in {source}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> list[int]:
        return self.examples[index]


class CausalLMCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[list[int]]) -> dict[str, torch.Tensor]:
        width = max(len(example) for example in examples)
        input_ids = torch.full((len(examples), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
        for row, example in enumerate(examples):
            length = len(example)
            input_ids[row, :length] = torch.tensor(example, dtype=torch.long)
            attention_mask[row, :length] = 1
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: Path,
    step: int,
    epoch: int,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {"global_step": step, "epoch": epoch, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint_dir / "trainer_state.pt",
    )


def main() -> None:
    args = parse_args()
    if args.max_length < 2:
        raise ValueError("--max_length must be at least 2")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be positive")
    if args.num_train_epochs < 1 and args.max_steps < 1:
        raise ValueError("Set --num_train_epochs or a positive --max_steps")

    set_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(args.device)
    model_dtype = choose_dtype(args.dtype, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device} model_dtype={model_dtype} model_dir={args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id/eos_token_id")

    load_kwargs = {
        "trust_remote_code": True,
        "dtype": model_dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    model.config.use_cache = False
    model.config.wkv_implementation = args.wkv_implementation
    if args.fused_wkv:
        if args.wkv_implementation != "chunked":
            raise ValueError("--fused_wkv requires --wkv_implementation chunked")
        from fused_wkv import install_fused_wkv

        install_fused_wkv(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if not args.full_finetune:
        model = add_lora(model, args)
    model.to(device)
    model.train()

    dataset = TokenizedTextDataset(args.train_file, tokenizer, args.max_length, args.text_key)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CausalLMCollator(tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_steps = args.max_steps if args.max_steps > 0 else updates_per_epoch * args.num_train_epochs
    warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else int(total_steps * args.warmup_ratio)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    amp_enabled = device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and model_dtype == torch.float16)

    print(f"examples={len(dataset)} batches/epoch={len(loader)} total_steps={total_steps} warmup_steps={warmup_steps}")
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    start_time = time.time()

    epochs_to_run = args.num_train_epochs
    if args.max_steps > 0:
        epochs_to_run = max(epochs_to_run, math.ceil(args.max_steps / max(updates_per_epoch, 1)))

    for epoch in range(epochs_to_run):
        for batch_index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=model_dtype, enabled=amp_enabled):
                output = model(**batch, use_cache=False)
                loss = output.loss
                scaled_loss = loss / args.gradient_accumulation_steps

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch} batch={batch_index}: {loss.item()}")
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            is_last_batch = batch_index + 1 == len(loader)
            should_step = micro_step % args.gradient_accumulation_steps == 0 or is_last_batch
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            clip_grad_norm_(parameters, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step == 1 or global_step % args.logging_steps == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                print(
                    f"step={global_step}/{total_steps} epoch={epoch + 1} "
                    f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.3e} "
                    f"steps/s={global_step / elapsed:.3f}"
                )
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir, global_step, epoch)
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save({"global_step": global_step, "epoch": epoch}, output_dir / "trainer_state.pt")
    print(f"saved={output_dir} global_step={global_step}")


if __name__ == "__main__":
    main()
