"""Train RWKV7+ROSA on a fresh stream of random fixed-width additions."""

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F

from rosa_add100_toy import CONTEXT_LEN, DIGITS, VOCAB_SIZE, make_batch
from rosa_add100_overfit import greedy_sums, losses_and_accuracy
from rosa_add100_smoke import (
    AdditionModel,
    DEVICE,
    exact_rosa_targets,
    load_wkv7_kernel,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", type=int, default=DIGITS)
    parser.add_argument(
        "--context-len",
        type=int,
        help="input length; default is the smallest multiple of 16 that fits the longest sum",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--greedy-every", type=int, default=500)
    parser.add_argument("--greedy-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rosa-dropout", type=float, default=0.0)
    parser.add_argument("--rosa-sign-flip", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reproduce-unsafe-nan",
        action="store_true",
        help="replay the original run's behavior: apply non-finite gradients and stop at the first NaN loss",
    )
    args = parser.parse_args()
    if args.digits < 1:
        parser.error("--digits must be at least 1")
    minimum_context = 3 * args.digits + 3
    context_len = args.context_len
    if context_len is None:
        context_len = ((minimum_context + 15) // 16) * 16
    if context_len < minimum_context:
        parser.error(
            f"--context-len must be at least {minimum_context} for {args.digits}-digit addition"
        )
    if context_len % 16:
        parser.error("--context-len must be divisible by 16 (RWKV7 chunk length)")
    if context_len > CONTEXT_LEN:
        parser.error(
            f"--context-len cannot exceed {CONTEXT_LEN}; ROSA proxy positional embeddings have that limit"
        )
    args.context_len = context_len
    for name in (
        "steps",
        "batch_size",
        "validation_size",
        "eval_every",
        "greedy_every",
        "greedy_size",
        "save_every",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be nonnegative")
    if args.reproduce_unsafe_nan and args.weight_decay != 0:
        parser.error("--reproduce-unsafe-nan requires --weight-decay 0")
    if not torch.cuda.is_available():
        raise RuntimeError("This training run needs a CUDA GPU for the RWKV7 kernel.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_rng = random.Random(args.seed)
    validation_rng = random.Random(args.seed + 1)
    if args.run_dir is None and args.resume is not None:
        run_dir = args.resume.resolve().parent
    elif args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(__file__).resolve().parent / "runs" / f"add{args.digits}_stream_{stamp}"
    else:
        run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "last.pt"

    print(
        f"run_dir={run_dir} digits={args.digits} context={context_len} "
        f"steps={args.steps} batch={args.batch_size} "
        f"validation={args.validation_size} lr={args.learning_rate:g} "
        f"weight_decay={args.weight_decay:g}(linear matrices only) "
        f"rosa_dropout={args.rosa_dropout:g} sign_flip={args.rosa_sign_flip:g} "
        f"seed={args.seed}",
        flush=True,
    )
    load_wkv7_kernel()
    # Compile and warm the Numba operator before timing the actual run.
    warmup = torch.zeros((1, 1, 1), device=DEVICE)
    exact_rosa_targets(warmup, warmup, warmup)

    validation = make_batch(
        args.validation_size,
        device=DEVICE,
        rng=validation_rng,
        digits=args.digits,
        context_len=context_len,
    )
    model = AdditionModel(
        rosa_dropout=args.rosa_dropout,
        rosa_sign_flip_p=args.rosa_sign_flip,
    ).to(DEVICE)
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and not name.startswith("emb.") and ".position." not in name:
            decay_params.append(parameter)
        else:
            no_decay_params.append(parameter)
    if args.weight_decay == 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=args.learning_rate,
        )
    start_step = 0
    bad_grad_steps = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        bad_grad_steps = state.get("bad_grad_steps", 0)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        train_rng.setstate(state["train_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"resumed_from={args.resume} step={start_step}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    fields = [
        "step",
        "examples_seen",
        "train_loss",
        "train_total_loss",
        "train_rosa_loss",
        "train_token_acc",
        "validation_loss",
        "validation_total_loss",
        "validation_rosa_loss",
        "validation_token_acc",
        "greedy_exact",
        "greedy_count",
        "bad_grad_steps",
        "elapsed_seconds",
    ]
    start = time.perf_counter()
    train_loss_window = []
    train_total_window = []
    train_rosa_window = []
    train_accuracy_window = []

    log_mode = "a" if args.resume is not None and log_path.exists() else "w"
    resume_baseline_logged = False
    if args.resume is not None and log_path.exists() and log_path.stat().st_size > 0:
        with log_path.open(newline="", encoding="utf-8") as existing_log:
            existing_rows = list(csv.DictReader(existing_log))
        if existing_rows and existing_rows[-1]["step"]:
            resume_baseline_logged = int(existing_rows[-1]["step"]) == start_step
    with log_path.open(log_mode, newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fields)
        if log_path.stat().st_size == 0:
            writer.writeheader()
            log_file.flush()

        if args.resume is None:
            model.eval()
            with torch.no_grad():
                val_loss, val_rosa, val_accuracy = losses_and_accuracy(model, validation)
            writer.writerow(
                {
                    "step": 0,
                    "examples_seen": 0,
                    "train_loss": "",
                    "train_total_loss": "",
                    "train_rosa_loss": "",
                    "train_token_acc": "",
                    "validation_loss": f"{val_loss.item():.6f}",
                    "validation_total_loss": f"{(val_loss + 0.1 * val_rosa).item():.6f}",
                    "validation_rosa_loss": f"{val_rosa.item():.6f}",
                    "validation_token_acc": f"{val_accuracy.item():.6f}",
                    "greedy_exact": "",
                    "greedy_count": "",
                    "elapsed_seconds": f"{time.perf_counter() - start:.1f}",
                }
            )
            log_file.flush()
            print(
                f"step=0 val_loss={val_loss.item():.4f} "
                f"val_rosa={val_rosa.item():.4f} val_token_acc={val_accuracy.item():.3%}",
                flush=True,
            )
        elif not resume_baseline_logged:
            model.eval()
            with torch.no_grad():
                val_loss, val_rosa, val_accuracy = losses_and_accuracy(model, validation)
            writer.writerow(
                {
                    "step": start_step,
                    "examples_seen": start_step * args.batch_size,
                    "validation_loss": f"{val_loss.item():.6f}",
                    "validation_total_loss": f"{(val_loss + 0.1 * val_rosa).item():.6f}",
                    "validation_rosa_loss": f"{val_rosa.item():.6f}",
                    "validation_token_acc": f"{val_accuracy.item():.6f}",
                    "bad_grad_steps": bad_grad_steps,
                    "elapsed_seconds": f"{time.perf_counter() - start:.1f}",
                }
            )
            log_file.flush()
            print(
                f"resume_eval step={start_step} val_loss={val_loss.item():.4f} "
                f"val_token_acc={val_accuracy.item():.3%}",
                flush=True,
            )

        model.train()
        for step in range(start_step + 1, args.steps + 1):
            batch = make_batch(
                args.batch_size,
                device=DEVICE,
                rng=train_rng,
                digits=args.digits,
                context_len=context_len,
            )
            logits, surrogate_loss = model(batch["input_ids"])
            per_token = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                batch["target_ids"].reshape(-1),
                reduction="none",
            ).view_as(batch["target_ids"])
            task_loss = (per_token * batch["loss_mask"]).sum() / batch["loss_mask"].sum()
            loss = task_loss + 0.1 * surrogate_loss
            if not torch.isfinite(loss).item():
                if args.reproduce_unsafe_nan:
                    raise RuntimeError(
                        f"reproduced non-finite loss at step={step} "
                        f"task_loss={task_loss.item()} rosa_loss={surrogate_loss.item()}"
                    )
                bad_grad_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if bad_grad_steps <= 5 or bad_grad_steps % 25 == 0:
                    print(
                        f"skip_nonfinite_loss step={step} total_skips={bad_grad_steps}",
                        flush=True,
                    )
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.reproduce_unsafe_nan:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                bad_names = [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all().item()
                ]
                if bad_names:
                    bad_grad_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    if bad_grad_steps <= 5 or bad_grad_steps % 25 == 0:
                        print(
                            f"skip_nonfinite_grad step={step} total_skips={bad_grad_steps} "
                            f"params={','.join(bad_names[:8])}",
                            flush=True,
                        )
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            with torch.no_grad():
                token_correct = (
                    logits.argmax(dim=-1).eq(batch["target_ids"]) & batch["loss_mask"]
                ).sum()
                token_count = batch["loss_mask"].sum()
                token_accuracy = token_correct.float() / token_count
            train_loss_window.append(task_loss.item())
            train_total_window.append(loss.item())
            train_rosa_window.append(surrogate_loss.item())
            train_accuracy_window.append(token_accuracy.item())

            is_eval_step = step % args.eval_every == 0 or step == args.steps
            if is_eval_step:
                model.eval()
                with torch.no_grad():
                    val_loss, val_rosa, val_accuracy = losses_and_accuracy(model, validation)

                greedy_exact = ""
                greedy_count = ""
                do_greedy = step % args.greedy_every == 0 or step == args.steps
                if do_greedy:
                    greedy_examples = validation["examples"][: min(args.greedy_size, args.validation_size)]
                    decoded = greedy_sums(model, greedy_examples, context_len=context_len)
                    greedy_exact = sum(
                        predicted == expected and terminated
                        for (_, _, expected), (predicted, terminated) in zip(
                            greedy_examples, decoded
                        )
                    )
                    greedy_count = len(greedy_examples)

                elapsed = time.perf_counter() - start
                train_loss = (
                    sum(train_loss_window) / len(train_loss_window)
                    if train_loss_window
                    else float("nan")
                )
                train_total = (
                    sum(train_total_window) / len(train_total_window)
                    if train_total_window
                    else float("nan")
                )
                train_rosa = (
                    sum(train_rosa_window) / len(train_rosa_window)
                    if train_rosa_window
                    else float("nan")
                )
                train_accuracy = (
                    sum(train_accuracy_window) / len(train_accuracy_window)
                    if train_accuracy_window
                    else float("nan")
                )
                writer.writerow(
                    {
                        "step": step,
                        "examples_seen": step * args.batch_size,
                        "train_loss": f"{train_loss:.6f}",
                        "train_total_loss": f"{train_total:.6f}",
                        "train_rosa_loss": f"{train_rosa:.6f}",
                        "train_token_acc": f"{train_accuracy:.6f}",
                        "validation_loss": f"{val_loss.item():.6f}",
                        "validation_total_loss": f"{(val_loss + 0.1 * val_rosa).item():.6f}",
                        "validation_rosa_loss": f"{val_rosa.item():.6f}",
                        "validation_token_acc": f"{val_accuracy.item():.6f}",
                        "greedy_exact": greedy_exact,
                        "greedy_count": greedy_count,
                        "bad_grad_steps": bad_grad_steps,
                        "elapsed_seconds": f"{elapsed:.1f}",
                    }
                )
                log_file.flush()
                greedy_report = (
                    f" greedy_exact={greedy_exact}/{greedy_count}"
                    if do_greedy
                    else ""
                )
                print(
                    f"step={step}/{args.steps} examples={step * args.batch_size} "
                    f"train_loss={train_loss:.4f} train_rosa={train_rosa:.4f} "
                    f"train_total={train_total:.4f} "
                    f"train_token_acc={train_accuracy:.3%} "
                    f"val_loss={val_loss.item():.4f} val_rosa={val_rosa.item():.4f} "
                    f"val_token_acc={val_accuracy.item():.3%} "
                    f"elapsed={elapsed / 60:.1f}m{greedy_report}",
                    flush=True,
                )
                train_loss_window.clear()
                train_total_window.clear()
                train_rosa_window.clear()
                train_accuracy_window.clear()
                model.train()

            if step % args.save_every == 0 or step == args.steps:
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "train_rng_state": train_rng.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all(),
                        "bad_grad_steps": bad_grad_steps,
                    },
                    checkpoint_path,
                )
                print(f"saved_checkpoint={checkpoint_path} step={step}", flush=True)

    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    print(
        f"finished steps={args.steps} examples_seen={args.steps * args.batch_size} "
        f"elapsed={(time.perf_counter() - start) / 60:.1f}m "
        f"peak_cuda_memory={peak_memory:.2f}GiB metrics={log_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
