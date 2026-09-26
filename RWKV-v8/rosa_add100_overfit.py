"""Overfit four fixed 100-digit additions for a short training smoke test."""

import argparse
import random

import torch
from torch.nn import functional as F

from rosa_add100_toy import (
    CONTEXT_LEN,
    DIGITS,
    EQUALS,
    PLUS,
    VOCAB_SIZE,
    make_batch,
)
from rosa_add100_smoke import (
    AdditionModel,
    DEVICE,
    CONTEXT_LEN,
    ROSA_DROPOUT,
    exact_rosa_targets,
    load_wkv7_kernel,
)

LEARNING_RATE = 1e-4


def losses_and_accuracy(model, batch):
    logits, surrogate_loss = model(batch["input_ids"])
    per_token = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        batch["target_ids"].reshape(-1),
        reduction="none",
    ).view_as(batch["target_ids"])
    task_loss = (per_token * batch["loss_mask"]).sum() / batch["loss_mask"].sum()
    correct = logits.argmax(dim=-1).eq(batch["target_ids"]) & batch["loss_mask"]
    accuracy = correct.sum().float() / batch["loss_mask"].sum()
    return task_loss, surrogate_loss, accuracy


@torch.no_grad()
def greedy_sums(model, examples):
    """Autoregressively decode the answer digits, stopping at '='."""
    prompts = [
        [int(char) for char in a] + [PLUS] + [int(char) for char in b] + [EQUALS]
        for a, b, _ in examples
    ]
    sequences = [prompt[:] for prompt in prompts]
    finished = [False] * len(examples)

    for _ in range(DIGITS + 2):  # allow a carry digit and the '=' terminator
        input_ids = torch.full(
            (len(sequences), CONTEXT_LEN), EQUALS, dtype=torch.long, device=DEVICE
        )
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device=DEVICE
            )
        logits, _ = model(input_ids)
        next_tokens = logits[
            torch.arange(len(sequences), device=DEVICE),
            torch.tensor([len(sequence) - 1 for sequence in sequences], device=DEVICE),
        ].argmax(dim=-1).tolist()

        for row, token in enumerate(next_tokens):
            if not finished[row]:
                sequences[row].append(token)
                if token == EQUALS:
                    finished[row] = True
        if all(finished):
            break

    decoded = []
    for row, sequence in enumerate(sequences):
        answer_tokens = sequence[len(prompts[row]) :]
        terminated = EQUALS in answer_tokens
        if terminated:
            answer_tokens = answer_tokens[: answer_tokens.index(EQUALS)]
        digits = "".join(str(token) for token in answer_tokens if 0 <= token <= 9)
        decoded.append((digits, terminated))
    return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rosa-dropout", type=float, default=ROSA_DROPOUT)
    parser.add_argument("--rosa-sign-flip", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("This overfit check needs a CUDA GPU for the RWKV7 kernel.")
    torch.manual_seed(321)
    torch.cuda.manual_seed_all(321)
    load_wkv7_kernel()
    warmup = torch.zeros((1, 1, 1), device=DEVICE)
    exact_rosa_targets(warmup, warmup, warmup)

    batch = make_batch(4, device=DEVICE, rng=random.Random(321))
    model = AdditionModel(
        rosa_dropout=args.rosa_dropout,
        rosa_sign_flip_p=args.rosa_sign_flip,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats()

    model.eval()
    with torch.no_grad():
        task_loss, surrogate_loss, accuracy = losses_and_accuracy(model, batch)
    print(
        f"step=0 task_loss={task_loss.item():.4f} "
        f"rosa_loss={surrogate_loss.item():.4f} token_acc={accuracy.item():.3%}"
    )

    model.train()
    checkpoints = {args.steps} | {step for step in (20, 50, 100) if step < args.steps}
    last_decoded = None
    for step in range(1, args.steps + 1):
        logits, surrogate_loss = model(batch["input_ids"])
        per_token = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            batch["target_ids"].reshape(-1),
            reduction="none",
        ).view_as(batch["target_ids"])
        task_loss = (per_token * batch["loss_mask"]).sum() / batch["loss_mask"].sum()
        loss = task_loss + 0.1 * surrogate_loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"non-finite training loss at step={step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step in checkpoints:
            model.eval()
            with torch.no_grad():
                task_loss, surrogate_loss, accuracy = losses_and_accuracy(model, batch)
            last_decoded = greedy_sums(model, batch["examples"])
            greedy_exact = sum(
                predicted == expected and terminated
                for (_, _, expected), (predicted, terminated) in zip(
                    batch["examples"], last_decoded
                )
            )
            print(
                f"step={step:2d} task_loss={task_loss.item():.4f} "
                f"rosa_loss={surrogate_loss.item():.4f} token_acc={accuracy.item():.3%} "
                f"greedy_exact={greedy_exact}/{len(batch['examples'])}"
            )
            model.train()

    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    print(
        f"finished {args.steps} fixed-batch steps; context={CONTEXT_LEN}; "
        f"rosa_dropout={args.rosa_dropout:g}; "
        f"rosa_sign_flip={args.rosa_sign_flip:g}; lr={LEARNING_RATE:g}; "
        f"peak_cuda_memory={peak_memory:.2f} GiB"
    )
    model.eval()
    assert last_decoded is not None
    exact_matches = 0
    for index, ((a, b, expected), (predicted, terminated)) in enumerate(
        zip(batch["examples"], last_decoded), start=1
    ):
        exact = predicted == expected and terminated
        exact_matches += exact
        print(
            f"greedy {index}: {a} + {b} = {predicted or '<empty>'} "
            f"({'exact' if exact else 'wrong'}; terminated={terminated}; expected={expected})"
        )
    print(f"greedy exact sums: {exact_matches}/{len(batch['examples'])}")


if __name__ == "__main__":
    main()
