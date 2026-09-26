"""Data harness for fixed-width 100-digit addition with a ROSA model."""

import argparse
import random

import torch


DIGITS = 100
PLUS, MINUS, EQUALS = 10, 11, 12
VOCAB_SIZE = 13
PROMPT_LEN = 2 * DIGITS + 2  # A + B =
MAX_SUM_LEN = DIGITS + 1
CONTEXT_LEN = 304  # divisible by the RWKV7 CUDA kernel's chunk length (16)
MAX_SEQUENCE_LEN = CONTEXT_LEN + 1  # input plus next-token target


def encode_digits(number: str) -> list[int]:
    return [ord(char) - ord("0") for char in number]


def make_batch(batch_size: int, device="cpu", rng=None, digits=DIGITS, context_len=CONTEXT_LEN):
    """Create random fixed-width additions and next-token LM targets.

    Each input is ``A+B=`` followed by the correct sum and ``=`` padding, as in
    the existing RWKV-8 arithmetic demo. The loss mask selects only sum digits
    and the first ``=`` after the sum. ``context_len`` must fit the longest
    possible sum and be divisible by the RWKV7 CUDA chunk length (16).
    """
    if rng is None:
        rng = random
    if digits < 1:
        raise ValueError("digits must be at least 1")
    prompt_len = 2 * digits + 2
    max_sum_len = digits + 1
    max_sequence_len = context_len + 1
    if context_len % 16 != 0:
        raise ValueError("context_len must be divisible by 16")
    if context_len < prompt_len + max_sum_len:
        raise ValueError(
            f"context_len={context_len} cannot fit the longest {digits}-digit sum; "
            f"need at least {prompt_len + max_sum_len}"
        )

    inputs = torch.full((batch_size, max_sequence_len - 1), EQUALS, dtype=torch.long)
    targets = torch.full_like(inputs, EQUALS)
    loss_mask = torch.zeros_like(inputs, dtype=torch.bool)
    examples = []

    lower = 10 ** (digits - 1)
    upper = 10**digits
    for row in range(batch_size):
        a = rng.randrange(lower, upper)
        b = rng.randrange(lower, upper)
        a_text, b_text = str(a), str(b)
        sum_text = str(a + b)

        prompt = encode_digits(a_text) + [PLUS] + encode_digits(b_text) + [EQUALS]
        answer = encode_digits(sum_text)
        sequence = prompt + answer
        sequence += [EQUALS] * (max_sequence_len - len(sequence))

        inputs[row] = torch.tensor(sequence[:-1], dtype=torch.long)
        targets[row] = torch.tensor(sequence[1:], dtype=torch.long)
        first_answer_target = prompt_len - 1
        after_answer_target = prompt_len + len(answer)
        loss_mask[row, first_answer_target:after_answer_target] = True
        examples.append((a_text, b_text, sum_text))

    return {
        "input_ids": inputs.to(device),
        "target_ids": targets.to(device),
        "loss_mask": loss_mask.to(device),
        "examples": examples,
    }


def self_check(samples: int, seed: int, digits=DIGITS, context_len=None):
    if context_len is None:
        minimum_context = 3 * digits + 3
        context_len = ((minimum_context + 15) // 16) * 16
    prompt_len = 2 * digits + 2
    batch = make_batch(
        samples, rng=random.Random(seed), digits=digits, context_len=context_len
    )
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    loss_mask = batch["loss_mask"]
    observed_lengths = {digits: 0, digits + 1: 0}

    assert input_ids.shape == (samples, context_len)
    assert target_ids.shape == input_ids.shape
    assert loss_mask.shape == input_ids.shape

    for row, (a_text, b_text, sum_text) in enumerate(batch["examples"]):
        assert len(a_text) == len(b_text) == digits
        assert a_text[0] != "0" and b_text[0] != "0"
        assert int(a_text) + int(b_text) == int(sum_text)
        assert len(sum_text) in observed_lengths
        observed_lengths[len(sum_text)] += 1

        active_positions = torch.where(loss_mask[row])[0]
        expected_count = len(sum_text) + 1  # sum digits plus terminator '='
        assert active_positions.numel() == expected_count
        assert active_positions[0].item() == prompt_len - 1
        assert active_positions[-1].item() == prompt_len + len(sum_text) - 1
        assert not loss_mask[row, : prompt_len - 1].any()
        assert not loss_mask[row, prompt_len + len(sum_text) :].any()

        predicted = target_ids[row, active_positions].tolist()
        assert predicted[:-1] == encode_digits(sum_text)
        assert predicted[-1] == EQUALS

    assert observed_lengths[digits] > 0 and observed_lengths[digits + 1] > 0
    print(
        f"OK: {samples} examples; both operands are {digits} digits; "
        f"sum lengths={observed_lengths}; input context={context_len}; "
        f"input shape={tuple(input_ids.shape)}; loss mask covers sum + terminator only"
    )
    print("Example:", batch["examples"][0][0], "+", batch["examples"][0][1], "=", batch["examples"][0][2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="verify generated examples and masks")
    parser.add_argument("--samples", type=int, default=256, help="number of examples to check")
    parser.add_argument("--digits", type=int, default=DIGITS, help="digits per operand")
    parser.add_argument(
        "--context-len", type=int, help="model input length (default: smallest multiple of 16 that fits)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.self_check:
        self_check(args.samples, args.seed, args.digits, args.context_len)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
