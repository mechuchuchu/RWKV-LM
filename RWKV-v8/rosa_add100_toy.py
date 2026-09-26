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


def make_batch(batch_size: int, device="cpu", rng=None):
    """Create random exact-100-digit additions and next-token LM targets.

    Each input is ``A+B=`` followed by the correct sum and ``=`` padding, as in
    the existing RWKV-8 arithmetic demo. The loss mask selects only sum digits
    and the first ``=`` after the sum.
    """
    if rng is None:
        rng = random

    inputs = torch.full((batch_size, MAX_SEQUENCE_LEN - 1), EQUALS, dtype=torch.long)
    targets = torch.full_like(inputs, EQUALS)
    loss_mask = torch.zeros_like(inputs, dtype=torch.bool)
    examples = []

    lower = 10 ** (DIGITS - 1)
    upper = 10**DIGITS
    for row in range(batch_size):
        a = rng.randrange(lower, upper)
        b = rng.randrange(lower, upper)
        a_text, b_text = str(a), str(b)
        sum_text = str(a + b)

        prompt = encode_digits(a_text) + [PLUS] + encode_digits(b_text) + [EQUALS]
        answer = encode_digits(sum_text)
        sequence = prompt + answer
        sequence += [EQUALS] * (MAX_SEQUENCE_LEN - len(sequence))

        inputs[row] = torch.tensor(sequence[:-1], dtype=torch.long)
        targets[row] = torch.tensor(sequence[1:], dtype=torch.long)
        first_answer_target = PROMPT_LEN - 1
        after_answer_target = PROMPT_LEN + len(answer)
        loss_mask[row, first_answer_target:after_answer_target] = True
        examples.append((a_text, b_text, sum_text))

    return {
        "input_ids": inputs.to(device),
        "target_ids": targets.to(device),
        "loss_mask": loss_mask.to(device),
        "examples": examples,
    }


def self_check(samples: int, seed: int):
    batch = make_batch(samples, rng=random.Random(seed))
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    loss_mask = batch["loss_mask"]
    observed_lengths = {DIGITS: 0, DIGITS + 1: 0}

    assert input_ids.shape == (samples, MAX_SEQUENCE_LEN - 1)
    assert target_ids.shape == input_ids.shape
    assert loss_mask.shape == input_ids.shape

    for row, (a_text, b_text, sum_text) in enumerate(batch["examples"]):
        assert len(a_text) == len(b_text) == DIGITS
        assert a_text[0] != "0" and b_text[0] != "0"
        assert int(a_text) + int(b_text) == int(sum_text)
        assert len(sum_text) in observed_lengths
        observed_lengths[len(sum_text)] += 1

        active_positions = torch.where(loss_mask[row])[0]
        expected_count = len(sum_text) + 1  # sum digits plus terminator '='
        assert active_positions.numel() == expected_count
        assert active_positions[0].item() == PROMPT_LEN - 1
        assert active_positions[-1].item() == PROMPT_LEN + len(sum_text) - 1
        assert not loss_mask[row, : PROMPT_LEN - 1].any()
        assert not loss_mask[row, PROMPT_LEN + len(sum_text) :].any()

        predicted = target_ids[row, active_positions].tolist()
        assert predicted[:-1] == encode_digits(sum_text)
        assert predicted[-1] == EQUALS

    assert observed_lengths[DIGITS] > 0 and observed_lengths[DIGITS + 1] > 0
    print(
        f"OK: {samples} examples; both operands are {DIGITS} digits; "
        f"sum lengths={observed_lengths}; input context={CONTEXT_LEN}; "
        f"input shape={tuple(input_ids.shape)}; loss mask covers sum + terminator only"
    )
    print("Example:", batch["examples"][0][0], "+", batch["examples"][0][1], "=", batch["examples"][0][2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="verify generated examples and masks")
    parser.add_argument("--samples", type=int, default=256, help="number of examples to check")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.self_check:
        self_check(args.samples, args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
