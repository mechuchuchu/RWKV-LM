"""Toy check of exact ROSA forward with a differentiable surrogate backward.

The teacher is the same online suffix-automaton lookup used by the 1-bit ROSA
prototype. A small causal Transformer learns its per-position outputs. The
downstream task uses an exact ROSA value in the forward pass and the surrogate
probabilities in the backward pass (straight-through estimator).

Run: python RWKV-v8/rosa_surrogate_toy.py
"""

import random

import torch
from torch import nn
from torch.nn import functional as F


def rosa_qkv_ref(q, k, v):
    """Return ROSA outputs, keeping -1 distinct as the no-match sentinel."""
    n = len(q)
    y = [-1] * n
    size = 2 * n + 1
    transitions = [None] * size
    link = [-1] * size
    max_len = [0] * size
    rightmost_end = [-1] * size
    transitions[0] = {}
    last = 0
    used = 1
    matched_state = matched_len = 0
    assert n == len(k) == len(v)

    for i, (query_symbol, key_symbol) in enumerate(zip(q, k)):
        state, length = matched_state, matched_len
        while state != -1 and query_symbol not in transitions[state]:
            length = max_len[state] if length > max_len[state] else length
            state = link[state]
        if state != -1:
            length += 1
            state = transitions[state][query_symbol]
        else:
            state, length = 0, 0

        candidate = state
        while link[candidate] != -1 and max_len[link[candidate]] >= length:
            candidate = link[candidate]
        while candidate != -1 and (
            max_len[candidate] <= 0 or rightmost_end[candidate] < 0
        ):
            candidate = link[candidate]
        y[i] = v[rightmost_end[candidate] + 1] if candidate != -1 else -1
        matched_state, matched_len = state, length

        new_state = used
        used += 1
        transitions[new_state] = {}
        max_len[new_state] = max_len[last] + 1
        state = last
        while state != -1 and key_symbol not in transitions[state]:
            transitions[state][key_symbol] = new_state
            state = link[state]
        if state == -1:
            link[new_state] = 0
        else:
            next_state = transitions[state][key_symbol]
            if max_len[state] + 1 == max_len[next_state]:
                link[new_state] = next_state
            else:
                clone = used
                used += 1
                transitions[clone] = transitions[next_state].copy()
                max_len[clone] = max_len[state] + 1
                link[clone] = link[next_state]
                rightmost_end[clone] = rightmost_end[next_state]
                link[next_state] = link[new_state] = clone
                while state != -1 and transitions[state][key_symbol] == next_state:
                    transitions[state][key_symbol] = clone
                    state = link[state]

        last = new_state
        state = last
        while state != -1 and rightmost_end[state] < i:
            rightmost_end[state] = i
            state = link[state]

    return y


def make_dataset(count, length, channels, seed, device):
    rng = random.Random(seed)
    q = torch.randint(0, 2, (count, length, channels), dtype=torch.long)
    k = torch.randint(0, 2, (count, length, channels), dtype=torch.long)
    v = torch.randint(0, 2, (count, length, channels), dtype=torch.long)

    # Half the examples contain shifted q/k substrings, producing longer
    # matches as well as the short matches found in independent random strings.
    for b in range(count):
        if b % 2 == 0:
            lag = rng.randint(1, max(2, length // 3))
            k[b, :-lag] = q[b, lag:]

    target = torch.empty((count, length, channels), dtype=torch.long)
    for b in range(count):
        for c in range(channels):
            y = rosa_qkv_ref(q[b, :, c].tolist(), k[b, :, c].tolist(), v[b, :, c].tolist())
            # Match the repository's 1-bit operator: no-match and matched 0
            # both map to 0; matched 1 maps to 1.
            target[b, :, c] = torch.tensor([max(0, symbol) for symbol in y])

    x = torch.cat((q, k, v), dim=-1).float() * 2 - 1
    return x.to(device), target.to(device)


def exact_targets(q, k, v):
    """Run the reference ROSA operator on batched 0/1 tensors."""
    q_cpu, k_cpu, v_cpu = q.detach().cpu(), k.detach().cpu(), v.detach().cpu()
    target = torch.empty_like(q_cpu)
    for b in range(q.shape[0]):
        for c in range(q.shape[2]):
            y = rosa_qkv_ref(q_cpu[b, :, c].tolist(), k_cpu[b, :, c].tolist(), v_cpu[b, :, c].tolist())
            target[b, :, c] = torch.tensor([max(0, symbol) for symbol in y])
    return target.to(q.device)


class SurrogateToy(nn.Module):
    def __init__(self, channels, width=64, layers=2):
        super().__init__()
        self.channels = channels
        self.input = nn.Linear(3 * channels, width)
        self.position = nn.Embedding(128, width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.output = nn.Linear(width, channels * 2)
        self.task = nn.Sequential(
            nn.Linear(channels * 2, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        length = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
        )
        positions = torch.arange(length, device=x.device)
        h = self.transformer(self.input(x) + self.position(positions), mask=causal_mask)
        return self.output(h).view(x.shape[0], length, self.channels, 2)


def task_label(target):
    # A tiny downstream task: XOR whether each of two ROSA channels returned
    # 1 at the final position.
    bit0 = target[:, -1, 0].eq(1)
    bit1 = target[:, -1, 1].eq(1)
    return torch.logical_xor(bit0, bit1).long()


@torch.no_grad()
def evaluate(model, x, target, batch_size=128):
    model.eval()
    proxy_correct = proxy_total = 0
    exact_task_correct = proxy_task_correct = task_total = 0
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size]
        tb = target[start : start + batch_size]
        logits = model(xb)
        proxy_pred = logits.argmax(dim=-1)
        proxy_correct += proxy_pred.eq(tb).sum().item()
        proxy_total += tb.numel()

        exact_hot = F.one_hot(tb, num_classes=2).float()
        exact_task = model.task(exact_hot[:, -1].flatten(1))
        proxy_hot = F.one_hot(proxy_pred, num_classes=2).float()
        proxy_task = model.task(proxy_hot[:, -1].flatten(1))
        labels = task_label(tb)
        exact_task_correct += exact_task.argmax(-1).eq(labels).sum().item()
        proxy_task_correct += proxy_task.argmax(-1).eq(labels).sum().item()
        task_total += labels.numel()

    return (
        proxy_correct / proxy_total,
        exact_task_correct / task_total,
        proxy_task_correct / task_total,
    )


def optimize_upstream_projection(model, device, read_index=6, steps=120):
    """Check whether the surrogate gradient can train an upstream v projection."""
    length, channels = 12, 2
    q_pattern = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0]
    k_pattern = [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0]

    # This q/k pair makes the final ROSA output read v[read_index]. Verify the
    # source position directly by probing with one-hot value sequences.
    selected = []
    for i in range(length):
        value = [0] * length
        value[i] = 1
        if rosa_qkv_ref(q_pattern, k_pattern, value)[-1] == 1:
            selected.append(i)
    assert selected == [read_index], selected

    q = torch.tensor(q_pattern, device=device).view(1, length, 1).expand(128, -1, channels).clone()
    k = torch.tensor(k_pattern, device=device).view(1, length, 1).expand(128, -1, channels).clone()
    raw_valid = torch.randn(32, length, channels, device=device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    alpha = nn.Parameter(torch.tensor(-2.0, device=device))
    bias = nn.Parameter(torch.tensor(0.0, device=device))
    optimizer = torch.optim.Adam([alpha, bias], lr=0.08)

    def exact_accuracy(raw_v):
        with torch.no_grad():
            logits = alpha * raw_v + bias
            hard_v = logits.gt(0).long()
            target = exact_targets(q[: raw_v.shape[0]], k[: raw_v.shape[0]], hard_v)
            labels = raw_v[:, read_index, 0].gt(0).long()
            return target[:, -1, 0].eq(labels).float().mean().item()

    before = exact_accuracy(raw_valid)
    for step in range(steps):
        raw_v = torch.randn(128, length, channels, device=device)
        labels = raw_v[:, read_index, 0].gt(0).long()
        v_probability = (alpha * raw_v + bias).sigmoid()
        v_hard = v_probability.detach().gt(0.5).long()
        v_soft_input = 2 * v_probability - 1
        v_hard_input = 2 * v_hard.float() - 1
        v_st = v_hard_input + v_soft_input - v_soft_input.detach()

        target = exact_targets(q, k, v_hard)
        x = torch.cat((q.float() * 2 - 1, k.float() * 2 - 1, v_st), dim=-1)
        probabilities = model(x).softmax(dim=-1)
        exact = F.one_hot(target, num_classes=2).to(probabilities.dtype)
        y_st = exact + probabilities - probabilities.detach()
        task_logits = 8 * y_st[:, -1, 0, :]
        loss = F.cross_entropy(task_logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    after = exact_accuracy(raw_valid)
    print(
        f"upstream_v_projection: exact_task={before:.3%}->{after:.3%}, "
        f"alpha={alpha.item():.3f}, bias={bias.item():.3f}"
    )


def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels, length = 2, 12
    train_x, train_y = make_dataset(4096, length, channels, seed, device)
    valid_x, valid_y = make_dataset(512, length, channels, seed + 1, device)
    model = SurrogateToy(channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps, batch_size, distill_weight = 1000, 128, 1.0

    print(f"device={device}, train={len(train_x)}, valid={len(valid_x)}, T={length}, C={channels}")
    for step in range(1, steps + 1):
        model.train()
        ids = torch.randint(0, train_x.shape[0], (batch_size,), device=device)
        xb, tb = train_x[ids], train_y[ids]
        logits = model(xb)
        probs = logits.softmax(dim=-1)

        # Exact ROSA values are the forward activations. The surrogate supplies
        # the backward path, and the auxiliary CE trains it to match ROSA.
        exact = F.one_hot(tb, num_classes=2).to(probs.dtype)
        rosa_with_surrogate_grad = exact + probs - probs.detach()
        task_logits = model.task(rosa_with_surrogate_grad[:, -1].flatten(1))
        loss_task = F.cross_entropy(task_logits, task_label(tb))
        loss_distill = F.cross_entropy(logits.flatten(0, 2), tb.flatten())
        loss = loss_task + distill_weight * loss_distill

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step in (1, 100, 200, 400, 600, 800, steps):
            match_acc, exact_task_acc, proxy_task_acc = evaluate(model, valid_x, valid_y)
            print(
                f"step={step:3d} loss={loss.item():.4f} "
                f"distill={loss_distill.item():.4f} "
                f"proxy_match={match_acc:.3%} "
                f"exact_ROSA_task={exact_task_acc:.3%} "
                f"proxy_task={proxy_task_acc:.3%}"
            )

    optimize_upstream_projection(model, device)


if __name__ == "__main__":
    main()
