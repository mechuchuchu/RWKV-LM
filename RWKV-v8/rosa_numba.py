"""Numba-compiled exact ROSA lookup for binary Q/K/V sequences."""

import numpy as np
from numba import njit


@njit(cache=True)
def _rosa_qkv_one(q, k, values):
    n = q.shape[0]
    state_capacity = 2 * n + 1
    transitions = np.full((state_capacity, 2), -1, dtype=np.int64)
    suffix_link = np.full(state_capacity, -1, dtype=np.int64)
    max_length = np.zeros(state_capacity, dtype=np.int64)
    rightmost_end = np.full(state_capacity, -1, dtype=np.int64)
    output = np.zeros(n, dtype=np.uint8)

    last_state = 0
    used_states = 1
    matched_state = 0
    matched_length = 0

    for i in range(n):
        query_symbol = int(q[i])
        key_symbol = int(k[i])

        state = matched_state
        length = matched_length
        while state != -1 and transitions[state, query_symbol] == -1:
            if max_length[state] > length:
                length = max_length[state]
            state = suffix_link[state]

        if state != -1:
            length += 1
            state = transitions[state, query_symbol]
        else:
            state = 0
            length = 0

        candidate = state
        while suffix_link[candidate] != -1 and max_length[suffix_link[candidate]] >= length:
            candidate = suffix_link[candidate]
        while candidate != -1 and (max_length[candidate] <= 0 or rightmost_end[candidate] < 0):
            candidate = suffix_link[candidate]

        if candidate != -1:
            output[i] = values[rightmost_end[candidate] + 1]

        matched_state = state
        matched_length = length

        new_state = used_states
        used_states += 1
        max_length[new_state] = max_length[last_state] + 1
        state = last_state

        while state != -1 and transitions[state, key_symbol] == -1:
            transitions[state, key_symbol] = new_state
            state = suffix_link[state]

        if state == -1:
            suffix_link[new_state] = 0
        else:
            next_state = transitions[state, key_symbol]
            if max_length[state] + 1 == max_length[next_state]:
                suffix_link[new_state] = next_state
            else:
                clone = used_states
                used_states += 1
                transitions[clone, :] = transitions[next_state, :]
                max_length[clone] = max_length[state] + 1
                suffix_link[clone] = suffix_link[next_state]
                rightmost_end[clone] = rightmost_end[next_state]
                suffix_link[next_state] = clone
                suffix_link[new_state] = clone
                while state != -1 and transitions[state, key_symbol] == next_state:
                    transitions[state, key_symbol] = clone
                    state = suffix_link[state]

        last_state = new_state
        state = last_state
        while state != -1 and rightmost_end[state] < i:
            rightmost_end[state] = i
            state = suffix_link[state]

    return output


@njit(cache=True)
def _rosa_qkv_batch(q, k, values):
    rows, length = q.shape
    output = np.empty((rows, length), dtype=np.uint8)
    for row in range(rows):
        output[row] = _rosa_qkv_one(q[row], k[row], values[row])
    return output


def rosa_qkv_batch_numba(q, k, values):
    """Run exact ROSA on contiguous uint8 arrays shaped [batch*channels, T]."""
    if q.dtype != np.uint8 or k.dtype != np.uint8 or values.dtype != np.uint8:
        raise TypeError("q, k, and values must be numpy.uint8 arrays")
    if q.ndim != 2 or q.shape != k.shape or q.shape != values.shape:
        raise ValueError("q, k, and values must share shape [rows, T]")
    return _rosa_qkv_batch(
        np.ascontiguousarray(q),
        np.ascontiguousarray(k),
        np.ascontiguousarray(values),
    )
