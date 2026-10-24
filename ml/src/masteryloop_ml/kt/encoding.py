"""Sequence encoding for SAKT - deliberately torch-free.
This is the most leak-prone code in the project: an off-by-one here hands the
model the answer to the question it is predicting, the AUC jumps to 0.99, and
nothing raises. So it lives in its own module with no heavy dependency, and the
test that proves interactions[t] precedes next_items[t] runs on any laptop with
numpy installed.
"""
from __future__ import annotations
import numpy as np

PAD = 0

def encode_interaction(item_index: int, correct: bool, n_items: int) -> int:
    """(item, correct) -> one symbol. +1 throughout because 0 is PAD, and an
    off-by-one here maps every interaction to the wrong embedding row without
    raising anything."""
    return int(item_index) + 1 + (n_items if correct else 0)

def encode_sequence(items, corrects, n_items: int, max_len: int = 200):
    """Returns (interactions, next_items, targets), all left-padded to max_len.
    interactions[t] is the interaction BEFORE next_items[t], which is the whole
    contract the causal mask depends on."""
    items = [int(i) for i in items][-max_len:]
    corrects = [bool(c) for c in corrects][-len(items):]
    inter = [PAD] + [encode_interaction(i, c, n_items)
                     for i, c in zip(items[:-1], corrects[:-1])]
    nxt = [i + 1 for i in items]
    targets = [int(c) for c in corrects]
    pad = max_len - len(nxt)
    return (np.array([PAD] * pad + inter, dtype=np.int64),
            np.array([PAD] * pad + nxt, dtype=np.int64),
            np.array([-100] * pad + targets, dtype=np.int64))
