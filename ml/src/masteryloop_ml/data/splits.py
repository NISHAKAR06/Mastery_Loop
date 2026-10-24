"""Leakage-safe splits for knowledge tracing (FR-4).
Knowledge tracing leaks in a way most ML does not, and the leak is two-layered:
 1. a random INTERACTION split puts a learner's answer to item 40 in train and
 their answer to item 41 in test. The model learns the learner, scores
 beautifully, and predicts nothing on someone new;
 2. splitting by learner fixes that and is still not enough. The evaluation
 question is "given everything up to time t, will they get the NEXT item
 right", so a validation sequence must be TRUNCATED at its origin. Feed the
 whole sequence and the model attends to answers that had not happened yet.
Both are handled here, and the second has a poisoned-future test.
"""
from __future__ import annotations
import hashlib

VAL_FRACTION = 0.2

def split_of(learner_id, val_fraction: float = VAL_FRACTION, salt: str = "") -> str:
    """Hash-stable, so a learner never changes halves between runs or machines."""
    h = hashlib.sha1(f"{salt}{learner_id}".encode()).hexdigest()
    return "val" if int(h[:8], 16) / 0xFFFFFFFF < val_fraction else "train"

def assign(learner_ids, val_fraction: float = VAL_FRACTION, salt: str = "") -> dict:
    return {lid: split_of(lid, val_fraction, salt)
            for lid in dict.fromkeys(learner_ids)}

def check_no_learner_leak(train_ids, val_ids) -> list[str]:
    tr, va = set(map(str, train_ids)), set(map(str, val_ids))
    problems = []
    shared = tr & va
    if shared:
        problems.append(f"{len(shared)} learners in both splits: "
                        f"{sorted(shared)[:5]}")
    if not tr or not va:
        problems.append("train or validation split is empty")
    return problems

def truncate_at_origin(sequence, origin: int) -> dict:
    """Everything strictly BEFORE `origin` is the model's input; the interaction
    AT origin is the thing being predicted. Returning both halves explicitly
    means no caller can accidentally hand the target to the encoder."""
    if not 0 < origin < len(sequence):
        raise ValueError(f"origin {origin} must be inside the sequence "
                         f"(1..{len(sequence) - 1})")
    return {"history": list(sequence[:origin]), "target": sequence[origin]}

def evaluation_points(sequence, min_history: int = 5, stride: int = 1) -> list[dict]:
    """Every position in a validation learner's sequence with enough history
    behind it. One learner contributes many prediction points, which is what
    makes a few thousand held-out learners a usable test set - but every point
    is still truncated, so none of them sees its own future."""
    return [truncate_at_origin(sequence, o)
            for o in range(min_history, len(sequence), stride)]

def build_split(interactions, val_fraction: float = VAL_FRACTION,
                min_history: int = 5) -> dict:
    """interactions: [{'learner_id','item_id','skill_id','correct','ts'}].
    Returns training sequences (whole) and validation prediction points
    (truncated), plus the report the leakage test asserts on."""
    by_learner: dict = {}
    for row in sorted(interactions, key=lambda r: (str(r["learner_id"]), r["ts"])):
        by_learner.setdefault(str(row["learner_id"]), []).append(row)
    split = assign(by_learner)
    train = {lid: seq for lid, seq in by_learner.items() if split[lid] == "train"}
    val = {lid: seq for lid, seq in by_learner.items() if split[lid] == "val"}
    problems = check_no_learner_leak(train, val)
    if problems:
        raise AssertionError("learner leakage: " + "; ".join(problems))
    points = []
    for lid, seq in val.items():
        if len(seq) > min_history:
            for p in evaluation_points(seq, min_history):
                points.append({"learner_id": lid, **p})
    return {"train_sequences": train, "val_points": points,
            "report": {"train_learners": len(train), "val_learners": len(val),
                       "val_points": len(points)}}
