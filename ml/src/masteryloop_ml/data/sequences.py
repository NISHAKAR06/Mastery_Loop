"""Sequence extraction and utilities for knowledge tracing."""
from __future__ import annotations
import pandas as pd

def extract_sequences(interactions_df: pd.DataFrame) -> dict[str, list[dict]]:
    """Group interactions DataFrame into a dictionary of lists of interaction dicts per learner."""
    by_learner: dict[str, list[dict]] = {}
    sorted_df = interactions_df.sort_values(["learner_id", "ts"])
    for row in sorted_df.to_dict(orient="records"):
        lid = str(row["learner_id"])
        by_learner.setdefault(lid, []).append(row)
    return by_learner

