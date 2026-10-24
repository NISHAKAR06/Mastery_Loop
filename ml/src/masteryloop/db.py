"""Database helpers for writeback. ml_predictions is created by migration 002."""
import json
import pandas as pd
from sqlalchemy import create_engine, text
from .config import DATABASE_URL

_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


UPSERT = text("""
INSERT INTO ml_predictions (entity_id, step, model_name, model_version, score, label, extra)
VALUES (:entity_id, :step, :model_name, :model_version, :score, :label, CAST(:extra AS JSONB))
ON CONFLICT (entity_id, step, model_name) DO UPDATE
SET model_version = EXCLUDED.model_version, score = EXCLUDED.score,
    label = EXCLUDED.label, extra = EXCLUDED.extra, created_at = now()
""")


def upsert_predictions(df: pd.DataFrame, model_name: str, model_version: str) -> int:
    """df columns: entity_id, step, score, label (nullable), extra (dict)."""
    rows = [{"entity_id": str(r.entity_id), "step": str(r.step), "model_name": model_name,
             "model_version": model_version,
             "score": None if pd.isna(r.score) else float(r.score),
             "label": None if pd.isna(r.label) else str(r.label),
             "extra": json.dumps(r.extra, default=float)} for r in df.itertuples()]
    with engine().begin() as cx:
        for i in range(0, len(rows), 1000):
            cx.execute(UPSERT, rows[i:i + 1000])
    return len(rows)


def run_sql(sql: str, **params) -> int:
    with engine().begin() as cx:
        return cx.execute(text(sql), params).rowcount


def scalar(sql: str, **params):
    with engine().connect() as cx:
        return cx.execute(text(sql), params).scalar()
