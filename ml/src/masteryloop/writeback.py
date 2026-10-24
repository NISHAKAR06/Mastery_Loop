"""Publish the champion's calibration: every item's (a, b) into ml_predictions, the
M1 items table's nullable a / b / param_version columns, and a versioned row in
item_param_history - the audit trail misfit detection and the generated-item
acceptance gate will read from M3 onward."""
import joblib
import mlflow
import pandas as pd
from sqlalchemy import text
from .config import PROCESSED, MODEL_NAME, MLFLOW_URI
from . import db, tracking

# M1 seeded items.item_ref as 'ML-I-<problem_id>'. Adjust if your M1 differs.
ENTITY = "ML-I-{}"
FILL_SQL = """
UPDATE items i
SET a = p.score,
    b = (p.extra->>'b')::real,
    param_version = (p.extra->>'version')::int
FROM ml_predictions p
WHERE p.model_name = :model AND i.item_ref = p.entity_id
"""
HISTORY_SQL = """
INSERT INTO item_param_history (item_id, version, a, b, n_responses, fitted_at)
SELECT i.item_id, (p.extra->>'version')::int, p.score, (p.extra->>'b')::real,
       (p.extra->>'n')::int, now()
FROM ml_predictions p JOIN items i ON i.item_ref = p.entity_id
WHERE p.model_name = :model
ON CONFLICT DO NOTHING
"""


def main():
    label, run_id = tracking.champion()
    mlflow.set_tracking_uri(MLFLOW_URI)
    b = joblib.load(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="bundle/model.joblib"))
    items, version = b["items"], b["param_version"]
    out = pd.DataFrame({
        "entity_id": items.problem_id.astype(str).map(ENTITY.format),
        "step": f"2pl-v{version}",
        "score": items.a.astype(float),
        "label": None})
    out["extra"] = [{"b": round(float(r.b), 4), "n": int(r.n_responses),
                     "version": version} for r in items.itertuples()]
    n = db.upsert_predictions(out, MODEL_NAME, label)

    # Ensure items have item_ref mapped if null from M1
    with db.engine().begin() as cx:
        res = cx.execute(text("SELECT item_id FROM items WHERE item_ref IS NULL ORDER BY item_code")).fetchall()
        if res:
            for (item_id,), pid in zip(res, items.problem_id):
                cx.execute(text("UPDATE items SET item_ref = :ref WHERE item_id = :item_id"),
                           {"ref": f"ML-I-{pid}", "item_id": item_id})

    filled = db.run_sql(FILL_SQL, model=MODEL_NAME)
    hist = db.run_sql(HISTORY_SQL, model=MODEL_NAME)
    print(f"{n} item calibrations stored as {label}; {filled} items updated; "
          f"{hist} history rows written")


if __name__ == "__main__":
    main()
