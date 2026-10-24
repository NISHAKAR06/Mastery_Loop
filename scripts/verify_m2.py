"""M2 definition-of-done checks. Exit code 0 means ready to tag v0.2.0-M2."""
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ml" / "src"))

from masteryloop.config import ROOT, REPORTS, MODEL_NAME  # noqa: E402
from masteryloop import db, tracking  # noqa: E402
from masteryloop.promote import check  # noqa: E402

FILES = [
    'ml/src/masteryloop/config.py',
    'ml/src/masteryloop/tracking.py',
    'ml/src/masteryloop/db.py',
    'ml/src/masteryloop/split.py',
    'ml/src/masteryloop/features.py',
    'ml/src/masteryloop/baseline.py',
    'ml/src/masteryloop/train.py',
    'ml/src/masteryloop/hpo.py',
    'ml/src/masteryloop/evaluate.py',
    'ml/src/masteryloop/promote.py',
    'ml/src/masteryloop/writeback.py',
    'configs/train.yaml',
    'configs/gates.yaml',
    'docs/model_card.md',
    'db/migrations/002_m2_predictions.sql',
    'ml/tests/test_m2_leakage.py',
]

FILLED_SQL = """SELECT count(*) FROM items WHERE a IS NOT NULL"""
results = []


def ok(name, cond, detail=""):
    results.append(cond)
    print(f"[{'OK ' if cond else 'XX '}] {name} {detail}")


for f in FILES:
    ok(f"file {f}", (ROOT / f).exists())

env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT / "ml" / "src")
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "ml/tests/test_m2_leakage.py"],
                   cwd=ROOT, env=env)
ok("leakage tests", r.returncode == 0)

rep = REPORTS / "eval.json"
ok("evaluation report", rep.exists())
if rep.exists():
    for path, val, op, thr, passed in check(json.loads(rep.read_text())):
        ok(f"gate {path}", passed, f"{val:.4f} {op} {thr}")

try:
    label, _ = tracking.champion()
    ok("registry champion", True, label)
except Exception as e:  # noqa: BLE001
    ok("registry champion", False, str(e)[:80])

try:
    n_pred = db.scalar("SELECT count(*) FROM ml_predictions WHERE model_name = :m", m=MODEL_NAME)
    ok("ml_predictions rows", (n_pred or 0) > 0, str(n_pred))
    n_fill = db.scalar(FILLED_SQL)
    ok("M1 nullable column filled", (n_fill or 0) > 0, str(n_fill))
except Exception as e:  # noqa: BLE001
    ok("database writeback", False, str(e)[:80])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
