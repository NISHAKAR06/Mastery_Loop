"""M3 definition-of-done checks. Exit 0 means ready to tag v0.3.0-M3."""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = [
    "ml/src/masteryloop_ml/data/splits.py",
    "ml/src/masteryloop_ml/kt/bkt_baseline.py",
    "ml/src/masteryloop_ml/kt/encoding.py",
    "ml/src/masteryloop_ml/kt/sakt.py",
    "ml/src/masteryloop_ml/kt/evaluate.py",
    "services/kt/src/kt_svc/mastery.py",
    "services/kt/src/kt_svc/updater.py",
    "services/kt/src/kt_svc/consumer.py",
    "services/kt/src/kt_svc/main.py",
    "db/migrations/versions/m3_0001_mastery.py",
    "docs/adr/0003-m3-knowledge-tracing.md",
    "reports/m3_kt.json",
]
results = []
def ok(name, cond, detail=""):
    results.append(bool(cond))
    print(f"[{'OK ' if cond else 'XX '}] {name} {detail}")

def run(args):
    sep = ";" if os.name == "nt" else ":"
    env = {**os.environ,
           "PYTHONPATH": sep.join(["ml/src", "libs/psychocore/src", "services/kt/src", "."])}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env).returncode

for f in FILES:
    ok(f"file {f}", (ROOT / f).exists())

ok("unit tests", run(["-m", "pytest", "-q", "tests/m3/unit"]) == 0)

report = ROOT / "reports/m3_kt.json"
if report.exists():
    r = json.loads(report.read_text())
    ok("SAKT AUC >= 0.77 (objective 5.1)", r["gates"][0]["pass"],
       r["gates"][0]["detail"])
    ok("SAKT beats BKT by >= 4 points", r["gates"][1]["pass"],
       r["gates"][1]["detail"])
    ok("a recommendation was recorded either way", "recommendation" in r,
       r.get("recommendation", ""))
    ok("calibration reported, not just AUC",
       r["sakt_calibration"]["ece"] <= 0.10,
       f"(ECE {r['sakt_calibration']['ece']:.4f})")
    per_skill = r.get("sakt_per_skill", {})
    ok("the worst scored skill is not a coin flip",
       (per_skill.get("worst_skill_auc") or 0) >= 0.60,
       f"({per_skill.get('worst_skill_auc')})")
    ok("evaluation used held-out learners, not held-out rows",
       r.get("split", {}).get("val_learners", 0) > 0
       and r.get("split", {}).get("train_learners", 0) > 0,
       f"({r.get('split', {})})")
else:
    ok("reports/m3_kt.json - run `make eval-kt`", False)

if os.getenv("DATABASE_URL"):
    import sqlalchemy as sa
    eng = sa.create_engine(os.environ["DATABASE_URL"])
    with eng.connect() as c:
        rows = c.execute(sa.text("SELECT count(*) FROM mastery")).scalar()
        out_of_range = c.execute(sa.text(
            "SELECT count(*) FROM mastery WHERE estimate < 0 OR estimate > 1"
        )).scalar()
        noversion = c.execute(sa.text(
            "SELECT count(*) FROM mastery WHERE model_version IS NULL")).scalar()
        zero_n = c.execute(sa.text(
            "SELECT count(*) FROM mastery WHERE n_responses < 1")).scalar()
        applied = c.execute(sa.text(
            "SELECT count(*) FROM mastery_applied")).scalar()
        cycles = c.execute(sa.text("""
            WITH RECURSIVE walk(root, node, depth) AS (
                SELECT prereq_id, skill_id, 1 FROM skill_edges
                UNION ALL
                SELECT w.root, e.skill_id, w.depth + 1
                FROM walk w JOIN skill_edges e ON e.prereq_id = w.node
                WHERE w.depth < 50)
            SELECT count(*) FROM walk WHERE root = node""")).scalar()
        ok("mastery rows written", rows > 0, f"({rows})")
        ok("every estimate is a probability", out_of_range == 0)
        ok("every row records the model that produced it", noversion == 0)
        ok("confidence is backed by a response count", zero_n == 0)
        ok("the idempotency guard is being used", applied > 0, f"({applied} applied)")
        ok("the skill graph is acyclic", cycles == 0, f"({cycles} cycles)")
else:
    print("[-- ] DATABASE_URL not set: skipping the database checks")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)

