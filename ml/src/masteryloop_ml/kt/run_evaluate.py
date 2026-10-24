"""Evaluate KT models and print dual-verdict report."""
from __future__ import annotations
import json
from pathlib import Path
import sys

def main():
    report_file = Path("reports/m3_kt.json")
    if not report_file.exists():
        print(f"Error: {report_file} does not exist. Run training first.", file=sys.stderr)
        sys.exit(1)

    with open(report_file) as f:
        r = json.load(f)

    for g in r["gates"]:
        status = "OK " if g["pass"] else "XX "
        print(f"[{status}] {g['gate']} ({g['detail']})")

    cal = r.get("sakt_calibration", {})
    brier = cal.get("brier", 0.0)
    ece = cal.get("ece", 0.0)
    per_skill = r.get("sakt_per_skill", {})
    worst = per_skill.get("worst_skill_auc")
    worst_str = f"{worst:.3f}" if worst is not None else "None"

    print(f"calibration: brier {brier:.4f} ece {ece:.4f} | worst scored skill auc {worst_str}")
    print(f"recommendation: {r.get('recommendation', '')}")

if __name__ == "__main__":
    main()

