"""KT evaluation (objective 5.1): AUC >= 0.77 AND at least 4 points above BKT.
Both halves matter. An AUC of 0.78 that is 0.5 points above a forty-year-old
four-parameter model does not justify a transformer in the serving path, and
saying so is the honest outcome of this milestone rather than a failure.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

MIN_AUC = 0.77
MIN_LIFT_POINTS = 0.04

def auc(labels, predictions) -> float:
    y = np.asarray(labels, int)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(predictions, float)))

def calibration(labels, predictions, bins: int = 10) -> dict:
    """A KT model's output drives item selection, so it must be a probability
    and not just a ranking. AUC is blind to calibration; Brier is not."""
    y = np.asarray(labels, int)
    p = np.clip(np.asarray(predictions, float), 1e-6, 1 - 1e-6)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
                         "predicted": float(p[m].mean()),
                         "observed": float(y[m].mean())})
    n = sum(r["n"] for r in rows) or 1
    return {"brier": float(brier_score_loss(y, p)),
            "ece": float(sum(r["n"] * abs(r["predicted"] - r["observed"])
                             for r in rows) / n),
            "bins": rows}

def per_skill_auc(labels, predictions, skills, min_n: int = 50) -> dict:
    """A headline AUC can hide a model that works on the three most common
    skills and is a coin flip everywhere else - which is exactly the model that
    misroutes a learner on a rare skill."""
    y = np.asarray(labels, int)
    p = np.asarray(predictions, float)
    s = np.asarray([str(x) for x in skills])
    out = {}
    for skill in sorted(set(s.tolist())):
        m = s == skill
        if m.sum() < min_n:
            out[skill] = {"n": int(m.sum()), "auc": None,
                          "note": "too few responses to score"}
            continue
        out[skill] = {"n": int(m.sum()), "auc": auc(y[m], p[m])}
    scored = [v["auc"] for v in out.values()
              if v["auc"] is not None and not np.isnan(v["auc"])]
    return {"per_skill": out,
            "worst_skill_auc": float(min(scored)) if scored else None,
            "skills_scored": len(scored)}

def compare(labels, bkt_predictions, sakt_predictions, skills=None,
            min_auc: float = MIN_AUC,
            min_lift: float = MIN_LIFT_POINTS) -> dict:
    """The milestone's verdict, with both gates stated separately so a failure
    says which one failed."""
    bkt, sakt = auc(labels, bkt_predictions), auc(labels, sakt_predictions)
    lift = sakt - bkt
    gates = [(f"SAKT AUC >= {min_auc}", sakt >= min_auc, f"{sakt:.4f}"),
             (f"SAKT beats BKT by >= {min_lift * 100:.0f} points",
              lift >= min_lift, f"{bkt:.4f} -> {sakt:.4f} ({lift * 100:+.1f} pts)")]
    out = {"bkt_auc": bkt, "sakt_auc": sakt, "lift_points": float(lift * 100),
           "sakt_calibration": calibration(labels, sakt_predictions),
           "bkt_calibration": calibration(labels, bkt_predictions),
           "gates": [{"gate": n, "pass": bool(ok), "detail": d}
                     for n, ok, d in gates],
           "passes": all(ok for _, ok, _ in gates)}
    if skills is not None:
        out["sakt_per_skill"] = per_skill_auc(labels, sakt_predictions, skills)
    out["recommendation"] = (
        "ship SAKT" if out["passes"]
        else "ship BKT: the transformer does not earn its serving cost")
    return out

