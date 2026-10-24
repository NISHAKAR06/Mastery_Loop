"""Post-response mastery update (FR-5, FR-7). Budget: under 300 ms.
Two models run side by side and both are written, tagged with their version:
SAKT when it is the promoted model, BKT always. Keeping BKT live costs
microseconds and means a SAKT outage degrades the mastery map's accuracy rather
than freezing it - and it gives the rolling comparison in M6 real production
data rather than a replay.
"""
from __future__ import annotations
import json
from masteryloop_ml.kt.bkt_baseline import BktParams, update as bkt_update
from .mastery import confidence

UPSERT = """
INSERT INTO mastery (learner_id, skill_id, estimate, confidence, n_responses,
                     updated_at, model_version)
VALUES (:learner, :skill, :estimate, :confidence, 1, now(), :version)
ON CONFLICT (learner_id, skill_id) DO UPDATE SET
    estimate = EXCLUDED.estimate,
    confidence = EXCLUDED.confidence,
    n_responses = mastery.n_responses + 1,
    updated_at = now(),
    model_version = EXCLUDED.model_version"""

class MasteryUpdater:
    def __init__(self, bkt_params: dict, sakt=None, model_version: str = "bkt@1"):
        self.bkt_params = bkt_params
        self.sakt = sakt
        self.model_version = model_version

    def params_for(self, skill_id: str) -> BktParams:
        """An unseen skill falls back to the population defaults rather than
        raising. A brand-new skill must still produce a mastery estimate on the
        learner's first answer, or their map has a hole in it."""
        return self.bkt_params.get(str(skill_id), BktParams())

    def apply(self, conn, event: dict) -> list[dict]:
        """One row per skill the item is tagged with (FR-1 allows 1-3). An item
        tagged with three skills credits all three - that is what the curriculum
        mapping asserts, and splitting the credit would make a multi-skill item
        worth less than three single-skill ones."""
        import sqlalchemy as sa
        rows = []
        for skill_id in event["skill_ids"]:
            current = conn.execute(sa.text(
                "SELECT estimate, n_responses FROM mastery "
                "WHERE learner_id = :l AND skill_id = :s"),
                {"l": event["learner_id"], "s": skill_id}).mappings().first()
            params = self.params_for(skill_id)
            prior = float(current["estimate"]) if current else params.p_init
            n = int(current["n_responses"]) + 1 if current else 1
            estimate = bkt_update(prior, bool(event["correct"]), params)
            if self.sakt is not None:
                estimate = self._sakt_estimate(event, skill_id, estimate)
            conn.execute(sa.text(UPSERT), {
                "learner": event["learner_id"], "skill": skill_id,
                "estimate": estimate, "confidence": confidence(n),
                "version": self.model_version})
            rows.append({"skill_id": skill_id, "estimate": estimate,
                         "confidence": confidence(n), "n_responses": n})
        return rows

    def _sakt_estimate(self, event, skill_id, fallback: float) -> float:
        """SAKT predicts P(correct on the next item), which is not the same
        quantity as P(mastered). It is used here only when the promoted model
        is SAKT, and a failure falls back to the BKT number rather than to no
        number - an empty mastery map is worse than a slightly stale one."""
        try:
            return float(self.sakt.mastery_for(event["learner_id"], skill_id))
        except Exception: # noqa: BLE001
            return fallback

def mastery_payload(rows) -> str:
    return json.dumps(rows, default=float)

