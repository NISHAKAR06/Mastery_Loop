"""Mastery API (FR-7): the skill map a learner, a teacher and the adaptive
engine all read.
Reads only - every write comes through the stream consumer. That separation is
what keeps the FR-3b guarantee true: nothing a dashboard does can block or lose
a learner's interaction.
"""
from __future__ import annotations
import os
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import create_engine, text
from .mastery import MASTERY_THRESHOLD, skill_map, weakest_skills

DATABASE_URL = os.getenv("DATABASE_URL",
                         "postgresql+psycopg://masteryloop:masteryloop@localhost:5432/masteryloop")
app = FastAPI(title="MasteryLoop knowledge tracing", version="0.3.0")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def prerequisites() -> dict:
    with engine.connect() as c:
        rows = c.execute(text(
            "SELECT skill_id, prereq_id FROM skill_edges")).all()
        out: dict = {}
        for skill_id, prereq_id in rows:
            out.setdefault(str(skill_id), []).append(str(prereq_id))
        return out

def learner_state(learner_id: str) -> tuple[dict, dict]:
    with engine.connect() as c:
        rows = c.execute(text(
            "SELECT skill_id, estimate, confidence, n_responses, "
            "EXTRACT(EPOCH FROM (now() - updated_at)) / 86400.0 AS days_since, "
            "model_version FROM mastery WHERE learner_id = :l"),
            {"l": learner_id}).mappings().all()
        state = {str(r["skill_id"]): {"estimate": float(r["estimate"]),
                                      "confidence": float(r["confidence"]),
                                      "n_responses": int(r["n_responses"]),
                                      "model_version": r["model_version"]}
                 for r in rows}
        days = {str(r["skill_id"]): float(r["days_since"]) for r in rows}
        return state, days

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/v1/learners/{learner_id}/mastery")
def mastery(learner_id: str,
            threshold: float = Query(MASTERY_THRESHOLD, ge=0.5, le=0.99)):
    """The full map. Decay is applied at READ time, not by a nightly job: a
    nightly job means a learner who logs in on Sunday sees Saturday's numbers,
    and it writes a row for every learner every night for no reason."""
    state, days = learner_state(learner_id)
    if not state:
        # an empty map is a valid answer for a new learner, not a 404 - the
        # learner app renders "nothing measured yet", which is the truth
        return {"learner_id": learner_id, "skills": [], "measured": False}
    rows = skill_map(state, prerequisites(), days, threshold)
    return {"learner_id": learner_id, "measured": True, "skills": rows,
            "mastered": sum(1 for r in rows if r["state"] == "mastered"),
            "locked": sum(1 for r in rows if r["state"] == "locked"),
            "model_versions": sorted({v["model_version"] for v in state.values()})}

@app.get("/api/v1/learners/{learner_id}/next-skills")
def next_skills(learner_id: str, n: int = Query(3, ge=1, le=10)):
    """What this learner should work on. Locked skills are excluded, so the
    answer is always something they can actually start today."""
    state, days = learner_state(learner_id)
    rows = skill_map(state, prerequisites(), days)
    return {"learner_id": learner_id, "recommended": weakest_skills(rows, n)}

@app.get("/api/v1/classes/{class_id}/weak-skills")
def class_weak_skills(class_id: str, n: int = Query(3, ge=1, le=10)):
    """Objective 5.6, under five seconds: the weakest skills for a class with
    the learner count behind each. Aggregated in SQL rather than by looping
    learners in Python - a class of forty would otherwise be forty round trips."""
    with engine.connect() as c:
        rows = c.execute(text("""
            SELECT m.skill_id, s.name, avg(m.estimate) AS mean_estimate,
                   count(*) AS learners,
                   count(*) FILTER (WHERE m.estimate < 0.6) AS struggling
            FROM mastery m
            JOIN enrollments e ON e.learner_id = m.learner_id
            JOIN skills s ON s.id = m.skill_id
            WHERE e.class_id = :c
            GROUP BY m.skill_id, s.name
            HAVING count(*) >= 3
            ORDER BY mean_estimate ASC
            LIMIT :n"""), {"c": class_id, "n": n}).mappings().all()
        if not rows:
            raise HTTPException(404, "no mastery data for this class yet")
        return {"class_id": class_id,
                "weak_skills": [dict(r) | {"mean_estimate": float(r["mean_estimate"])}
                                for r in rows]}

