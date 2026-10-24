"""Mastery state, decay and prerequisite-aware readiness (FR-7).
Three things the mastery API has to get right, none of which the KT model
itself provides:
 decay mastery measured six weeks ago is not mastery today, and a map
 that never fades tells a learner they have mastered something they
 have forgotten;
 confidence a skill answered twice and a skill answered forty times should not
 look identical on a dashboard;
 readiness a skill whose prerequisites are unmastered is not "weak", it is
 NOT YET REACHABLE, and sending a learner there is the single most
 common way an adaptive system wastes their time.
"""
from __future__ import annotations
import math

MASTERY_THRESHOLD = 0.80
HALF_LIFE_DAYS = 30.0
CONFIDENCE_SATURATION = 20 # responses at which confidence is effectively full

def decay(estimate: float, days_since: float,
          half_life_days: float = HALF_LIFE_DAYS) -> float:
    """Exponential decay toward the 0.5 prior, not toward zero. Forgetting
    makes a skill uncertain again; it does not make a learner worse than
    someone who has never seen it."""
    if days_since <= 0:
        return float(estimate)
    retained = 0.5 ** (days_since / max(half_life_days, 1e-9))
    return float(0.5 + (float(estimate) - 0.5) * retained)

def confidence(n_responses: int, saturation: int = CONFIDENCE_SATURATION) -> float:
    """Saturating in the number of observations. Reported separately from the
    estimate so a dashboard can grey out a skill measured from two answers
    instead of pretending it knows."""
    return float(1.0 - math.exp(-max(int(n_responses), 0) / max(saturation, 1)))

def is_mastered(estimate: float, conf: float,
                threshold: float = MASTERY_THRESHOLD,
                min_confidence: float = 0.5) -> bool:
    """Both conditions. A 0.95 estimate from one lucky answer is not mastery,
    and treating it as one is how a learner gets advanced past a gap."""
    return bool(estimate >= threshold and conf >= min_confidence)

def validate_dag(edges) -> list[str]:
    """edges: [(prereq_id, skill_id)]. A cycle in the curriculum makes every
    skill permanently unreachable, and it is the kind of thing a well-meaning
    content author creates on a Friday afternoon."""
    graph: dict = {}
    for prereq, skill in edges:
        graph.setdefault(str(prereq), set()).add(str(skill))
    colour: dict = {}
    problems: list[str] = []
    def visit(node, stack):
        colour[node] = "grey"
        for nxt in sorted(graph.get(node, ())):
            if colour.get(nxt) == "grey":
                cycle = stack[stack.index(nxt):] + [nxt] if nxt in stack \
                    else [node, nxt]
                problems.append("prerequisite cycle: " + " -> ".join(cycle))
            elif colour.get(nxt) is None:
                visit(nxt, stack + [nxt])
        colour[node] = "black"
    for node in sorted(graph):
        if colour.get(node) is None:
            visit(node, [node])
    return problems

def readiness(skill_id: str, prereq_map: dict, mastery: dict,
              threshold: float = MASTERY_THRESHOLD) -> dict:
    """prereq_map: {skill: [prereq, ...]}. mastery: {skill: {'estimate',
    'confidence'}}. Reports which prerequisites are blocking, because
    'not ready' without a reason is not actionable for a learner or a teacher."""
    prereqs = [str(p) for p in prereq_map.get(str(skill_id), [])]
    blocking = []
    for p in prereqs:
        row = mastery.get(p)
        if row is None or not is_mastered(row.get("estimate", 0.0),
                                          row.get("confidence", 0.0), threshold):
            blocking.append(p)
    own = mastery.get(str(skill_id), {})
    return {"skill_id": str(skill_id), "ready": not blocking,
            "blocking_prerequisites": blocking,
            "estimate": own.get("estimate"), "confidence": own.get("confidence"),
            "state": ("mastered" if is_mastered(own.get("estimate", 0.0),
                                                own.get("confidence", 0.0),
                                                threshold)
                      else "in_progress" if not blocking else "locked")}

def skill_map(mastery: dict, prereq_map: dict, now_days: dict | None = None,
              threshold: float = MASTERY_THRESHOLD) -> list[dict]:
    """The full learner map the API returns: decayed estimate, confidence,
    readiness and what is blocking each locked skill."""
    now_days = now_days or {}
    decayed = {s: {**row,
                   "estimate": decay(row["estimate"], now_days.get(s, 0.0))}
               for s, row in mastery.items()}
    return [readiness(s, prereq_map, decayed, threshold)
            for s in sorted(set(decayed) | set(prereq_map))]

def weakest_skills(skill_rows, n: int = 3) -> list[dict]:
    """Objective 5.6: the weakest three skills a class should work on next.
    Locked skills are excluded - telling a class to practise something they
    cannot yet reach is advice nobody can act on."""
    reachable = [r for r in skill_rows
                 if r["state"] != "locked" and r.get("estimate") is not None]
    return sorted(reachable, key=lambda r: r["estimate"])[:n]

