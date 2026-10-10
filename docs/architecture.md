---
noteId: "2898cd80ae7711f1ad27637c7c4c4864"
tags: []
---

# MasteryLoop — Architecture

## 1. Context

Two scales exist side by side. The <b>latent scale</b> (theta) places learners and items on a
common logit continuum, typically -4 to +4. The <b>skill graph</b> is a directed acyclic graph
of prerequisite relations, and mastery propagates along it. Items map to one or more skills,
so a single response is evidence about several nodes. M1 models all of this; no calibration
model is trained yet.

## 2. Component Responsibilities

- **item-bank-service** — Serves items, enforces content balancing and keeps the active calibration set
- **ability-estimator** — Updates the learner's ability estimate and standard error after each response
- **item-selector** — Chooses the next item by maximum information, subject to exposure and content
  constraints
- **mastery-tracker** — Rolls item-level evidence up the skill graph into per-skill mastery states
- **learner-api** — FastAPI service driving the assessment session and returning the study path

## 3. Data Flow

1. `learner-api` opens a session, writes an `assessment_sessions` row and initialises theta
   from the learner's prior mastery state rather than from zero.
2. `item-selector` computes Fisher information for every eligible item at the current theta,
   filters by exposure and content constraints, and returns the highest-information item.
3. The learner responds; `responses` records the answer, the item, the theta at the time of
   administration and the response latency.
4. `ability-estimator` re-estimates theta and its standard error over all responses so far,
   writing an `ability_estimates` row per step — the full trajectory, not just the final value.
5. When the stopping rule fires, `mastery-tracker` propagates evidence through the skill
   graph and writes `mastery_states` plus a ranked study path.

## 4. Non-Functional Targets

- Every ability estimate stores the theta and standard error at that step, so a session is replayable
- No item is administered without an active calibration; uncalibrated items are seeded only
- Item exposure is capped per item per window, enforced before selection returns
- A session that cannot meet its content constraints fails loudly rather than silently relaxing them
