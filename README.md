---
noteId: "289942b0ae7711f1ad27637c7c4c4864"
tags: []
---
# MasteryLoop — Adaptive Assessment & Personalised Mastery Platform

MasteryLoop is an adaptive assessment engine. Instead of asking every learner the same forty questions,
it estimates ability after each response and asks the one question that reduces uncertainty most —
then converts the final estimate into a skill-level mastery map and a study path.

## Problem Statement

A fixed test wastes most of its questions: items far above or below a learner's level tell
you almost nothing about them. Percentage scores make it worse by conflating item difficulty
with learner ability, so 70% on an easy paper and 70% on a hard one look identical.
MasteryLoop uses item response theory to put learners and items on the same scale, which
makes a shorter test more accurate than a longer one and makes scores comparable across
different item sets and different sittings.

## Architecture (M1 baseline)

```
learner starts a session
|
v
[learner-api] --session state--> Redis
|
v
[item-selector] --max information + exposure + content balance--> next item
| ^
v |
learner responds |
| |
v |
[ability-estimator] --theta, SE--> ability_estimates
| |
+--------- stop rule met? -- no -----+
| yes
v
[mastery-tracker] --roll up the skill graph--> mastery_states --> study path
```

## Milestones

| Milestone | Focus                                                                   |
| --------- | ----------------------------------------------------------------------- |
| M1        | Architecture, infrastructure, skill graph, item bank and IRT foundation |
| M2        | Item parameter calibration and knowledge-tracing model training         |
| M3        | Adaptive session service with live ability estimation                   |
| M4        | Agentic tutoring: distractor diagnosis and explanation generation       |
| M5 | Learner and instructor dashboards with mastery visualisation |
| M6 | Deployment, item drift monitoring and bank health reporting |
## Quick Start
```bash
cp .env.example .env
make up # start the full infrastructure stack
make db-init # apply schema (auto-applied on first boot)
make seed # load reference + demo data
make verify # M1 acceptance checks
```
## Repository Layout
- `libs/mastery_domain/` — pure IRT and item selection maths
- `services/` — one container per agent
- `scripts/` — schema, item bank ETL, seeding, verification
- `config/` — assessment rules and the skill graph definition
- `docs/adr/` — architecture decision records
## Tech Stack
- **PostgreSQL** — Skill graph, item bank, sessions, responses and mastery state
- **Redis** — Live session state and item exposure counters
- **Qdrant** — Item and misconception embeddings for distractor analysis from M4
- **MinIO** — Item media — diagrams, audio and code attachments
- **MLflow** — Calibration and knowledge-tracing model tracking from M2
## License
MIT

