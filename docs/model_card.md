# Model card - masteryloop-irt-2pl

## Purpose
Calibrate item difficulty and discrimination and estimate learner ability, so M4 can select
maximally-informative next questions and M5 can statistically gate AI-generated items.

## Not for
Skill-level mastery tracking (that is the SAKT knowledge tracer, M3); items in the pilot lifecycle
with too few responses; any scoring of learners outside the calibrated item pool.

## Data
- Source: ASSISTments skill-builder log: first attempts only, thin items and learners filtered in two passes.
- Split: Learner-wise 70/10/20; held-out learners scored first-half-evidence -> second-half-target
- Labels: Observed correctness of each first attempt.

## Model
2PL IRT fitted by alternating ridge-Newton steps with an N(0,1) ability prior and per-round
standardisation; abilities for new learners via grid EAP.

## Incumbent baseline
Classical test theory: item-mean and two-way (item easiness + learner skill) logistic - the thing
this model replaces or augments.

## Evaluation (held-out test set; numbers filled from reports/eval.json)
Promotion gates (all must pass):
- `model.auc` >= 0.7 - Published-competitive held-out prediction on tutoring data (Actual: 0.7368)
- `derived.auc_gain_vs_item_mean` >= 0.03 - Clearly beats 'every learner is average' (Actual: +0.0551)
- `derived.auc_gain_vs_two_way` >= 0.0 - At least matches the best no-IRT predictor (Actual: +0.0237)
- `recovery.corr_a` >= 0.8 - Discrimination estimates track truth on synthetic learners (Actual: 0.8050)
- `recovery.rmse_b` <= 0.35 - Difficulty recovered within a third of an ability SD (Actual: 0.2512)

Actual results: see `reports/eval.json` and the MLflow run tagged `champion`.

## Known limitations
- Unidimensional: one ability per learner, so multi-skill items blur (MIRT is a documented future
  enhancement).
- Parameters sit on this cohort's scale; a new population needs re-anchoring.
- Items below the response floor stay uncalibrated by design - they must pilot longer, not be
  guessed at.

## Ownership
Group: MasteryLoop candidates.
Retrain trigger: nightly once live interactions flow in M3; param_version increments on every accepted refit.
