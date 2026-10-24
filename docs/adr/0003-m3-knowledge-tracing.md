---
noteId: "b790c9d0cf8711f1adbc43e9eb186c4a"
tags: []

---

# ADR 0003 - Milestone 3 knowledge-tracing decisions

Status: accepted (M3)

- **D1 Split on learners, then truncate at the origin.** A learner split alone
  is not leakage-safe: the evaluation question is "given everything up to t,
  will they get the NEXT item right", so a validation sequence fed whole lets
  the model attend to answers that had not happened yet. Both halves are
  enforced in `data/splits.py` and the truncation has a poisoned-future test.
- **D2 The encoding lives in its own torch-free module.** An off-by-one in the
  interaction encoding hands the model the answer to the item it is predicting,
  the AUC jumps to 0.99, and nothing raises. That code must be testable on any
  laptop, so it is not inside the file that imports PyTorch.
- **D3 The causal mask is strictly upper-triangular.** `diagonal=1`. Position t
  attending to position t is the same leak by another route.
- **D4 BKT is a real baseline, fitted per skill.** A single global BKT would be
  a straw man and would make the four-point gate trivial. Per-skill EM with the
  standard guess/slip bound is what the literature means by "BKT baseline".
- **D5 Guess and slip are clamped at 0.30.** Above that the model explains the
  data by claiming everyone guesses rather than that anyone learned: the
  likelihood improves and the mastery estimates become meaningless.
- **D6 Both gates are reported separately, and failing the second is a result.**
  An AUC of 0.78 that is half a point above a four-parameter model does not
  justify a transformer in the serving path. The recommendation field says
  "ship BKT" when that is the honest answer.
- **D7 Calibration is reported alongside AUC.** The KT output drives item
  selection, so it must be a probability and not merely a ranking. AUC is blind
  to that; Brier and ECE are not.
- **D8 Per-skill AUC is reported.** A headline number can hide a model that
  works on the three commonest skills and is a coin flip on the rest - exactly
  the model that misroutes a learner on a rare skill.
- **D9 Mastery decays toward the 0.5 prior, at read time.** Toward the prior
  because forgetting makes a skill uncertain, not worse than never-seen; at read
  time because a nightly job means Sunday shows Saturday's numbers and writes a
  row per learner per night for nothing.
- **D10 Mastery needs an estimate AND a confidence.** A 0.95 from one lucky
  answer is not mastery, and advancing a learner past a gap on that basis is the
  most expensive mistake an adaptive system makes.
- **D11 An unreachable skill is `locked`, not `weak`, and names its blockers.**
  Sending a learner to a skill whose prerequisites are unmastered wastes their
  time, and "not ready" without a reason is not actionable for a teacher.
- **D12 The consumer is idempotent and acknowledges only after commit.**
  At-least-once delivery means it WILL see an interaction twice after a
  restart; without `mastery_applied` every pod recycle drifts the estimates
  upward, and acknowledging before committing loses the update silently.
