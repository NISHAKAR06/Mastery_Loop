---
noteId: "289942b2ae7711f1ad27637c7c4c4864"
tags: []
---

# ADR-0001: Score with item response theory, not percentage correct

**Status:** Accepted
**Date:** 2026-09-11

## Context

Percentage correct confounds item difficulty with learner ability. Two learners
with the same percentage on different item sets are not comparable, and adaptive
selection makes the problem worse because everyone sees a different test.

## Decision

Learner ability (theta) and item difficulty are estimated on a common logit scale
using a three-parameter logistic model. `libs/mastery_domain/irt.py` implements
the probability function, the Fisher information function and Newton-Raphson
maximum-likelihood estimation as pure functions with a bounded fallback.

## Consequences

+ Scores are comparable across sittings, item sets and cohorts.
+ Shorter adaptive tests reach higher precision than longer fixed ones.

- Items must be calibrated before use, which needs response volume.
- Explaining theta to learners requires a translation layer in the UI.
