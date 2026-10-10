---
noteId: "289942b1ae7711f1ad27637c7c4c4864"
tags: []
---

# ADR-0002: Item exposure control is enforced in the selector, not advised

**Status:** Accepted
**Date:** 2026-09-11

## Context

Pure maximum-information selection keeps choosing the same small set of highly
discriminating items. Those items leak, the bank degrades, and scores become
invalid — a slow failure that is invisible until it is expensive.

## Decision

`libs/mastery_domain/selection.py` applies a hard exposure cap and randomises
among the top-k informative items. A candidate over its cap is removed before
selection, not down-weighted. Exposure counters live in Redis with a rolling
window and are persisted to `item_exposure` for bank health reporting.

## Consequences

+ Bank life is extended and item leakage is measurable.
+ Two learners of identical ability do not see an identical test.

- A small measurement-efficiency cost per session, accepted deliberately.
- The bank must be large enough per content area or selection will fail.
