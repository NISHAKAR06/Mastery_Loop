"""Bayesian Knowledge Tracing - the baseline SAKT must beat by 4 AUC points
(objective 5.1).
Four parameters per skill and a closed-form update. It is forty years old, it
runs in microseconds, and on many public datasets it is within a couple of
points of a transformer. Shipping it as a real baseline rather than a straw man
is what makes the SAKT number mean something - and if SAKT cannot beat it by
four points, BKT is what should serve, because it is interpretable and free.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# sensible, literature-standard starting points for EM
DEFAULTS = {"p_init": 0.25, "p_transit": 0.10, "p_guess": 0.20, "p_slip": 0.10}
MAX_GUESS, MAX_SLIP = 0.30, 0.30 # the classic identifiability bound

@dataclass
class BktParams:
    p_init: float = DEFAULTS["p_init"]
    p_transit: float = DEFAULTS["p_transit"]
    p_guess: float = DEFAULTS["p_guess"]
    p_slip: float = DEFAULTS["p_slip"]

    def clamp(self) -> "BktParams":
        """Guess and slip above 0.3 make the model 'degenerate': it explains the
        data by claiming everyone guesses rather than by claiming anyone
        learned, and the mastery estimates become meaningless while the
        likelihood improves. The bound is standard and it is not optional."""
        return BktParams(
            p_init=float(np.clip(self.p_init, 0.01, 0.99)),
            p_transit=float(np.clip(self.p_transit, 0.001, 0.99)),
            p_guess=float(np.clip(self.p_guess, 0.01, MAX_GUESS)),
            p_slip=float(np.clip(self.p_slip, 0.01, MAX_SLIP)))

def predict(mastery: float, p: BktParams) -> float:
    """P(correct) given current mastery: master and do not slip, or do not
    master and guess."""
    return mastery * (1 - p.p_slip) + (1 - mastery) * p.p_guess

def posterior(mastery: float, correct: bool, p: BktParams) -> float:
    """Bayes update of P(mastered) given the observation, BEFORE learning."""
    if correct:
        num = mastery * (1 - p.p_slip)
        den = num + (1 - mastery) * p.p_guess
    else:
        num = mastery * p.p_slip
        den = num + (1 - mastery) * (1 - p.p_guess)
    return float(num / den) if den > 0 else mastery

def update(mastery: float, correct: bool, p: BktParams) -> float:
    """Posterior, then the learning transition. Order matters: transitioning
    first would credit the learner with learning from an item they have not
    answered yet."""
    post = posterior(mastery, correct, p)
    return float(post + (1 - post) * p.p_transit)

def sequence_predictions(responses, p: BktParams):
    """Returns (predictions, final_mastery). One prediction per response, each
    made from the state BEFORE that response - which is exactly the quantity
    AUC is computed on."""
    mastery = p.p_init
    out = []
    for correct in responses:
        out.append(predict(mastery, p))
        mastery = update(mastery, bool(correct), p)
    return np.asarray(out, dtype=float), float(mastery)

def fit(sequences, iterations: int = 30, init: BktParams | None = None) -> BktParams:
    """Expectation-maximisation over one skill's response sequences. Crude but
    honest: a grid-free EM in fifteen lines, bounded by clamp() every iteration
    so it cannot wander into the degenerate region."""
    p = (init or BktParams()).clamp()
    sequences = [list(map(bool, s)) for s in sequences if len(s)]
    if not sequences:
        return p
    for _ in range(iterations):
        init_sum = transit_num = transit_den = 0.0
        guess_num = guess_den = slip_num = slip_den = 0.0
        for seq in sequences:
            mastery = p.p_init
            init_sum += posterior(mastery, seq[0], p)
            for correct in seq:
                post = posterior(mastery, correct, p)
                guess_den += (1 - post)
                slip_den += post
                if correct:
                    guess_num += (1 - post)
                else:
                    slip_num += post
                transit_den += (1 - post)
                transit_num += (1 - post) * p.p_transit
                mastery = post + (1 - post) * p.p_transit
        p = BktParams(
            p_init=init_sum / len(sequences),
            p_transit=(transit_num / transit_den) if transit_den > 0 else p.p_transit,
            p_guess=(guess_num / guess_den) if guess_den > 0 else p.p_guess,
            p_slip=(slip_num / slip_den) if slip_den > 0 else p.p_slip).clamp()
    return p

def fit_per_skill(interactions, iterations: int = 30) -> dict:
    """One parameter set per skill. A single global BKT is the straw-man version
    of this baseline and would make the four-point gate trivial to clear."""
    by_skill: dict = {}
    for row in interactions:
        by_skill.setdefault(str(row["skill_id"]), {}).setdefault(
            str(row["learner_id"]), []).append(bool(row["correct"]))
    return {skill: fit(list(seqs.values()), iterations)
            for skill, seqs in by_skill.items()}
