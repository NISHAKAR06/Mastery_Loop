"""Pure item response theory.

Three-parameter logistic model, Fisher information, and bounded Newton-Raphson
maximum-likelihood ability estimation. No database, no framework — which is what
lets the whole measurement model be verified against hand-computed values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

THETA_MIN = -4.0
THETA_MAX = 4.0


class IRTError(ValueError):
    pass


@dataclass(frozen=True)
class ItemParameters:
    """a = discrimination, b = difficulty, c = pseudo-guessing."""
    item_id: str
    a: float
    b: float
    c: float = 0.0

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise IRTError(f'{self.item_id}: discrimination must be positive')
        if not 0.0 <= self.c < 1.0:
            raise IRTError(f'{self.item_id}: guessing must be in [0, 1)')
        if not -5.0 <= self.b <= 5.0:
            raise IRTError(f'{self.item_id}: difficulty out of plausible range')


@dataclass(frozen=True)
class Response:
    item: ItemParameters
    correct: bool


@dataclass(frozen=True)
class AbilityEstimate:
    theta: float
    standard_error: float
    iterations: int
    converged: bool


def probability(theta: float, item: ItemParameters) -> float:
    """3PL: P = c + (1 - c) / (1 + exp(-a(theta - b)))."""
    z = -item.a * (theta - item.b)
    # guard against overflow at extreme thetas
    if z > 700:
        logistic = 0.0
    elif z < -700:
        logistic = 1.0
    else:
        logistic = 1.0 / (1.0 + math.exp(z))
    return item.c + (1.0 - item.c) * logistic


def information(theta: float, item: ItemParameters) -> float:
    """Fisher information for a 3PL item.

    I = a^2 * (P - c)^2 * (1 - P) / ((1 - c)^2 * P)
    For c = 0 this reduces to a^2 * P * (1 - P), which is the 2PL form.
    """
    p = probability(theta, item)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    numerator = (item.a ** 2) * ((p - item.c) ** 2) * (1.0 - p)
    denominator = ((1.0 - item.c) ** 2) * p
    return numerator / denominator


def test_information(theta: float, items: list[ItemParameters]) -> float:
    """Information is additive across items — the basis of adaptive efficiency."""
    return sum(information(theta, i) for i in items)


test_information.__test__ = False


def standard_error(theta: float, items: list[ItemParameters]) -> float:
    total = test_information(theta, items)
    if total <= 0:
        return float('inf')
    return 1.0 / math.sqrt(total)


def log_likelihood(theta: float, responses: list[Response]) -> float:
    total = 0.0
    for r in responses:
        p = min(max(probability(theta, r.item), 1e-12), 1 - 1e-12)
        total += math.log(p) if r.correct else math.log(1 - p)
    return total


def _score_and_info(theta: float, responses: list[Response]) -> tuple[float, float]:
    """First derivative of the log-likelihood, and the information at theta."""
    score = 0.0
    info = 0.0
    for r in responses:
        p = min(max(probability(theta, r.item), 1e-12), 1 - 1e-12)
        a, c = r.item.a, r.item.c
        # dP/dtheta for the 3PL
        dp = a * (p - c) * (1.0 - p) / (1.0 - c)
        observed = 1.0 if r.correct else 0.0
        score += dp * (observed - p) / (p * (1.0 - p))
        info += information(theta, r.item)
    return score, info


def all_same_response(responses: list[Response]) -> bool:
    return len({r.correct for r in responses}) == 1


def estimate_ability(responses: list[Response], initial_theta: float = 0.0,
                     max_iterations: int = 40, tolerance: float = 1e-4) -> AbilityEstimate:
    """Bounded Newton-Raphson MLE.

    An all-correct or all-incorrect pattern has no finite maximum likelihood, so
    the estimate is pinned to the scale boundary rather than diverging. That is a
    deliberate, tested behaviour — not a numerical accident.
    """
    if not responses:
        raise IRTError('cannot estimate ability without responses')
    if all_same_response(responses):
        theta = THETA_MAX if responses[0].correct else THETA_MIN
        return AbilityEstimate(theta=theta,
                               standard_error=standard_error(theta, [r.item for r in responses]),
                               iterations=0, converged=False)
    theta = min(max(initial_theta, THETA_MIN), THETA_MAX)
    for iteration in range(1, max_iterations + 1):
        score, info = _score_and_info(theta, responses)
        if info <= 1e-9:
            break
        step = score / info
        # damp large steps so a flat likelihood cannot throw the estimate off scale
        step = max(-1.0, min(1.0, step))
        new_theta = min(max(theta + step, THETA_MIN), THETA_MAX)
        if abs(new_theta - theta) < tolerance:
            theta = new_theta
            return AbilityEstimate(
                theta=round(theta, 4),
                standard_error=standard_error(theta, [r.item for r in responses]),
                iterations=iteration, converged=True)
        theta = new_theta
    return AbilityEstimate(
        theta=round(theta, 4),
        standard_error=standard_error(theta, [r.item for r in responses]),
        iterations=max_iterations, converged=False)


def should_stop(responses_count: int, current_se: float, min_items: int = 8,
                max_items: int = 30, target_se: float = 0.30) -> tuple[bool, str]:
    """Stopping rule. Minimum length wins over precision, deliberately."""
    if min_items > max_items:
        raise IRTError('min_items cannot exceed max_items')
    if responses_count < min_items:
        return False, ''
    if current_se <= target_se:
        return True, 'target_se_reached'
    if responses_count >= max_items:
        return True, 'max_items'
    return False, ''
