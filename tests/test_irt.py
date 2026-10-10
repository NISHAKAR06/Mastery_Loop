import math
import pytest
from libs.mastery_domain.irt import (
    IRTError, ItemParameters, Response, estimate_ability, information, probability,
    should_stop, standard_error, test_information, THETA_MAX, THETA_MIN,
)

test_information.__test__ = False


def item(a=1.0, b=0.0, c=0.0, name='I1'):
    return ItemParameters(name, a, b, c)


def test_probability_is_half_at_difficulty_when_no_guessing():
    assert probability(0.0, item(a=1.2, b=0.0)) == pytest.approx(0.5)


def test_guessing_raises_the_floor():
    p = probability(-6.0, item(a=1.0, b=0.0, c=0.25))
    assert p == pytest.approx(0.25, abs=2e-3)


def test_probability_is_monotonic_in_ability():
    i = item(a=1.4, b=0.3, c=0.2)
    values = [probability(t, i) for t in (-3, -1, 0, 1, 3)]
    assert values == sorted(values)


def test_probability_never_leaves_the_unit_interval():
    i = item(a=2.5, b=-3.0, c=0.3)
    assert 0.0 <= probability(-50.0, i) <= 1.0
    assert 0.0 <= probability(50.0, i) <= 1.0


def test_invalid_parameters_are_rejected():
    with pytest.raises(IRTError):
        ItemParameters('bad', a=0.0, b=0.0)
    with pytest.raises(IRTError):
        ItemParameters('bad', a=1.0, b=0.0, c=1.0)
    with pytest.raises(IRTError):
        ItemParameters('bad', a=1.0, b=9.0)


def test_information_peaks_at_the_item_difficulty_for_2pl():
    i = item(a=1.0, b=0.8)
    peak = information(0.8, i)
    assert peak > information(-0.5, i)
    assert peak > information(2.2, i)


def test_2pl_information_reduces_to_a_squared_pq():
    i = item(a=1.3, b=0.0, c=0.0)
    p = probability(0.0, i)
    assert information(0.0, i) == pytest.approx(i.a ** 2 * p * (1 - p))


def test_higher_discrimination_carries_more_information():
    assert information(0.0, item(a=1.8)) > information(0.0, item(a=0.6))


def test_information_is_additive_across_items():
    items = [item(a=1.0, b=-0.5, name='A'), item(a=1.2, b=0.5, name='B')]
    assert test_information(0.0, items) == pytest.approx(
        information(0.0, items[0]) + information(0.0, items[1]))


def test_standard_error_falls_as_items_accumulate():
    few = [item(a=1.0, name=f'I{i}') for i in range(4)]
    many = [item(a=1.0, name=f'I{i}') for i in range(16)]
    assert standard_error(0.0, many) < standard_error(0.0, few)
    # SE scales as 1/sqrt(n): four times the items halves the error
    assert standard_error(0.0, many) == pytest.approx(standard_error(0.0, few) / 2, rel=1e-6)


def test_standard_error_is_infinite_with_no_information():
    assert standard_error(0.0, []) == float('inf')


def test_all_correct_is_pinned_to_the_scale_maximum():
    responses = [Response(item(b=b, name=f'I{b}'), True) for b in (-1.0, 0.0, 1.0)]
    est = estimate_ability(responses)
    assert est.theta == THETA_MAX
    assert not est.converged


def test_all_incorrect_is_pinned_to_the_scale_minimum():
    responses = [Response(item(b=b, name=f'I{b}'), False) for b in (-1.0, 0.0, 1.0)]
    assert estimate_ability(responses).theta == THETA_MIN


def test_mixed_pattern_converges_near_the_generating_ability():
    # correct on easy items, wrong on hard ones -> ability sits in the middle
    responses = (
        [Response(item(a=1.2, b=b, name=f'E{b}'), True) for b in (-2.0, -1.5, -1.0, -0.5)]
        + [Response(item(a=1.2, b=b, name=f'H{b}'), False) for b in (0.5, 1.0, 1.5, 2.0)]
    )
    est = estimate_ability(responses)
    assert est.converged
    assert -0.5 < est.theta < 0.5


def test_estimate_never_leaves_the_bounded_scale():
    responses = (
        [Response(item(a=2.4, b=-3.0, name=f'E{i}'), True) for i in range(9)]
        + [Response(item(a=2.4, b=-2.9, name='H'), False)]
    )
    est = estimate_ability(responses)
    assert THETA_MIN <= est.theta <= THETA_MAX


def test_estimate_requires_at_least_one_response():
    with pytest.raises(IRTError):
        estimate_ability([])


def test_stopping_rule_respects_the_minimum_length():
    stop, reason = should_stop(responses_count=3, current_se=0.10)
    assert not stop and reason == ''


def test_stopping_rule_fires_on_target_precision():
    stop, reason = should_stop(responses_count=12, current_se=0.28)
    assert stop and reason == 'target_se_reached'


def test_stopping_rule_fires_on_maximum_length():
    stop, reason = should_stop(responses_count=30, current_se=0.55)
    assert stop and reason == 'max_items'


def test_stopping_rule_rejects_impossible_bounds():
    with pytest.raises(IRTError):
        should_stop(10, 0.3, min_items=30, max_items=8)
