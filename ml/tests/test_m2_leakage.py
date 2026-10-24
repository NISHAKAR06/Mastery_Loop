"""Leakage and statistical-sanity tests for IRT calibration."""
import numpy as np
import pandas as pd
from masteryloop.split import filter_thin, learner_split
from masteryloop.features import eap_theta, halves, p_correct, simulate
from masteryloop.train import fit_2pl


def toy():
    df, a, b = simulate(300, 60, 25, seed=4)
    return learner_split(df, 0.2, 0.1, seed=4), a, b


def test_each_learner_lives_in_exactly_one_split():
    df, _, _ = toy()
    assert (df.groupby("user_id")["split"].nunique() == 1).all()
    assert set(df["split"]) == {"train", "val", "test"}


def test_filtering_is_a_fixed_point_after_two_passes():
    df, _, _ = toy()
    once = filter_thin(df, 20, 10)
    twice = filter_thin(once, 20, 10)
    pd.testing.assert_frame_equal(once, twice)


def test_halves_are_time_ordered_and_disjoint():
    df, _, _ = toy()
    first, second = halves(df)
    assert len(first) + len(second) == len(df)
    joint = first.groupby("user_id")["order_id"].max().to_frame("f").join(
        second.groupby("user_id")["order_id"].min().to_frame("s"), how="inner")
    assert (joint["f"] < joint["s"]).all()


def test_eap_moves_the_right_way():
    a = np.array([1.0, 1.0, 1.0, 1.0])
    b = np.zeros(4)
    hi = eap_theta(a, b, np.array([1, 1, 1, 1]))
    lo = eap_theta(a, b, np.array([0, 0, 0, 0]))
    assert hi > 0.5 and lo < -0.5 and abs(hi + lo) < 0.2  # symmetric prior


def test_parameter_recovery_on_synthetic_learners():
    """The statistical unit test: fit learners with KNOWN a, b; assert recovery."""
    df, a_true, b_true = simulate(500, 50, 30, seed=9)
    items, _ = fit_2pl(df, l2=1.0, outer_iters=6)
    fit = items.set_index("problem_id").loc[np.arange(50)]
    assert np.corrcoef(fit["a"], a_true)[0, 1] > 0.6  # looser than the gate:
    assert np.sqrt(np.mean((fit["b"] - b_true) ** 2)) < 0.5  # tiny data, quick test


def test_probabilities_are_monotone_in_ability():
    p = p_correct(np.linspace(-3, 3, 13), 1.2, 0.0)
    assert (np.diff(p) > 0).all()
