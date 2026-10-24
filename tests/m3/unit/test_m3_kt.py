"""Splits, BKT, mastery and the KT gates. Every number a teacher sees on a
dashboard, and every decision about what a learner is served next, flows
through these four modules."""
import numpy as np
import pytest

from kt_svc import mastery as M
from masteryloop_ml.data import splits as S
from masteryloop_ml.kt import bkt_baseline as B
from masteryloop_ml.kt.evaluate import auc, calibration, compare, per_skill_auc

def test_learner_split_is_hash_stable_and_the_right_size():
    ids = [f"L{i}" for i in range(400)]
    a = S.assign(ids)
    assert a == S.assign(ids)
    assert 0.13 < sum(v == "val" for v in a.values()) / len(ids) < 0.27
    assert S.check_no_learner_leak(["a"], ["b"]) == []
    assert S.check_no_learner_leak(["a"], ["a"])[0].startswith("1 learners")
    assert any("empty" in p for p in S.check_no_learner_leak(["a"], []))

def test_truncation_hands_the_encoder_no_future():
    seq = list(range(10))
    out = S.truncate_at_origin(seq, 4)
    assert out["history"] == [0, 1, 2, 3] and out["target"] == 4
    with pytest.raises(ValueError):
        S.truncate_at_origin(seq, 0)
    with pytest.raises(ValueError):
        S.truncate_at_origin(seq, 10)

def test_poisoning_the_future_leaves_every_history_unchanged():
    seq = list(range(30))
    clean = S.evaluation_points(seq, min_history=5)
    poisoned_seq = seq[:15] + [-999] * 15
    poisoned = S.evaluation_points(poisoned_seq, min_history=5)
    # every prediction point whose origin is at or before 15 must be identical
    for a, b in zip(clean, poisoned):
        if len(a["history"]) <= 15:
            assert a["history"] == b["history"]

def interactions(n_learners=50, n_each=20):
    rows = []
    for i in range(n_learners):
        for t in range(n_each):
            rows.append({"learner_id": f"L{i}", "item_id": f"I{t}",
                         "skill_id": f"S{t % 3}", "correct": (t + i) % 3 != 0,
                         "ts": t})
    return rows

def test_build_split_separates_learners_and_truncates_every_point():
    out = S.build_split(interactions())
    assert out["report"]["train_learners"] and out["report"]["val_learners"]
    assert set(out["train_sequences"]) & {p["learner_id"]
                                          for p in out["val_points"]} == set()
    for p in out["val_points"]:
        assert len(p["history"]) >= 5
        assert p["target"] not in p["history"]

def test_bkt_prediction_is_hand_checkable():
    p = B.BktParams(p_init=0.0, p_transit=0.0, p_guess=0.2, p_slip=0.1)
    # mastery 0 -> only the guess rate; mastery 1 -> 1 minus the slip rate
    assert B.predict(0.0, p) == pytest.approx(0.2)
    assert B.predict(1.0, p) == pytest.approx(0.9)
    assert B.predict(0.5, p) == pytest.approx(0.55)

def test_a_correct_answer_raises_mastery_and_a_wrong_one_lowers_it():
    p = B.BktParams(p_init=0.3, p_transit=0.0, p_guess=0.2, p_slip=0.1)
    assert B.posterior(0.3, True, p) > 0.3
    assert B.posterior(0.3, False, p) < 0.3
    # hand-worked: 0.3*0.9 / (0.3*0.9 + 0.7*0.2) = 0.27 / 0.41
    assert B.posterior(0.3, True, p) == pytest.approx(0.27 / 0.41)

def test_learning_is_applied_after_the_observation_not_before():
    p = B.BktParams(p_init=0.3, p_transit=0.5, p_guess=0.2, p_slip=0.1)
    post = B.posterior(0.3, True, p)
    assert B.update(0.3, True, p) == pytest.approx(post + (1 - post) * 0.5)

def test_a_sequence_predicts_from_the_state_before_each_response():
    p = B.BktParams(p_init=0.2, p_transit=0.3, p_guess=0.2, p_slip=0.1)
    preds, final = B.sequence_predictions([True] * 6, p)
    assert preds[0] == pytest.approx(B.predict(0.2, p))
    assert np.all(np.diff(preds) > 0) # consistent success raises the forecast
    assert final > 0.9

def test_em_stays_out_of_the_degenerate_region():
    rng = np.random.default_rng(0)
    # learners who start weak and improve - the regime BKT is built for
    seqs = [[bool(rng.random() < min(0.15 + 0.08 * t, 0.95)) for t in range(20)]
            for _ in range(200)]
    fitted = B.fit(seqs, iterations=20)
    assert 0.01 <= fitted.p_guess <= B.MAX_GUESS
    assert 0.01 <= fitted.p_slip <= B.MAX_SLIP
    assert fitted.p_transit > 0.0
    wild = B.BktParams(p_guess=0.9, p_slip=0.8).clamp()
    assert wild.p_guess == B.MAX_GUESS and wild.p_slip == B.MAX_SLIP

def test_parameters_are_fitted_per_skill_not_globally():
    params = B.fit_per_skill(interactions(), iterations=5)
    assert set(params) == {"S0", "S1", "S2"}
    assert all(isinstance(v, B.BktParams) for v in params.values())

def test_mastery_decays_toward_the_prior_not_toward_zero():
    assert M.decay(0.9, 0) == pytest.approx(0.9)
    one_half_life = M.decay(0.9, M.HALF_LIFE_DAYS)
    assert one_half_life == pytest.approx(0.7)
    assert M.decay(0.9, 10_000) == pytest.approx(0.5, abs=1e-6)
    # a weak skill decays UP toward the prior, not further down
    assert M.decay(0.1, M.HALF_LIFE_DAYS) == pytest.approx(0.3)

def test_one_lucky_answer_is_not_mastery():
    assert M.confidence(0) == 0.0
    assert M.confidence(2) < 0.5 < M.confidence(20)
    assert not M.is_mastered(0.95, M.confidence(1))
    assert M.is_mastered(0.95, M.confidence(25))

def test_a_curriculum_cycle_is_caught():
    assert M.validate_dag([("a", "b"), ("b", "c")]) == []
    problems = M.validate_dag([("a", "b"), ("b", "c"), ("c", "a")])
    assert problems and "cycle" in problems[0]

def test_an_unreachable_skill_is_locked_with_its_blockers_named():
    prereqs = {"algebra2": ["algebra1"], "algebra1": []}
    low = {"algebra1": {"estimate": 0.3, "confidence": 0.9},
           "algebra2": {"estimate": 0.2, "confidence": 0.9}}
    out = M.readiness("algebra2", prereqs, low)
    assert out["ready"] is False and out["state"] == "locked"
    assert out["blocking_prerequisites"] == ["algebra1"]
    high = {**low, "algebra1": {"estimate": 0.9, "confidence": 0.9}}
    assert M.readiness("algebra2", prereqs, high)["state"] == "in_progress"

def test_the_weakest_skills_exclude_the_ones_a_class_cannot_reach():
    prereqs = {"a": [], "b": ["a"], "c": []}
    state = {"a": {"estimate": 0.4, "confidence": 0.9},
             "b": {"estimate": 0.1, "confidence": 0.9},
             "c": {"estimate": 0.6, "confidence": 0.9}}
    rows = M.skill_map(state, prereqs)
    weakest = M.weakest_skills(rows, n=2)
    assert [r["skill_id"] for r in weakest] == ["a", "c"] # b is locked by a
    assert all(r["state"] != "locked" for r in weakest)

def test_the_skill_map_applies_decay_before_judging_readiness():
    prereqs = {"b": ["a"]}
    state = {"a": {"estimate": 0.85, "confidence": 0.9},
             "b": {"estimate": 0.4, "confidence": 0.9}}
    fresh = M.skill_map(state, prereqs)
    stale = M.skill_map(state, prereqs, now_days={"a": 120.0})
    assert next(r for r in fresh if r["skill_id"] == "b")["state"] == "in_progress"
    assert next(r for r in stale if r["skill_id"] == "b")["state"] == "locked"

def predictions(y, separation):
    rng = np.random.default_rng(1)
    return 1 / (1 + np.exp(-(rng.normal(0, 1, len(y)) + separation * np.asarray(y))))

def test_both_gates_are_reported_separately():
    y = np.r_[np.zeros(500, int), np.ones(500, int)]
    bkt = predictions(y, 1.2)
    sakt = predictions(y, 3.0)
    out = compare(y, bkt, sakt)
    assert out["passes"] is True and out["recommendation"] == "ship SAKT"
    assert len(out["gates"]) == 2
    tie = compare(y, sakt, sakt)
    assert tie["passes"] is False
    assert "does not earn" in tie["recommendation"]
    assert [g["pass"] for g in tie["gates"]] == [True, False]

def test_calibration_is_reported_because_auc_is_blind_to_it():
    y = np.r_[np.zeros(500, int), np.ones(500, int)]
    ranked_well_but_squashed = 0.5 + 0.01 * (np.asarray(y) - 0.5)
    out = calibration(y, ranked_well_but_squashed)
    assert auc(y, ranked_well_but_squashed) > 0.9 # perfect ranking
    assert out["ece"] > 0.4 # and terrible probabilities

def test_a_rare_skill_with_too_few_responses_is_not_scored():
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    p = predictions(y, 3.0)
    skills = ["common"] * 390 + ["rare"] * 10
    out = per_skill_auc(y, p, skills)
    assert out["per_skill"]["rare"]["auc"] is None
    assert out["per_skill"]["common"]["auc"] is not None
    assert out["skills_scored"] == 1
    assert np.isnan(auc([1, 1, 1], [0.1, 0.2, 0.3]))

def test_sakt_encoding_never_shows_an_item_its_own_answer():
    from masteryloop_ml.kt.encoding import PAD, encode_interaction, encode_sequence
    items, corrects = [3, 7, 2], [True, False, True]
    inter, nxt, targets = encode_sequence(items, corrects, n_items=10, max_len=5)
    assert len(inter) == len(nxt) == len(targets) == 5
    # position of the first real prediction: its interaction slot is PAD, so the
    # model predicts item 3 with no knowledge of how it was answered
    first = int(np.argmax(nxt > 0))
    assert inter[first] == PAD
    assert nxt[first] == 4 # item 3, +1 for the PAD offset
    assert targets[first] == 1
    # the NEXT position sees the previous interaction, not its own
    assert inter[first + 1] == encode_interaction(3, True, 10)
    assert nxt[first + 1] == 8 # item 7
    assert targets[:first].tolist() == [-100] * first

def test_the_interaction_symbol_separates_correct_from_incorrect():
    from masteryloop_ml.kt.encoding import encode_interaction
    assert encode_interaction(0, False, 10) == 1 # 0 is reserved for PAD
    assert encode_interaction(0, True, 10) == 11
    assert encode_interaction(9, False, 10) != encode_interaction(9, True, 10)

def test_the_consumer_parses_and_rejects_malformed_events():
    import json as _json
    from kt_svc.consumer import parse
    good = {"payload": _json.dumps(
        {"interaction_id": "i1", "learner_id": "L1", "item_id": "I1",
         "skill_ids": ["S1"], "correct": True})}
    assert parse(good)["learner_id"] == "L1"
    with pytest.raises(ValueError, match="missing"):
        parse({"payload": _json.dumps({"interaction_id": "i1"})})

def test_the_updater_falls_back_to_defaults_for_an_unseen_skill():
    from kt_svc.updater import MasteryUpdater
    u = MasteryUpdater({"S1": B.BktParams(p_init=0.9)})
    assert u.params_for("S1").p_init == pytest.approx(0.9)
    assert u.params_for("brand-new").p_init == pytest.approx(B.DEFAULTS["p_init"])

