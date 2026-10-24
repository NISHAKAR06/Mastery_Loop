"""Mastery writes, the idempotency guard and the class aggregate, against the
real database. Skipped without DATABASE_URL."""
import os
import uuid
import pytest

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="needs the stack")

import sqlalchemy as sa # noqa: E402
from kt_svc.consumer import already_applied, mark_applied # noqa: E402
from kt_svc.updater import MasteryUpdater # noqa: E402
from masteryloop_ml.kt.bkt_baseline import BktParams # noqa: E402

@pytest.fixture()
def learner():
    eng = sa.create_engine(os.environ["DATABASE_URL"])
    lid, sid = uuid.uuid4(), uuid.uuid4()
    with eng.begin() as c:
        c.execute(sa.text("INSERT INTO skills (id, name, subject) "
                          "VALUES (:i, :n, 'maths')"),
                  {"i": sid, "n": f"skill-{sid.hex[:6]}"})
        c.execute(sa.text("INSERT INTO learners (id, display_name, cohort) "
                          "VALUES (:i, :n, 'demo')"),
                  {"i": lid, "n": f"L{lid.hex[:6]}"})
    yield eng, lid, sid
    with eng.begin() as c:
        c.execute(sa.text("DELETE FROM mastery WHERE learner_id = :l"), {"l": lid})
        c.execute(sa.text("DELETE FROM learners WHERE id = :l"), {"l": lid})
        c.execute(sa.text("DELETE FROM skills WHERE id = :s"), {"s": sid})

def event(lid, sid, interaction_id, correct=True):
    return {"interaction_id": interaction_id, "learner_id": str(lid),
            "item_id": str(uuid.uuid4()), "skill_ids": [str(sid)],
            "correct": correct}

def test_a_correct_response_raises_the_estimate_and_counts_it(learner):
    eng, lid, sid = learner
    updater = MasteryUpdater({str(sid): BktParams(p_init=0.3, p_transit=0.1)})
    with eng.begin() as c:
        first = updater.apply(c, event(lid, sid, 1))[0]
        second = updater.apply(c, event(lid, sid, 2))[0]
        row = c.execute(sa.text(
            "SELECT estimate, confidence, n_responses, model_version "
            "FROM mastery WHERE learner_id = :l AND skill_id = :s"),
            {"l": lid, "s": sid}).one()
        assert second["estimate"] > first["estimate"] > 0.3
        assert row.n_responses == 2 and 0.0 < row.confidence < 1.0
        assert row.model_version

def test_replaying_an_interaction_does_not_double_count(learner):
    eng, lid, sid = learner
    updater = MasteryUpdater({str(sid): BktParams()})
    with eng.begin() as c:
        updater.apply(c, event(lid, sid, 11))
        mark_applied(c, 11, sa)
    with eng.begin() as c:
        assert already_applied(c, 11, sa) is True
        if not already_applied(c, 11, sa): # the consumer's own guard
            updater.apply(c, event(lid, sid, 11))
        n = c.execute(sa.text("SELECT n_responses FROM mastery "
                              "WHERE learner_id = :l"), {"l": lid}).scalar()
        assert n == 1
    with eng.begin() as c:
        c.execute(sa.text("DELETE FROM mastery_applied WHERE interaction_id = 11"))

def test_an_estimate_outside_zero_to_one_is_rejected_by_the_database(learner):
    eng, lid, sid = learner
    with pytest.raises(sa.exc.IntegrityError):
        with eng.begin() as c:
            c.execute(sa.text(
                "INSERT INTO mastery (learner_id, skill_id, estimate, confidence, "
                "n_responses, model_version) VALUES (:l, :s, 1.4, 0.9, 1, 'x')"),
                {"l": lid, "s": sid})

