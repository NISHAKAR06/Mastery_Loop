#!/usr/bin/env python
"""Seed the MasteryLoop skill graph, item bank, learners and response history."""
from __future__ import annotations

import math
import os
import random
from datetime import datetime, timedelta, timezone

import psycopg
from psycopg.rows import dict_row

DSN = (
    f"host={os.getenv('DB_HOST', 'localhost')} port={os.getenv('DB_PORT', '5432')} "
    f"dbname={os.getenv('DB_NAME', 'masteryloop')} user={os.getenv('DB_USER', 'mastery')} "
    f"password={os.getenv('DB_PASS', 'mastery_dev_2026')}"
)

DOMAIN = 'Python Programming'

# (skill_id, name, [prerequisites])
SKILLS = [
    ('PY.VAR', 'Variables and types', []),
    ('PY.SEQ', 'Sequences and slicing', ['PY.VAR']),
    ('PY.CTRL', 'Control flow', ['PY.VAR']),
    ('PY.FUNC', 'Functions and scope', ['PY.CTRL']),
    ('PY.DICT', 'Dictionaries and sets', ['PY.SEQ']),
    ('PY.COMP', 'Comprehensions', ['PY.SEQ', 'PY.CTRL']),
    ('PY.OOP', 'Classes and objects', ['PY.FUNC']),
    ('PY.EXC', 'Exceptions', ['PY.FUNC']),
    ('PY.ITER', 'Iterators and generators', ['PY.COMP', 'PY.FUNC']),
    ('PY.FILE', 'Files and context managers', ['PY.EXC', 'PY.OOP']),
]

CONTENT_AREAS = ['fundamentals', 'application', 'analysis', 'synthesis']


def depth_of(skill_id: str, prereq_map: dict, cache: dict) -> int:
    """Longest path to a root — also proves the graph is acyclic."""
    if skill_id in cache:
        return cache[skill_id]
    cache[skill_id] = 0  # provisional; a revisit means a cycle
    prereqs = prereq_map.get(skill_id, [])
    value = 0 if not prereqs else 1 + max(depth_of(p, prereq_map, cache) for p in prereqs)
    cache[skill_id] = value
    return value


def seed_skills(cur) -> list[str]:
    prereq_map = {sid: prereqs for sid, _, prereqs in SKILLS}
    cache: dict[str, int] = {}
    for skill_id, name, _ in SKILLS:
        cur.execute(
            'INSERT INTO skills (skill_id, skill_name, domain, depth) '
            'VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING',
            (skill_id, name, DOMAIN, depth_of(skill_id, prereq_map, cache)),
        )
    for skill_id, _, prereqs in SKILLS:
        for p in prereqs:
            cur.execute(
                'INSERT INTO skill_prerequisites (skill_id, prerequisite_id, strength) '
                'VALUES (%s,%s,%s) ON CONFLICT DO NOTHING', (skill_id, p, 1.0),
            )
    return [s[0] for s in SKILLS]


def seed_items(cur, skill_ids: list[str], per_skill: int = 24) -> list[dict]:
    items = []
    n = 0
    for skill_id in skill_ids:
        for k in range(per_skill):
            n += 1
            area = CONTENT_AREAS[k % len(CONTENT_AREAS)]
            cur.execute(
                'INSERT INTO items (item_code, stem, item_type, content_area, '
                'expected_seconds, status, authored_by) '
                "VALUES (%s,%s,'mcq',%s,%s,'active','item.author@iamneo.example') "
                'RETURNING item_id',
                (f'ITEM{n:05d}',
                 f'[{skill_id}] Demo question {k + 1} assessing {skill_id} at {area} level.',
                 area, random.choice([45, 60, 90, 120])),
            )
            item_id = cur.fetchone()['item_id']
            correct = random.choice('ABCD')
            for label in 'ABCD':
                cur.execute(
                    'INSERT INTO item_options (item_id, option_label, option_text, '
                    'is_correct, misconception) VALUES (%s,%s,%s,%s,%s)',
                    (item_id, label, f'Option {label} for {skill_id}',
                     label == correct,
                     None if label == correct else f'confuses {skill_id} with a related rule'),
                )
            cur.execute(
                'INSERT INTO item_skills (item_id, skill_id, weight) VALUES (%s,%s,%s)',
                (item_id, skill_id, 1.0),
            )
            # difficulty tracks the content area; discrimination varies realistically
            b = {'fundamentals': -1.2, 'application': -0.2,
                 'analysis': 0.7, 'synthesis': 1.5}[area] + random.gauss(0, 0.5)
            a = round(min(2.5, max(0.4, random.gauss(1.1, 0.35))), 3)
            c = round(random.uniform(0.15, 0.25), 3)  # 4-option MCQ guessing
            cur.execute(
                'INSERT INTO item_calibrations (item_id, version, discrimination, difficulty, '
                'guessing, sample_size, standard_error, is_active) '
                'VALUES (%s,1,%s,%s,%s,%s,%s,TRUE) RETURNING calibration_id',
                (item_id, a, round(b, 3), c, random.randint(250, 1800),
                 round(random.uniform(0.05, 0.18), 3)),
            )
            items.append({'item_id': item_id, 'a': a, 'b': b, 'c': c,
                          'calibration_id': cur.fetchone()['calibration_id'],
                          'skill_id': skill_id})
    return items


def seed_assessment(cur) -> str:
    cur.execute(
        'INSERT INTO assessments (assessment_code, title, domain, min_items, max_items, '
        'target_se, content_mix) '
        "VALUES ('PY-ADAPT-01','Python Adaptive Diagnostic',%s,8,30,0.300,"
        "'{\"fundamentals\":0.25,\"application\":0.35,\"analysis\":0.20,\"synthesis\":0.20}')"
        ' RETURNING assessment_id', (DOMAIN,),
    )
    return cur.fetchone()['assessment_id']


def p_correct(theta: float, a: float, b: float, c: float) -> float:
    return c + (1 - c) / (1 + math.exp(-a * (theta - b)))


def seed_sessions(cur, assessment_id: str, items: list[dict], n_learners: int = 120) -> None:
    now = datetime.now(timezone.utc)
    for i in range(n_learners):
        cur.execute(
            'INSERT INTO learners (external_ref, display_name, cohort) '
            'VALUES (%s,%s,%s) RETURNING learner_id',
            (f'LRN{i:05d}', f'Demo Learner {i:03d}',
             random.choice(['2026-JAN-A', '2026-JAN-B', '2026-JUL-A'])),
        )
        learner_id = cur.fetchone()['learner_id']
        true_theta = random.gauss(0, 1)
        cur.execute(
            'INSERT INTO assessment_sessions (assessment_id, learner_id, started_at, '
            'completed_at, items_administered, initial_theta, final_theta, final_se, '
            "stop_reason, engine_version) VALUES (%s,%s,%s,%s,%s,0,%s,%s,%s,'m1-seed') "
            'RETURNING session_id',
            (assessment_id, learner_id, now - timedelta(days=random.randint(1, 40)),
             now - timedelta(days=random.randint(0, 1)), 0,
             round(true_theta + random.gauss(0, 0.25), 3),
             round(random.uniform(0.22, 0.34), 3), 'target_se_reached'),
        )
        session_id = cur.fetchone()['session_id']
        chosen = random.sample(items, k=random.randint(12, 20))
        theta = 0.0
        for seq, item in enumerate(chosen, start=1):
            p = p_correct(true_theta, item['a'], item['b'], item['c'])
            correct = random.random() < p
            cur.execute(
                'INSERT INTO responses (session_id, item_id, calibration_id, sequence_no, '
                'selected_option, is_correct, theta_at_admin, latency_ms) '
                'VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',
                (session_id, item['item_id'], item['calibration_id'], seq,
                 random.choice('ABCD'), correct, round(theta, 3),
                 int(random.gauss(42_000, 15_000).__abs__())),
            )
            # crude running update purely so the trajectory is non-trivial demo data
            theta += (0.30 if correct else -0.30) / math.sqrt(seq)
            cur.execute(
                'INSERT INTO ability_estimates (session_id, after_sequence, theta, '
                "standard_error, estimator) VALUES (%s,%s,%s,%s,'mle')",
                (session_id, seq, round(theta, 3), round(1.0 / math.sqrt(seq + 1), 3)),
            )
            cur.execute(
                'INSERT INTO item_exposure (item_id, window_start, administered, correct_count) '
                'VALUES (%s, CURRENT_DATE, 1, %s) '
                'ON CONFLICT (item_id, window_start) DO UPDATE SET '
                'administered = item_exposure.administered + 1, '
                'correct_count = item_exposure.correct_count + EXCLUDED.correct_count',
                (item['item_id'], 1 if correct else 0),
            )
        cur.execute(
            'UPDATE assessment_sessions SET items_administered = %s, final_theta = %s '
            'WHERE session_id = %s', (len(chosen), round(theta, 3), session_id),
        )


def main() -> None:
    random.seed(20260911)
    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT COUNT(*) AS c FROM skills')
            if cur.fetchone()['c'] > 0:
                print('already seeded — run `make reset` first to reseed')
                return
            skill_ids = seed_skills(cur)
            items = seed_items(cur, skill_ids)
            assessment_id = seed_assessment(cur)
            seed_sessions(cur, assessment_id, items)
            conn.commit()
    print('seeded: 10 skills, 240 calibrated items, 1 assessment, 120 learners, ~2k responses')


if __name__ == '__main__':
    main()
