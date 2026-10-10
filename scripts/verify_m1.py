#!/usr/bin/env python
"""M1 acceptance checks for MasteryLoop."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import defaultdict

import psycopg

DSN = (
    f"host={os.getenv('DB_HOST', 'localhost')} port={os.getenv('DB_PORT', '5432')} "
    f"dbname={os.getenv('DB_NAME', 'masteryloop')} user={os.getenv('DB_USER', 'mastery')} "
    f"password={os.getenv('DB_PASS', 'mastery_dev_2026')}"
)

EXPECTED_TABLES = {
    'learners', 'skills', 'skill_prerequisites', 'items', 'item_options', 'item_skills',
    'item_calibrations', 'assessments', 'assessment_sessions', 'responses',
    'ability_estimates', 'item_exposure', 'mastery_states', 'recommendations',
}

failures: list[str] = []


def check(label: str, ok: bool, detail: str = '') -> None:
    print(f"{'PASS' if ok else 'FAIL'} {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def has_cycle(edges: dict[str, list[str]]) -> bool:
    """Depth-first search with a recursion stack — the skill graph must be a DAG."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = defaultdict(int)

    def visit(node: str) -> bool:
        colour[node] = GREY
        for nxt in edges.get(node, []):
            if colour[nxt] == GREY:
                return True
            if colour[nxt] == WHITE and visit(nxt):
                return True
        colour[node] = BLACK
        return False

    return any(colour[n] == WHITE and visit(n) for n in list(edges))


def main() -> int:
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema='public'")
            found = {r[0] for r in cur.fetchall()}
            missing = EXPECTED_TABLES - found
            check('schema: all 14 tables present', not missing,
                  f'missing {sorted(missing)}' if missing else '14/14')

            cur.execute('SELECT count(*) FROM items')
            n = cur.fetchone()[0]
            check('bank: items seeded', n >= 240, f'{n} items')

            cur.execute('SELECT count(*) FROM item_calibrations WHERE is_active')
            active = cur.fetchone()[0]
            check('bank: every item has an active calibration', active == n,
                  f'{active} active of {n} items')

            cur.execute("SELECT count(*) FROM items i WHERE NOT EXISTS "
                        "(SELECT 1 FROM item_skills s WHERE s.item_id = i.item_id)")
            check('bank: every item maps to at least one skill', cur.fetchone()[0] == 0)

            cur.execute('SELECT skill_id, prerequisite_id FROM skill_prerequisites')
            edges: dict[str, list[str]] = defaultdict(list)
            for skill, prereq in cur.fetchall():
                edges[skill].append(prereq)
            check('curriculum: skill graph is acyclic', not has_cycle(edges))

            cur.execute('SELECT count(*) FROM responses')
            r = cur.fetchone()[0]
            check('history: responses seeded', r >= 1000, f'{r} responses')

            cur.execute('SELECT count(*) FROM ability_estimates')
            check('history: ability trajectory recorded', cur.fetchone()[0] >= r)

            cur.execute('SELECT count(*) FROM item_calibrations '
                        'WHERE discrimination <= 0 OR guessing >= 0.5')
            check('calibration: all parameters inside bounds', cur.fetchone()[0] == 0)

            # the one-correct-option index must actually fire
            with conn.cursor() as cur:
                cur.execute('SELECT item_id FROM item_options WHERE is_correct LIMIT 1')
                item_id = cur.fetchone()[0]
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            'INSERT INTO item_options (item_id, option_label, option_text, is_correct) '
                            "VALUES (%s,'Z','second correct answer',TRUE)", (item_id,))
                        conn.rollback()
                        check('constraint: second correct option blocked', False, 'insert succeeded')
                except psycopg.errors.UniqueViolation:
                    conn.rollback()
                    check('constraint: second correct option blocked', True)

    cmd = ['pytest', 'tests/', '-q'] if shutil.which('pytest') else [sys.executable, '-m', 'pytest', 'tests/', '-q']
    rc = subprocess.run(cmd, capture_output=True, text=True)
    check('domain: unit tests', rc.returncode == 0)

    print()
    if failures:
        print(f'M1 NOT COMPLETE — {len(failures)} check(s) failed')
        return 1
    print('M1 COMPLETE — safe to tag v0.1.0-M1')
    return 0


if __name__ == '__main__':
    sys.exit(main())
