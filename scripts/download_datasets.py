#!/usr/bin/env python
"""Stage MasteryLoop response-log corpora and item banks."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
BANKS = RAW / 'item_banks'
RESPONSES = RAW / 'responses'

PUBLIC = {
    'ednet-kt1-sample.zip':
    'https://github.com/riiid/ednet/raw/master/sample/KT1_sample.zip',
}

REGISTERED = [
    ('EdNet full (KT1-KT4)', 'https://github.com/riiid/ednet',
     'download from the linked Drive and extract under data/raw/responses/ednet/'),
    ('ASSISTments 2009-2010', 'https://sites.google.com/site/assistmentsdata/',
     'place skill_builder_data.csv under data/raw/responses/assistments/'),
    ('Eedi / NeurIPS 2020', 'https://eedi.com/projects/neurips-education-challenge',
     'accept the terms, then extract under data/raw/responses/eedi/'),
]


def fetch(name: str, url: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / name
    if dest.exists():
        print(f'[skip] {name}')
        return
    print(f'[get ] {name}')
    urlretrieve(url, dest)
    if dest.suffix == '.zip':
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(target_dir)


def main() -> int:
    for d in (BANKS, RESPONSES):
        d.mkdir(parents=True, exist_ok=True)
        (d / '.gitkeep').touch()

    for name, url in PUBLIC.items():
        try:
            fetch(name, url, RESPONSES)
        except Exception as exc:
            print(f'[warn] {name} failed: {exc}')
            print('       seed_db.py generates a synthetic response history instead.')

    print()
    print('Corpora requiring registration or terms acceptance:')
    for name, url, hint in REGISTERED:
        print(f' - {name}: {url}')
        print(f'   {hint}')
    print()
    print('Live item content is exam material — data/raw/item_banks/ is git-ignored.')
    print('Next: dvc add data/raw && dvc push')
    return 0


if __name__ == '__main__':
    sys.exit(main())
