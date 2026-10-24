"""Run the interaction stream consumer."""
from __future__ import annotations
import json
import os
from pathlib import Path
import redis
import sqlalchemy as sa
from masteryloop_ml.kt.bkt_baseline import BktParams
from .consumer import run
from .updater import MasteryUpdater

def load_bkt_params():
    p_file = Path("artifacts/bkt/bkt_params.json")
    if p_file.exists():
        with open(p_file) as f:
            data = json.load(f)
            return {s: BktParams(**params) for s, params in data.items()}
    return {}

def main():
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    db_url = os.getenv("DATABASE_URL", "postgresql+psycopg://mastery:mastery_dev_2026@127.0.0.1:5432/masteryloop")
    
    r = redis.Redis.from_url(redis_url, decode_responses=False)
    engine = sa.create_engine(db_url, pool_pre_ping=True)
    
    bkt_params = load_bkt_params()
    updater = MasteryUpdater(bkt_params=bkt_params, model_version="bkt@1")
    
    print("Starting interaction stream consumer (kt-updater)...")
    stats = run(r, engine, updater, max_batches=10)
    print(f"Consumer finished batch run: {stats}")

if __name__ == "__main__":
    main()

