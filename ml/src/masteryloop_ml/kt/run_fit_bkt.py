"""Fit per-skill BKT baseline."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import yaml
from masteryloop_ml.data import splits
from masteryloop_ml.kt import bkt_baseline

def main():
    params_path = Path("params.yaml")
    params = {}
    if params_path.exists():
        with open(params_path) as f:
            params = yaml.safe_load(f)
    
    val_fraction = params.get("kt_data", {}).get("val_fraction", 0.2)
    min_history = params.get("kt_data", {}).get("min_history", 5)
    em_iterations = params.get("bkt", {}).get("em_iterations", 30)

    data_file = Path("data/processed/interactions.parquet")
    if not data_file.exists():
        raise FileNotFoundError(f"{data_file} not found. Run prepare_ednet first.")
    
    df = pd.read_parquet(data_file)
    interactions = df.to_dict(orient="records")
    
    split = splits.build_split(interactions, val_fraction=val_fraction, min_history=min_history)
    train_rows = [row for seq in split["train_sequences"].values() for row in seq]
    
    print(f"Fitting per-skill BKT on {len(train_rows):,} interactions across {split['report']['train_learners']:,} learners...")
    fitted_params = bkt_baseline.fit_per_skill(train_rows, iterations=em_iterations)
    
    out_dir = Path("artifacts/bkt")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bkt_params.json"
    
    serializable = {
        skill: {
            "p_init": p.p_init,
            "p_transit": p.p_transit,
            "p_guess": p.p_guess,
            "p_slip": p.p_slip,
        }
        for skill, p in fitted_params.items()
    }
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
        
    print(f"Per-skill BKT fitted for {len(fitted_params)} skills. Saved to {out_file}")

if __name__ == "__main__":
    main()

