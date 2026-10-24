"""Prepare interaction dataset for knowledge tracing."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml

def load_params():
    params_path = Path("params.yaml")
    if params_path.exists():
        with open(params_path) as f:
            return yaml.safe_load(f)
    return {}

def prepare_dataset(raw_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = load_params()
    min_item_count = params.get("kt_data", {}).get("min_item_count", 10)

    # 1. Check for EdNet files
    ednet_candidates = [
        raw_dir / "ednet",
        raw_dir / "responses" / "ednet",
        Path("data/raw/ednet"),
        Path("data/raw/responses/ednet"),
    ]
    ednet_dir = next((d for d in ednet_candidates if d.exists() and any(d.glob("*.csv"))), None)

    # 2. Check for ASSISTments file
    assistments_file = raw_dir / "assistments" / "skill_builder_data.csv"
    if not assistments_file.exists():
        assistments_file = Path("data/raw/assistments/skill_builder_data.csv")

    if ednet_dir and any(ednet_dir.glob("*.csv")):
        # Read from EdNet
        csv_files = list(ednet_dir.glob("*.csv"))
        dfs = []
        for f in csv_files:
            uid = f.stem
            sub_df = pd.read_csv(f)
            sub_df["learner_id"] = uid
            dfs.append(sub_df)
        df = pd.concat(dfs, ignore_index=True)
        # map columns standard to EdNet KT1
        if "question_id" in df.columns:
            df["item_id"] = df["question_id"].astype(str)
        if "user_answer" in df.columns and "correct" not in df.columns:
            df["correct"] = (df["user_answer"] == df.get("correct_answer", 1)).astype(int)
        if "timestamp" in df.columns:
            df["ts"] = df["timestamp"].astype("int64")
        if "skill_id" not in df.columns:
            df["skill_id"] = "general"
    elif assistments_file.exists():
        df = pd.read_csv(assistments_file, low_memory=False, encoding="ISO-8859-1")
        df = df.dropna(subset=["user_id", "problem_id", "skill_id", "correct", "order_id"])
        if "original" in df.columns:
            df = df[df["original"] == 1]
        
        item_counts = df["problem_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item_count].index
        df = df[df["problem_id"].isin(valid_items)].copy()
        
        df["learner_id"] = df["user_id"].astype(str)
        df["item_id"] = df["problem_id"].astype(str)
        df["skill_id"] = df["skill_id"].astype("int64").astype(str)
        df["correct"] = df["correct"].astype("int32")
        df["ts"] = df["order_id"].astype("int64")
    else:
        raise FileNotFoundError("Neither EdNet nor ASSISTments raw dataset found.")

    cols = ["learner_id", "item_id", "skill_id", "correct", "ts"]
    df_out = df[cols].sort_values(["learner_id", "ts"]).reset_index(drop=True)
    
    out_file = out_dir / "interactions.parquet"
    df_out.to_parquet(out_file, index=False)
    print(f"Prepared {len(df_out):,} interactions | {df_out['learner_id'].nunique():,} learners | "
          f"{df_out['item_id'].nunique():,} items | {df_out['skill_id'].nunique():,} skills")
    print(f"Saved to {out_file}")
    return out_file

def main():
    parser = argparse.ArgumentParser(description="Prepare interactions for knowledge tracing")
    parser.add_argument("--out", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--raw", type=str, default="data/raw", help="Raw data directory")
    args = parser.parse_args()

    prepare_dataset(Path(args.raw), Path(args.out))

if __name__ == "__main__":
    main()

