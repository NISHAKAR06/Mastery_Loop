"""Load the ASSISTments interaction log staged in M1, keep first attempts, filter
thin items and learners, and split LEARNER-wise.

Why learner-wise: the deployment case is a learner the platform has never seen - their
ability must be estimated live from their own early answers, never memorised from
training rows. Any learner with rows on both sides of the split turns "predict this
person" into "remember this person". Held-out learners are evaluated with the honest
protocol: estimate ability from the FIRST HALF of their (time-ordered) responses,
predict the second half."""
import numpy as np
import pandas as pd
from .config import RAW, PROCESSED, SEED, load_cfg

NEEDED = ["user_id", "problem_id", "correct"]


def load_raw(path=None) -> pd.DataFrame:
    base = RAW / "assistments"
    path = path or next(iter(sorted(base.glob("*.csv"))), None)
    assert path is not None, f"no CSV under {base}"
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    assert set(NEEDED) <= set(df.columns), f"missing columns in {path.name}"
    if "order_id" not in df.columns:
        df["order_id"] = np.arange(len(df))
    df = df[NEEDED + ["order_id"]].dropna()
    df["correct"] = (df["correct"].astype(float) > 0).astype(int)
    df = (df.sort_values("order_id")
            .drop_duplicates(["user_id", "problem_id"], keep="first"))
    return df.reset_index(drop=True)


def filter_thin(df, min_item, min_user):
    """Two passes: dropping thin items can make a learner thin, and vice versa."""
    for _ in range(2):
        df = df[df.groupby("problem_id")["correct"].transform("size") >= min_item]
        df = df[df.groupby("user_id")["correct"].transform("size") >= min_user]
    return df.reset_index(drop=True)


def learner_split(df, test_frac, val_frac, seed=SEED):
    users = np.array(sorted(df["user_id"].unique()))
    np.random.default_rng(seed).shuffle(users)
    n_test, n_val = round(len(users) * test_frac), round(len(users) * val_frac)
    test_u, val_u = set(users[:n_test]), set(users[n_test:n_test + n_val])
    split = np.where(df["user_id"].isin(test_u), "test",
                     np.where(df["user_id"].isin(val_u), "val", "train"))
    return df.assign(split=split)


def main():
    cfg = load_cfg()["split"]
    df = filter_thin(load_raw(), cfg["min_item_responses"], cfg["min_user_responses"])
    df = learner_split(df, cfg["test_frac"], cfg["val_frac"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "dataset.parquet", index=False)
    print(f"{df.user_id.nunique()} learners, {df.problem_id.nunique()} items, "
          f"{len(df)} first-attempt responses")
    print("learners per split:", df.groupby("split")["user_id"].nunique().to_dict())


if __name__ == "__main__":
    main()
