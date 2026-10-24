"""Incumbents from classical test theory - what every quiz platform without IRT
effectively uses:
 - item-mean: every learner gets the item's training-set proportion correct;
 - two-way logistic: sigmoid(item easiness + learner skill), skill estimated from the
   learner's own first-half accuracy - the strongest no-IRT predictor.
Both use exactly the first-half/second-half protocol the model is held to."""
import numpy as np
import pandas as pd
from .features import sigmoid


def logit(p, eps=0.02):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class ItemMean:
    def fit(self, train: pd.DataFrame):
        self.mean = train.groupby("problem_id")["correct"].mean()
        self.fallback = float(train["correct"].mean())
        return self

    def predict(self, second: pd.DataFrame, first: pd.DataFrame) -> np.ndarray:
        return self.mean.reindex(second["problem_id"]).fillna(self.fallback).to_numpy()


class TwoWay(ItemMean):
    def predict(self, second: pd.DataFrame, first: pd.DataFrame) -> np.ndarray:
        easiness = logit(super().predict(second, first))
        skill = logit(first.groupby("user_id")["correct"].mean())
        global_e = logit(np.array([self.fallback]))[0]
        return sigmoid(easiness - global_e
                       + skill.reindex(second["user_id"]).fillna(0.0).to_numpy())


if __name__ == "__main__":
    from .config import PROCESSED
    from .features import halves
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    first, second = halves(va)
    y = second["correct"].to_numpy()
    for name, m in [("item-mean", ItemMean()), ("two-way", TwoWay())]:
        auc = roc_auc_score(y, m.fit(tr).predict(second, first))
        print(f"val AUC {name}: {auc:.4f}")
