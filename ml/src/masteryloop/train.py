"""2PL calibration by alternating Newton steps - item step and ability step - with
an N(0,1) ability prior and per-iteration standardisation for identifiability. No
black box: ~60 lines of numpy that a psychometrician can read, and that the recovery
harness certifies. Validation metric: AUC predicting held-out learners' second-half
responses from first-half EAP abilities - the exact protocol evaluate.py gates."""
import json
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .config import PROCESSED, ARTIFACTS, load_cfg
from .features import eap_theta, halves, p_correct, sigmoid
from . import tracking


def fit_item(theta, y, l2, steps=8):
    """Newton fit of y ~ sigmoid(a*theta + c) for one item; returns (a, b)."""
    w = np.array([1.0, 0.0])
    X = np.c_[theta, np.ones_like(theta)]
    for _ in range(steps):
        p = sigmoid(X @ w)
        g = X.T @ (y - p) - l2 * w
        wgt = np.maximum(p * (1 - p), 1e-4)
        H = (X * wgt[:, None]).T @ X + l2 * np.eye(2)
        w = w + np.linalg.solve(H, g)
    a = float(np.clip(w[0], 0.2, 3.5))
    return a, float(np.clip(-w[1] / a, -4, 4))


def fit_2pl(train: pd.DataFrame, l2: float, outer_iters: int):
    """Alternate: (1) per-item logistic Newton given abilities, (2) vectorised
    Newton ability update given item params, (3) standardise abilities."""
    users = {u: i for i, u in enumerate(train["user_id"].unique())}
    items = {p: j for j, p in enumerate(train["problem_id"].unique())}
    u = train["user_id"].map(users).to_numpy()
    j = train["problem_id"].map(items).to_numpy()
    y = train["correct"].to_numpy(dtype=float)
    pct = train.groupby("user_id")["correct"].mean().reindex(users).to_numpy()
    theta = np.clip(np.log((pct + 0.02) / (1.02 - pct)), -3, 3)
    theta = (theta - theta.mean()) / max(theta.std(), 1e-6)
    a = np.ones(len(items))
    b = np.zeros(len(items))
    by_item = pd.Series(np.arange(len(train))).groupby(j).apply(np.array)
    for _ in range(outer_iters):
        for jj, rows in by_item.items():
            a[jj], b[jj] = fit_item(theta[u[rows]], y[rows], l2)
        for _ in range(3):  # ability Newton steps
            p = p_correct(theta[u], a[j], b[j])
            g, h = -theta.copy(), np.ones(len(users))  # N(0,1) prior
            np.add.at(g, u, a[j] * (y - p))
            np.add.at(h, u, a[j] ** 2 * p * (1 - p))
            theta = theta + g / h
        theta = (theta - theta.mean()) / max(theta.std(), 1e-6)
    n_resp = train.groupby("problem_id")["correct"].size()
    out = pd.DataFrame({"problem_id": list(items), "a": a, "b": b,
                         "n_responses": n_resp.reindex(items).to_numpy()})
    return out, theta


def protocol_auc(items: pd.DataFrame, holdout: pd.DataFrame) -> float:
    """First-half EAP ability -> predict second-half responses, pooled AUC."""
    par = items.set_index("problem_id")
    known = holdout[holdout["problem_id"].isin(par.index)]
    first, second = halves(known)
    preds, ys = [], []
    for uid, sec in second.groupby("user_id"):
        fir = first[first["user_id"] == uid]
        th = eap_theta(par.loc[fir["problem_id"], "a"].to_numpy(),
                       par.loc[fir["problem_id"], "b"].to_numpy(),
                       fir["correct"].to_numpy())
        preds.append(p_correct(th, par.loc[sec["problem_id"], "a"].to_numpy(),
                               par.loc[sec["problem_id"], "b"].to_numpy()))
        ys.append(sec["correct"].to_numpy())
    return float(roc_auc_score(np.concatenate(ys), np.concatenate(preds)))


def run(cfg: dict, log: bool = True) -> dict:
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    m = cfg["model"]
    items, theta = fit_2pl(df[df.split == "train"], m["l2"], m["outer_iters"])
    metrics = {"val_auc": protocol_auc(items, df[df.split == "val"])}
    if not log:
        return metrics
    with tracking.start(cfg) as r:
        mlflow.log_params({f"model.{k}": v for k, v in m.items()})
        mlflow.log_params({"n_items": len(items),
                           "n_train_learners": int((df.split == "train").sum())})
        mlflow.log_metrics(metrics)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        joblib.dump({"items": items, "param_version": 1}, ARTIFACTS / "model.joblib")
        mlflow.log_dict(items.to_dict("list"), "item_parameters.json")
        mlflow.sklearn.log_model(items, "model")  # registered artifact
        mlflow.log_artifact(str(ARTIFACTS / "model.joblib"), "bundle")
        tracking.save_run_id(r.info.run_id)
        metrics["run_id"] = r.info.run_id
        return metrics


def main():
    print(json.dumps(run(load_cfg()), indent=2))


if __name__ == "__main__":
    main()

