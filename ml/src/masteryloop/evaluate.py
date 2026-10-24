"""Two verdicts, both required.
PREDICTIVE: on untouched test learners, first-half EAP -> second-half AUC and Brier,
against the item-mean and two-way classical baselines under the identical protocol.
STATISTICAL: the parameter-recovery harness - simulate learners with KNOWN a and b,
fit them with the same code, and measure how well truth is recovered. A model that
predicts well but recovers garbage parameters would corrupt every adaptive-selection
decision M4 makes; both verdicts gate."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from .config import PROCESSED, ARTIFACTS, load_cfg
from .features import eap_theta, halves, p_correct, simulate
from .baseline import ItemMean, TwoWay
from .train import fit_2pl, protocol_auc
from . import tracking


def model_predictions(items, holdout):
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
    return np.concatenate(ys), np.concatenate(preds), first, second


def recovery(cfg) -> dict:
    sim = cfg["recovery"]
    df, a_true, b_true = simulate(sim["n_learners"], sim["n_items"],
                                  sim["k_per_learner"], seed=sim["seed"])
    items, _ = fit_2pl(df, cfg["model"]["l2"], cfg["model"]["outer_iters"])
    fit = items.set_index("problem_id").loc[np.arange(sim["n_items"])]
    return {"corr_a": float(np.corrcoef(fit["a"], a_true)[0, 1]),
            "rmse_b": float(np.sqrt(np.mean((fit["b"] - b_true) ** 2))),
            "n_items": int(sim["n_items"])}


def main():
    cfg = load_cfg()
    b = joblib.load(ARTIFACTS / "model.joblib")
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    tr, te = df[df.split == "train"], df[df.split == "test"]
    y, p, first, second = model_predictions(b["items"], te)
    model = {"auc": float(roc_auc_score(y, p)),
             "brier": float(brier_score_loss(y, p)),
             "test_learners": int(second.user_id.nunique())}
    yb = second["correct"].to_numpy()
    base = {"auc_item_mean": float(roc_auc_score(
                yb, ItemMean().fit(tr).predict(second, first))),
            "auc_two_way": float(roc_auc_score(
                yb, TwoWay().fit(tr).predict(second, first)))}
    rec = recovery(cfg)
    report = {"run_id": tracking.load_run_id(), "model": model, "baseline": base,
              "recovery": rec,
              "derived": {"auc_gain_vs_item_mean": model["auc"] - base["auc_item_mean"],
                          "auc_gain_vs_two_way": model["auc"] - base["auc_two_way"]}}
    tracking.write_report(report)
    print(f"test AUC 2PL={model['auc']:.4f} two-way={base['auc_two_way']:.4f} "
          f"item-mean={base['auc_item_mean']:.4f} | recovery corr(a)={rec['corr_a']:.3f} "
          f"rmse(b)={rec['rmse_b']:.3f}")


if __name__ == "__main__":
    main()
