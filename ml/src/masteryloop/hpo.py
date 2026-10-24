"""Optuna search over the space declared in configs/train.yaml.
Trials call train.run(cfg, log=False) so only the winning config reaches MLflow."""
import argparse
import copy
import optuna
import yaml
from .config import CONFIGS, SEED, load_cfg
from .train import run


def suggest(trial, space: dict) -> dict:
    out = {}
    for name, (kind, lo, hi) in space.items():
        if kind == "int":
            out[name] = trial.suggest_int(name, int(lo), int(hi))
        elif kind == "log":
            out[name] = trial.suggest_float(name, float(lo), float(hi), log=True)
        else:
            out[name] = trial.suggest_float(name, float(lo), float(hi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()
    base = load_cfg()
    h = base["hpo"]

    def objective(trial):
        cfg = copy.deepcopy(base)
        cfg["model"].update(suggest(trial, h["space"]))
        return run(cfg, log=False)[h["metric"]]

    study = optuna.create_study(direction=h["direction"],
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.trials)
    print("best", h["metric"], study.best_value, study.best_params)
    best = copy.deepcopy(base)
    best["model"].update(study.best_params)
    best["run_name"] = "hpo-best"
    (CONFIGS / "train.best.yaml").write_text(yaml.safe_dump(best, sort_keys=False))
    print(run(best, log=True))


if __name__ == "__main__":
    main()
