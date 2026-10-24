"""MLflow helpers shared by train, evaluate, promote and writeback."""
import json
import mlflow
from mlflow.tracking import MlflowClient
from .config import ARTIFACTS, REPORTS, MLFLOW_URI, MODEL_NAME


def start(cfg: dict):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg["experiment"])
    return mlflow.start_run(run_name=cfg.get("run_name", "train"))


def save_run_id(run_id: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "last_run.json").write_text(json.dumps({"run_id": run_id}))


def load_run_id() -> str:
    return json.loads((ARTIFACTS / "last_run.json").read_text())["run_id"]


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def write_report(report: dict) -> None:
    """Persist reports/eval.json and attach every numeric value to the training run."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "eval.json").write_text(json.dumps(report, indent=2, default=float))
    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_id=report["run_id"]):
        mlflow.log_metrics({f"test.{k}": v for k, v in flatten(report).items()})
        mlflow.log_artifact(str(REPORTS / "eval.json"), "evaluation")


def champion() -> tuple[str, str]:
    """(model_version_label, run_id) of the version holding the champion alias."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    return f"{MODEL_NAME}:v{mv.version}", mv.run_id
