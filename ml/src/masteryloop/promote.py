"""Registry promotion gate: register the evaluated run, and move the
'champion' alias only if every gate in configs/gates.yaml passes."""
import json
import operator
import sys
import mlflow
import yaml
from mlflow.tracking import MlflowClient
from .config import CONFIGS, REPORTS, MLFLOW_URI, MODEL_NAME

OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}


def lookup(report: dict, path: str):
    node = report
    for part in path.split("."):
        node = node[part]
    return float(node)


def check(report: dict) -> list[tuple[str, float, str, float, bool]]:
    gates = yaml.safe_load((CONFIGS / "gates.yaml").read_text())["gates"]
    out = []
    for g in gates:
        val = lookup(report, g["path"])
        out.append((g["path"], val, g["op"], g["threshold"], OPS[g["op"]](val, g["threshold"])))
    return out


def main():
    report = json.loads((REPORTS / "eval.json").read_text())
    results = check(report)
    for path, val, op, thr, ok in results:
        print(f"{'PASS' if ok else 'FAIL'} {path:<40} {val:>10.4f} {op} {thr}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{report['run_id']}/model", MODEL_NAME)
    passed = all(r[-1] for r in results)
    client.set_model_version_tag(MODEL_NAME, mv.version, "gates", "passed" if passed else "failed")
    if not passed:
        print(f"v{mv.version} registered but NOT promoted - champion unchanged.")
        sys.exit(1)
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print(f"{MODEL_NAME} v{mv.version} -> champion")


if __name__ == "__main__":
    main()
