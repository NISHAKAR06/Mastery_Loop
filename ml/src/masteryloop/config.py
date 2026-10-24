"""Single source of paths, connection strings and seeds for M2."""
import os
import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]  # repo root
RAW = ROOT / "data" / "raw"  # DVC-tracked in M1
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

_default_user = os.getenv("DB_USER", os.getenv("POSTGRES_USER", "mastery"))
_default_pass = os.getenv("DB_PASS", os.getenv("POSTGRES_PASSWORD", "mastery_dev_2026"))
_default_host = os.getenv("DB_HOST", "127.0.0.1")
_default_port = os.getenv("DB_PORT", "5432")
_default_db = os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "masteryloop"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+psycopg2://{_default_user}:{_default_pass}@{_default_host}:{_default_port}/{_default_db}",
)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "masteryloop-irt-2pl"
SEED = 42


def load_cfg(path=None) -> dict:
    with open(path or CONFIGS / "train.yaml") as fh:
        return yaml.safe_load(fh)
