"""Train the component-failure model in BigQuery ML and export it into the app.

1. Build features (factorymind.features) for every machine-hour in data/raw, and label each with
   the component that fails in the next 24h ("none" if none does).
2. Keep every 3rd hour. Train on readings before TRAIN_END; hold out readings from TEST_START
   (a day later, so no 24h label window crosses into the test period).
3. Upload the training rows to BigQuery and train a BOOSTED_TREE_CLASSIFIER.
4. Export the model to Cloud Storage and download it into factorymind/model/.
5. Evaluate the exported model on the held-out period and write factorymind/model/metrics.json.

Usage (needs gcloud signed in and the Kaggle dataset in data/raw/):
    uv run python scripts/train_model.py [--project P] [--region asia-southeast1]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factorymind.features import COMPONENTS, FEATURES, build_features  # noqa: E402

TRAIN_END = pd.Timestamp("2015-09-01")
TEST_START = pd.Timestamp("2015-09-02")
SAMPLE_EVERY_H = 3
DATASET = "factorymind"
TABLE = "train_features"
MODEL = "failure_model"
MODEL_DIR = ROOT / "factorymind" / "model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--project", default=None, help="defaults to gcloud's current project")
    parser.add_argument("--region", default="asia-southeast1")
    parser.add_argument("--skip-train", action="store_true",
                        help="only re-evaluate the model already in factorymind/model/")
    args = parser.parse_args()
    project = args.project or run(["gcloud", "config", "get-value", "project"]).strip()

    print("Building features and labels...")
    data = labelled_features(args.raw)
    train = data[data["datetime"] < TRAIN_END]
    test = data[data["datetime"] >= TEST_START]
    print(f"  train {len(train):,} rows, test {len(test):,} rows")
    print("  train labels:", train["label"].value_counts().to_dict())

    if not args.skip_train:
        train_in_bigquery(train, project, args.region)
        export_model(project, args.region)

    # Imported here so training works before the inference module exists.
    from factorymind.predict import evaluate
    metrics = evaluate(test[FEATURES], test["label"])
    metrics |= {
        "train_period": f"{data['datetime'].min():%Y-%m-%d} to {TRAIN_END:%Y-%m-%d}",
        "test_period": f"{TEST_START:%Y-%m-%d} to {data['datetime'].max():%Y-%m-%d}",
        "train_rows": len(train), "test_rows": len(test),
        "sample_every_hours": SAMPLE_EVERY_H,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def labelled_features(raw: Path) -> pd.DataFrame:
    tables = {}
    for name in ["telemetry", "errors", "maint", "failures", "machines"]:
        df = pd.read_csv(raw / f"PdM_{name}.csv")
        if "datetime" in df:
            df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
        tables[name] = df
    features = build_features(tables)
    features = features[features["datetime"].dt.hour % SAMPLE_EVERY_H == 0].reset_index(drop=True)
    features["label"] = label_next_24h(features, tables["failures"])
    return features


def label_next_24h(features: pd.DataFrame, failures: pd.DataFrame) -> np.ndarray:
    """The component whose failure comes first in the 24h after each reading, else "none".
    Simultaneous failures on one machine are broken by component name."""
    f = (failures.sort_values(["datetime", "failure"])
         .drop_duplicates(["machineID", "datetime"]))
    labels = np.full(len(features), "none", dtype=object)
    t_all = features["datetime"].to_numpy()
    for m, idx in features.groupby("machineID").indices.items():
        fm = f[f["machineID"] == m]
        if fm.empty:
            continue
        times, comps = fm["datetime"].to_numpy(), fm["failure"].to_numpy()
        t = t_all[idx]
        nxt = np.searchsorted(times, t, "right")  # first failure strictly after t
        has = nxt < len(times)
        soon = np.zeros(len(t), dtype=bool)
        soon[has] = times[nxt[has]] - t[has] <= np.timedelta64(24, "h")
        labels[idx[soon]] = comps[nxt[soon]]
    assert set(labels) <= {"none", *COMPONENTS}
    return labels


def train_in_bigquery(train: pd.DataFrame, project: str, region: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.parquet"
        train[[*FEATURES, "label"]].to_parquet(path, index=False)
        print(f"Uploading {path.stat().st_size / 2**20:.1f} MB to BigQuery...")
        run(["bq", f"--location={region}", "mk", "-f", "--dataset", f"{project}:{DATASET}"])
        run(["bq", f"--location={region}", "load", "--replace", "--source_format=PARQUET",
             f"{project}:{DATASET}.{TABLE}", str(path)])

    print("Training BOOSTED_TREE_CLASSIFIER (a few minutes)...")
    run(["bq", f"--location={region}", "query", "--use_legacy_sql=false", f"""
        CREATE OR REPLACE MODEL `{project}.{DATASET}.{MODEL}`
        OPTIONS (
          model_type = 'BOOSTED_TREE_CLASSIFIER',
          input_label_cols = ['label'],
          auto_class_weights = TRUE,
          max_iterations = 50,
          max_tree_depth = 6,
          subsample = 0.8,
          data_split_method = 'RANDOM',
          data_split_eval_fraction = 0.1,
          early_stop = TRUE
        ) AS
        SELECT * FROM `{project}.{DATASET}.{TABLE}`
    """])


def export_model(project: str, region: str) -> None:
    bucket = f"gs://{project}-factorymind-models"
    if subprocess.run(["gcloud", "storage", "buckets", "describe", bucket],
                      capture_output=True).returncode != 0:
        run(["gcloud", "storage", "buckets", "create", bucket, f"--location={region}",
             "--uniform-bucket-level-access"])
    dest = f"{bucket}/{MODEL}"
    print(f"Exporting model to {dest}...")
    run(["gcloud", "storage", "rm", "-r", "--quiet", f"{dest}/**"], check=False)
    run(["bq", f"--location={region}", "extract", "--destination_format=ML_XGBOOST_BOOSTER",
         "-m", f"{project}:{DATASET}.{MODEL}", dest])
    with tempfile.TemporaryDirectory() as tmp:
        run(["gcloud", "storage", "cp", "-r", f"{dest}/*", tmp])
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        MODEL_DIR.mkdir()
        shutil.copy(Path(tmp) / "assets" / "model_metadata.json", MODEL_DIR)
        # BigQuery exports XGBoost's legacy binary format, which XGBoost 3 can't read.
        # Convert it once with XGBoost 2.1 into JSON, which the app loads.
        run(["uv", "run", "--no-project", "--with", "xgboost==2.1.4", "python", "-c",
             "import sys, xgboost; xgboost.Booster(model_file=sys.argv[1]).save_model(sys.argv[2])",
             str(Path(tmp) / "model.bst"), str(MODEL_DIR / "model.json")])
    print("Saved:", sorted(p.name for p in MODEL_DIR.iterdir()))


def run(cmd: list[str], check: bool = True) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd[:4])}...\n{result.stderr or result.stdout}")
    return result.stdout


if __name__ == "__main__":
    main()
