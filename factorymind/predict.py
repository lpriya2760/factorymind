"""Runs the exported failure model inside the app (no network calls) and explains its predictions.

The model is a BigQuery ML boosted tree, exported and converted to XGBoost JSON by
scripts/train_model.py. It predicts which component (if any) fails in the next 24h.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from factorymind.features import COMPONENTS, FEATURES
from factorymind.report import Driver

MODEL_DIR = Path(__file__).parent / "model"
CLASSES = ["none", *COMPONENTS]
# Risk levels by the highest component probability. Chosen from the held-out evaluation
# (see model/metrics.json).
HIGH_RISK = 0.5
MEDIUM_RISK = 0.2
N_DRIVERS = 3
# Contributions (log-odds) below this are noise, not reasons worth showing.
MIN_DRIVER_IMPACT = 0.1


@dataclass(frozen=True)
class Assessment:
    component_risk: dict[str, float]
    top_drivers: list[Driver]
    risk_level: str


@cache
def _model() -> tuple[xgb.Booster, list[str]]:
    booster = xgb.Booster(model_file=str(MODEL_DIR / "model.json"))
    meta = json.loads((MODEL_DIR / "model_metadata.json").read_text())
    if meta["feature_names"] != FEATURES:
        raise RuntimeError("The model was trained on different features; rerun train_model.py")
    if sorted(meta["class_names"]) != sorted(CLASSES):
        raise RuntimeError(f"Unexpected model classes {meta['class_names']}")
    return booster, meta["class_names"]


def _matrix(features: pd.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(features[FEATURES].to_numpy(dtype=float))


def predict_proba(features: pd.DataFrame) -> pd.DataFrame:
    """Probability of each outcome in the next 24h, columns none, comp1..comp4."""
    booster, classes = _model()
    probs = booster.predict(_matrix(features))
    return pd.DataFrame(probs, columns=classes, index=features.index)[CLASSES]


def assess(features: pd.DataFrame) -> list[Assessment]:
    """Risk, risk level and the main drivers for each row of features."""
    booster, classes = _model()
    probs = predict_proba(features)
    # Per-class feature contributions (SHAP values): shape (rows, classes, features + bias).
    contribs = booster.predict(_matrix(features), pred_contribs=True)
    fleet_median = features[FEATURES].median()

    out = []
    for i, (_, row) in enumerate(features.iterrows()):
        risk = {c: float(probs.iloc[i][c]) for c in COMPONENTS}
        top = max(risk, key=risk.get)
        impacts = contribs[i, classes.index(top), :-1]
        drivers = [
            Driver(FEATURES[j], describe(FEATURES[j], row[FEATURES[j]], fleet_median[FEATURES[j]]),
                   round(float(impacts[j]), 3))
            for j in np.argsort(-impacts)
            if impacts[j] >= MIN_DRIVER_IMPACT and not _is_absent_error(FEATURES[j], row)
        ][:N_DRIVERS]
        out.append(Assessment(risk, drivers, risk_level(max(risk.values()))))
    return out


def _is_absent_error(feature: str, row: pd.Series) -> bool:
    """'0× error1 in the last 24h' is never a useful reason to give an operator."""
    return feature.endswith("_count_24h") and row[feature] == 0


def risk_level(p: float) -> str:
    return "high" if p >= HIGH_RISK else "medium" if p >= MEDIUM_RISK else "low"


def describe(feature: str, value: float, fleet_median: float) -> str:
    """A feature value in words, compared with the rest of the uploaded fleet."""
    if feature.startswith("hours_since_"):
        return f"{feature.removeprefix('hours_since_')} last replaced {value / 24:.0f} days ago"
    if feature.endswith("_count_24h"):
        return f"{value:.0f}× {feature.removesuffix('_count_24h')} in the last 24h"
    if feature.startswith("is_"):
        return f"machine is {feature.removeprefix('is_')}"
    if feature == "age":
        return f"machine is {value:.0f} years old"
    sensor, stat, window = feature.split("_")
    what = "average" if stat == "mean" else "variability"
    return f"{sensor} {window} {what} {value:.1f} (fleet median {fleet_median:.1f})"


def evaluate(features: pd.DataFrame, labels: pd.Series) -> dict:
    """Per-component AUC, precision and recall, plus how the risk levels perform."""
    probs = predict_proba(features)
    labels = labels.to_numpy()
    predicted = probs.idxmax(axis=1).to_numpy()
    components = {}
    for c in COMPONENTS:
        actual = labels == c
        hit = predicted == c
        components[c] = {
            "auc": _round(_auc(probs[c].to_numpy(), actual)),
            "precision": _round((hit & actual).sum() / hit.sum() if hit.any() else float("nan")),
            "recall": _round((hit & actual).sum() / actual.sum() if actual.any() else float("nan")),
            "support": int(actual.sum()),
        }
    any_failure = labels != "none"
    max_risk = probs[COMPONENTS].max(axis=1).to_numpy()
    levels = {}
    for name, threshold in [("high", HIGH_RISK), ("medium_or_high", MEDIUM_RISK)]:
        flagged = max_risk >= threshold
        levels[name] = {
            "threshold": threshold,
            "share_of_readings_flagged": _round(flagged.mean()),
            "precision": _round((flagged & any_failure).sum() / max(flagged.sum(), 1)),
            "recall": _round((flagged & any_failure).sum() / max(any_failure.sum(), 1)),
        }
    return {
        "components": components,
        "macro_auc": _round(np.nanmean([m["auc"] for m in components.values()])),
        "any_failure_auc": _round(_auc(1 - probs["none"].to_numpy(), any_failure)),
        "risk_levels": levels,
    }


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve (Mann–Whitney U), with ties averaged."""
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _round(x: float) -> float:
    return round(float(x), 4)
