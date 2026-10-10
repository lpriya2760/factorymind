"""Quality projection: how a machine's defect rate and scrap cost grow if it keeps running.

A transparent formula rather than a learned model; public machine datasets have no defect labels.

    defect(h) = min(MAX_RATE,
                    BASE_RATE
                    + WEAR_WEIGHT  * min(wear(h), WEAR_CAP)²
                    + DRIFT_WEIGHT * min(max(drift(h) - DRIFT_DEADBAND, 0), DRIFT_CAP)
                    + ERROR_WEIGHT * errors in the last 24h)

    wear(h)  = (hours since the oldest component was replaced + h) / TYPICAL_LIFE_H
    drift(h) = drift now + drift growth per hour (over the last 24h, never negative) * h
    drift    = the largest relative gap between a sensor's 24h average and the machine's own
               normal level (its median 24h average across the uploaded data)

Scrap cost over the horizon = sum over hours of units_per_hour * defect(h) * scrap_cost_per_unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorymind.features import COMPONENTS, SENSORS
from factorymind.report import QualityProjection

BASE_RATE = 0.005  # defect rate of a healthy machine
TYPICAL_LIFE_H = 24.0 * 180  # typical component life
WEAR_WEIGHT = 0.01  # added defect rate at one typical life; grows with the square of wear
WEAR_CAP = 2.0  # wear beyond two lifetimes adds nothing more
DRIFT_DEADBAND = 0.04  # sensor gaps under 4% are normal noise
DRIFT_WEIGHT = 0.5  # added defect rate per unit of drift beyond the deadband
DRIFT_CAP = 0.5
ERROR_WEIGHT = 0.005  # added defect rate per error in the last 24h
MAX_RATE = 0.5

# Shown in the app and README: name -> (value, meaning).
PARAMETERS = {
    "BASE_RATE": (BASE_RATE, "defect rate of a healthy machine"),
    "TYPICAL_LIFE_H": (TYPICAL_LIFE_H, "typical component life, in hours (180 days)"),
    "WEAR_WEIGHT": (WEAR_WEIGHT, "added defect rate at one typical life; grows with wear²"),
    "WEAR_CAP": (WEAR_CAP, "wear beyond this many lifetimes adds nothing more"),
    "DRIFT_DEADBAND": (DRIFT_DEADBAND, "sensor gaps below this are normal noise"),
    "DRIFT_WEIGHT": (DRIFT_WEIGHT, "added defect rate per unit of drift beyond the deadband"),
    "DRIFT_CAP": (DRIFT_CAP, "drift beyond the deadband is capped here"),
    "ERROR_WEIGHT": (ERROR_WEIGHT, "added defect rate per error in the last 24h"),
    "MAX_RATE": (MAX_RATE, "the defect rate never exceeds this"),
}


def project_quality(features: pd.DataFrame, schedule: pd.DataFrame, scrap_cost_per_unit: float,
                    horizon_hours: int) -> dict[int, QualityProjection]:
    """Projection for every machine, from the features at every reading (build_features)."""
    units_per_hour = schedule.groupby("machine_id")["units_per_hour"].sum()
    hours = np.arange(horizon_hours + 1)
    out = {}
    for machine_id, rows in features.groupby("machineID", sort=True):
        rows = rows.sort_values("datetime")
        now = rows.iloc[-1]

        component_age = max(now[f"hours_since_{c}"] for c in COMPONENTS)
        wear = (component_age + hours) / TYPICAL_LIFE_H

        normal = {s: rows[f"{s}_mean_24h"].median() for s in SENSORS}
        gaps_now = {s: _gap(now[f"{s}_mean_24h"], normal[s]) for s in SENSORS}
        sensor = max(gaps_now, key=gaps_now.get)
        before = _row_about_24h_earlier(rows)
        elapsed = (now["datetime"] - before["datetime"]) / pd.Timedelta(hours=1)
        growth = (max(0.0, (gaps_now[sensor] - _gap(before[f"{sensor}_mean_24h"], normal[sensor]))
                      / elapsed) if elapsed > 0 else 0.0)
        drift = gaps_now[sensor] + growth * hours

        errors = int(sum(now[c] for c in features.columns if c.endswith("_count_24h")))
        defect = np.minimum(
            MAX_RATE,
            BASE_RATE
            + WEAR_WEIGHT * np.minimum(wear, WEAR_CAP) ** 2
            + DRIFT_WEIGHT * np.minimum(np.maximum(drift - DRIFT_DEADBAND, 0), DRIFT_CAP)
            + ERROR_WEIGHT * errors,
        )
        units = float(units_per_hour.get(machine_id, 0.0))
        out[int(machine_id)] = QualityProjection(
            horizon_hours=hours.tolist(),
            defect_rate=[round(float(d), 5) for d in defect],
            scrap_cost=round(float(units * defect[1:].sum() * scrap_cost_per_unit), 2),
            units_per_hour=units,
            component_age_hours=float(component_age),
            drift=round(float(gaps_now[sensor]), 4),
            drift_sensor=sensor,
            drift_growth_per_hour=round(float(growth), 5),
            errors_24h=errors,
        )
    return out


def _gap(value: float, normal: float) -> float:
    return abs(value - normal) / normal if normal else 0.0


def _row_about_24h_earlier(rows: pd.DataFrame) -> pd.Series:
    """The latest reading at least 24h before the last one, or the first reading."""
    cutoff = rows["datetime"].iloc[-1] - pd.Timedelta(hours=24)
    earlier = rows[rows["datetime"] <= cutoff]
    return earlier.iloc[-1] if len(earlier) else rows.iloc[0]
