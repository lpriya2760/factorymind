"""Model features for each machine and telemetry hour. Training and inference share this code."""

from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS = ["volt", "rotate", "pressure", "vibration"]
ERRORS = [f"error{i}" for i in range(1, 6)]
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
MODELS = ["model1", "model2", "model3", "model4"]
WINDOWS_H = (3, 24)
# Used when a component has no replacement on record.
NO_REPLACEMENT_HOURS = 24.0 * 365

FEATURES = (
    [f"{s}_mean_{w}h" for w in WINDOWS_H for s in SENSORS]
    + [f"{s}_std_{w}h" for w in WINDOWS_H for s in SENSORS]
    + [f"{e}_count_24h" for e in ERRORS]
    + [f"hours_since_{c}" for c in COMPONENTS]
    + [f"is_{m}" for m in MODELS]
    + ["age"]
)


def build_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per telemetry reading: machineID, datetime and the FEATURES columns.

    Needs the telemetry, errors, maint and machines tables (as parsed by load_bundle).
    """
    tel = (tables["telemetry"][["machineID", "datetime", *SENSORS]]
           .sort_values(["machineID", "datetime"], kind="stable")
           .reset_index(drop=True))
    out = tel[["machineID", "datetime"]].copy()

    by_machine = tel.set_index("datetime").groupby("machineID", sort=True)[SENSORS]
    for w in WINDOWS_H:
        rolling = by_machine.rolling(f"{w}h")
        means, stds = rolling.mean(), rolling.std().fillna(0.0)  # std of one reading is 0
        for s in SENSORS:
            out[f"{s}_mean_{w}h"] = means[s].to_numpy()
            out[f"{s}_std_{w}h"] = stds[s].to_numpy()

    out = out.join(_error_counts(out, tables["errors"]))
    out = out.join(_hours_since_replacement(out, tables["maint"]))

    machines = tables["machines"].set_index("machineID")
    model = out["machineID"].map(machines["model"])
    for m in MODELS:
        out[f"is_{m}"] = (model == m).astype("int64")
    out["age"] = out["machineID"].map(machines["age"]).astype("float64")
    return out[["machineID", "datetime", *FEATURES]]


def latest_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Features at each machine's most recent telemetry reading, one row per machine."""
    return build_features(tables).groupby("machineID", sort=True).tail(1).reset_index(drop=True)


def _per_machine_event_times(events: pd.DataFrame, kind_col: str, kind: str):
    sel = events[events[kind_col] == kind]
    return {m: np.sort(g["datetime"].to_numpy()) for m, g in sel.groupby("machineID")}


def _error_counts(grid: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """Errors of each type in the 24h up to and including each reading."""
    counts = pd.DataFrame(0, index=grid.index, columns=[f"{e}_count_24h" for e in ERRORS])
    window = np.timedelta64(24, "h")
    for e in ERRORS:
        times = _per_machine_event_times(errors, "errorID", e)
        col = counts[f"{e}_count_24h"].to_numpy().copy()
        for m, idx in grid.groupby("machineID").indices.items():
            if m in times:
                t = grid["datetime"].to_numpy()[idx]
                col[idx] = (np.searchsorted(times[m], t, "right")
                            - np.searchsorted(times[m], t - window, "right"))
        counts[f"{e}_count_24h"] = col
    return counts


def _hours_since_replacement(grid: pd.DataFrame, maint: pd.DataFrame) -> pd.DataFrame:
    """Hours since each component was last replaced, as of each reading."""
    out = pd.DataFrame(NO_REPLACEMENT_HOURS, index=grid.index,
                       columns=[f"hours_since_{c}" for c in COMPONENTS])
    for c in COMPONENTS:
        times = _per_machine_event_times(maint, "comp", c)
        col = out[f"hours_since_{c}"].to_numpy().copy()
        for m, idx in grid.groupby("machineID").indices.items():
            if m not in times:
                continue
            t = grid["datetime"].to_numpy()[idx]
            last = np.searchsorted(times[m], t, "right") - 1
            known = last >= 0
            col[idx[known]] = (t[known] - times[m][last[known]]) / np.timedelta64(1, "h")
        out[f"hours_since_{c}"] = col
    return out
