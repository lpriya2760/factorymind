"""Replay: step through a bundle's telemetry hour by hour, as if it were arriving live.

The model's features only look back (rolling 3h and 24h windows, errors in the last 24h, time
since each replacement), so the features at hour h are exactly what a prediction made at hour h
would see. A Timeline therefore builds the features once and scores every hour in one pass;
stepping through it is the same as re-running the prediction on the data up to each hour.

`files_until` cuts a bundle at an hour, so the full analysis (and Gemini) can see the plant
exactly as it was when the risk crossed the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from factorymind.bundle import MANIFEST, canonical_files, load_bundle
from factorymind.features import COMPONENTS, build_features
from factorymind.predict import HIGH_RISK, assess, predict_proba
from factorymind.report import MachineSummary
from factorymind.synthetic import to_csv_bytes

DEFAULT_REPLAY_H = 48
# Tables with timestamps: what hadn't happened yet at the replay hour is dropped.
TIMED_TABLES = ("telemetry", "errors", "maint")


@dataclass(frozen=True)
class Crossing:
    step: int
    at: pd.Timestamp
    component: str
    probability: float


@dataclass(frozen=True)
class Timeline:
    hours: list[pd.Timestamp]  # the replayed hours, oldest first
    features: pd.DataFrame  # one row per machine and replayed hour
    risk: pd.DataFrame  # machineID, datetime, comp1..comp4
    machines: pd.DataFrame  # machineID -> model, age

    def __len__(self) -> int:
        return len(self.hours)

    def machines_at(self, step: int) -> list[MachineSummary]:
        """Every machine's risk and drivers as predicted at hour `step`, riskiest first: what
        analyze() would report if the data ended then."""
        rows = self.features[self.features["datetime"] == self.hours[step]].reset_index(drop=True)
        summaries = [
            MachineSummary(machine_id=int(m), model=str(self.machines.at[m, "model"]),
                           age=int(self.machines.at[m, "age"]), component_risk=a.component_risk,
                           top_drivers=a.top_drivers, risk_level=a.risk_level)
            for m, a in zip(rows["machineID"], assess(rows))
        ]
        return sorted(summaries, key=lambda s: (-s.max_risk, s.machine_id))

    def history(self, machine_id: int, step: int) -> pd.DataFrame:
        """A machine's component risks from the first replayed hour up to `step`."""
        r = self.risk[(self.risk["machineID"] == machine_id)
                      & (self.risk["datetime"] <= self.hours[step])]
        return r.set_index("datetime")[COMPONENTS]

    def first_crossing(self, machine_id: int, threshold: float = HIGH_RISK) -> Crossing | None:
        """The first replayed hour at which any of the machine's components reaches `threshold`."""
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        r = self.risk[self.risk["machineID"] == machine_id].reset_index(drop=True)
        over = r[COMPONENTS].max(axis=1) >= threshold
        if not over.any():
            return None
        row = r[over].iloc[0]
        component = max(COMPONENTS, key=lambda c: row[c])
        return Crossing(self.hours.index(row["datetime"]), row["datetime"], component,
                        float(row[component]))


def risk_timeline(files: Mapping[str, bytes], hours: int = DEFAULT_REPLAY_H) -> Timeline:
    """Score every machine at each of the bundle's last `hours` telemetry hours."""
    if hours < 1:
        raise ValueError("hours must be at least 1")
    tables = load_bundle(files).tables
    features = build_features(tables)
    all_hours = sorted(features["datetime"].unique())
    replayed = [pd.Timestamp(h) for h in all_hours[-hours:]]
    window = features[features["datetime"] >= replayed[0]].reset_index(drop=True)
    risk = pd.concat([window[["machineID", "datetime"]], predict_proba(window)[COMPONENTS]],
                     axis=1)
    return Timeline(hours=replayed, features=window, risk=risk,
                    machines=tables["machines"].set_index("machineID")[["model", "age"]])


def files_until(files: Mapping[str, bytes], at: pd.Timestamp) -> dict[str, bytes]:
    """The bundle as it stood at hour `at`: later telemetry, errors and maintenance dropped.
    The business tables and the manifest are kept as they are."""
    tables = dict(load_bundle(files).tables)
    for name in TIMED_TABLES:
        tables[name] = tables[name][tables[name]["datetime"] <= at]
    out = to_csv_bytes(tables)
    manifest = canonical_files(files).get(MANIFEST)
    if manifest is not None:
        out[MANIFEST] = manifest
    return out
