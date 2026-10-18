"""Real-time risk monitoring: score machines as readings arrive and alert the moment a component
becomes risky.

A RiskMonitor holds a rolling window of each machine's recent telemetry and errors. ingest()
adds new readings, re-scores the machines that got data with the failure model, and returns a
RiskEvent for every machine that has just crossed into the alert level (high by default). Each
event is enriched with the full analysis (expected loss and best action) and passed to `notify`,
e.g. send_slack_event.

An AlertLedger keeps it to one alert per machine and component per risk episode: a machine is
alerted again only after it drops back below the alert level (and not within a cooldown of the
last alert, so it can't flap).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Hashable, Mapping

import pandas as pd

from factorymind.alerts import LEVELS
from factorymind.bundle import load_bundle
from factorymind.engine import analyze
from factorymind.features import build_features
from factorymind.predict import assess
from factorymind.synthetic import to_csv_bytes

log = logging.getLogger(__name__)

WINDOW_H = 48  # readings kept per machine; the model's features look back 24h
DEFAULT_COOLDOWN = timedelta(hours=6)


class AlertLedger:
    """What has been alerted, so the same risk isn't alerted twice.

    A key (e.g. (machine_id, component)) is alerted once when it becomes risky, then stays quiet
    while it stays risky. Once it recovers (`clear`), it can alert again, but not within
    `cooldown` of the last alert, so a machine hovering at the threshold doesn't flap. `now` is
    passed in, so the stream can use the data's own clock and the app the wall clock.
    """

    def __init__(self, cooldown: timedelta = DEFAULT_COOLDOWN):
        self.cooldown = cooldown
        self._sent: dict[Hashable, datetime] = {}
        self._active: set[Hashable] = set()

    def should_alert(self, key: Hashable, now: datetime) -> bool:
        if key in self._active:
            return False
        last = self._sent.get(key)
        return last is None or now - last >= self.cooldown

    def record(self, key: Hashable, now: datetime) -> None:
        self._sent[key] = now
        self._active.add(key)

    def sent_at(self, key: Hashable) -> datetime | None:
        return self._sent.get(key)

    def clear(self, matches: Callable[[Hashable], bool]) -> None:
        """Re-arm the keys `matches` picks (e.g. a machine that recovered)."""
        self._active -= {k for k in self._active if matches(k)}


@dataclass(frozen=True)
class RiskEvent:
    machine_id: int
    model: str
    component: str
    failure_probability: float  # that component, next 24h
    risk_level: str
    drivers: list[str]
    at: datetime  # time of the reading that triggered it
    expected_loss: float  # USD, from the cascade
    best_action: str
    action_cost: float
    net_benefit: float  # 0 when the best action is to do nothing
    delivery_error: str | None = None  # set when notify failed


Notify = Callable[[RiskEvent], None]


@dataclass
class RiskMonitor:
    """Scores machines as readings stream in.

    `reference` is a bundle (file name -> contents) giving the machines, maintenance history and
    business tables; its telemetry and errors seed the rolling window.
    """
    reference: Mapping[str, bytes]
    notify: Notify | None = None
    assumptions: Mapping[str, float] | None = None
    min_level: str = "high"
    ledger: AlertLedger = field(default_factory=AlertLedger)

    def __post_init__(self):
        if self.min_level not in ("high", "medium"):
            raise ValueError("min_level must be 'high' or 'medium'")
        # The latest score of each machine: (riskiest component, its probability, risk level).
        self.scores: dict[int, tuple[str, float, str]] = {}
        tables = load_bundle(self.reference).tables
        self._static = {k: v for k, v in tables.items() if k not in ("telemetry", "errors")}
        self._telemetry = tables["telemetry"]
        self._errors = tables["errors"]
        self._trim()

    def ingest(self, telemetry: pd.DataFrame,
               errors: pd.DataFrame | None = None) -> list[RiskEvent]:
        """Add readings (columns as in telemetry.csv / errors.csv) and return the new risks."""
        telemetry = _typed(telemetry, "datetime")
        self._telemetry = pd.concat([self._telemetry, telemetry], ignore_index=True)
        if errors is not None and len(errors):
            self._errors = pd.concat([self._errors, _typed(errors, "datetime")],
                                     ignore_index=True)
        self._trim()

        latest = (build_features({**self._static, "telemetry": self._telemetry,
                                  "errors": self._errors})
                  .groupby("machineID").tail(1))
        latest = latest[latest["machineID"].isin(set(telemetry["machineID"]))]
        crossed = []
        for (_, row), a in zip(latest.iterrows(), assess(latest.reset_index(drop=True))):
            machine_id, at = int(row["machineID"]), row["datetime"].to_pydatetime()
            component = max(a.component_risk, key=a.component_risk.get)
            self.scores[machine_id] = (component, a.component_risk[component], a.risk_level)
            if LEVELS[a.risk_level] < LEVELS[self.min_level]:
                self.ledger.clear(lambda k, m=machine_id: k[0] == m)  # recovered: re-arm
                continue
            if self.ledger.should_alert((machine_id, component), at):
                crossed.append((machine_id, component, at, a))
        if not crossed:
            return []

        report = analyze(self._window_files(), self.assumptions)
        models = {m.machine_id: m.model for m in report.machines}
        events = []
        for machine_id, component, at, a in crossed:
            best = report.interventions[machine_id][0]
            acting = best.lever != "do_nothing"
            event = RiskEvent(
                machine_id=machine_id, model=models[machine_id], component=component,
                failure_probability=a.component_risk[component], risk_level=a.risk_level,
                drivers=[d.description for d in a.top_drivers], at=at,
                expected_loss=report.cascade[machine_id].expected_loss,
                best_action=best.label if acting else "Do nothing (no action pays for itself)",
                action_cost=best.cost if acting else 0.0,
                net_benefit=best.net_benefit if acting else 0.0)
            if self.notify:
                try:
                    self.notify(event)
                except Exception as exc:  # noqa: BLE001 - reported on the event and logged
                    log.warning("alert for machine %s failed: %s", machine_id, exc)
                    event = _with_error(event, str(exc))
            self.ledger.record((machine_id, component), at)
            log.info("risk event: machine %s %s %.0f%% at %s", machine_id, component,
                     100 * event.failure_probability, at)
            events.append(event)
        return events

    def _trim(self) -> None:
        cutoff = self._telemetry["datetime"].max() - pd.Timedelta(hours=WINDOW_H)
        self._telemetry = self._telemetry[self._telemetry["datetime"] > cutoff]
        self._errors = self._errors[self._errors["datetime"] > cutoff]

    def _window_files(self) -> dict[str, bytes]:
        return to_csv_bytes({**self._static, "telemetry": self._telemetry,
                             "errors": self._errors})


def _typed(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column])
    return df


def _with_error(event: RiskEvent, error: str) -> RiskEvent:
    from dataclasses import replace
    return replace(event, delivery_error=error)


def replay(files: Mapping[str, bytes], hours: int, notify: Notify | None = None,
           min_level: str = "high",
           on_hour: Callable[[datetime, RiskMonitor, list[RiskEvent]], None] | None = None,
           ) -> list[RiskEvent]:
    """Stream the last `hours` hours of a bundle into a RiskMonitor, one hour at a time, as if
    live; everything before seeds the monitor. Returns every RiskEvent raised."""
    tables = load_bundle(files).tables
    tel, err = tables["telemetry"], tables["errors"]
    start = tel["datetime"].max() - pd.Timedelta(hours=hours)
    seed = {**tables, "telemetry": tel[tel["datetime"] <= start],
            "errors": err[err["datetime"] <= start]}
    monitor = RiskMonitor(to_csv_bytes(seed), notify=notify, min_level=min_level)
    events: list[RiskEvent] = []
    for at in sorted(tel.loc[tel["datetime"] > start, "datetime"].unique()):
        new = monitor.ingest(tel[tel["datetime"] == at], err[err["datetime"] == at])
        events += new
        if on_hour:
            on_hour(pd.Timestamp(at).to_pydatetime(), monitor, new)
    return events
