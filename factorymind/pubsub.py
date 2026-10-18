"""Live monitoring over Google Cloud Pub/Sub, fed by replaying real sensor data.

FactoryMind's analysis normally runs on an uploaded bundle. This module runs it on a stream:

    replay publisher ──► topic factorymind-telemetry ──► monitor ──► topic factorymind-alerts
    (real Azure PdM readings,                         (re-analyses each       (JSON alert; Slack /
     hour by hour, sped up)                            completed hour)          email / BigQuery)

- split_bundle() cuts a bundle in two: the history the monitor starts from, and the last
  `replay_hours` of telemetry and error readings as events to publish. The readings are the
  real Azure Predictive Maintenance data, unchanged; only the clock is sped up.
- replay() publishes those events hour by hour.
- LiveMonitor ingests events in any order, and each time every machine has reported for the
  next hour it re-runs engine.analyze() on the history so far. When a machine reaches
  `min_level` risk it emits one alert (again only after it has dropped back below).
- LocalBus runs the whole pipeline in memory (tests, offline demo); PubSubBus uses Google
  Cloud Pub/Sub, or the Pub/Sub emulator when PUBSUB_EMULATOR_HOST is set.

The analysis itself is unchanged: the monitor calls the same analyze() as the app.
"""

from __future__ import annotations

import io
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping

import pandas as pd

from factorymind.alerts import LEVELS, Alert, build_alert, subject
from factorymind.bundle import load_bundle
from factorymind.engine import analyze
from factorymind.report import AnalysisReport

TELEMETRY_TOPIC = "factorymind-telemetry"
ALERTS_TOPIC = "factorymind-alerts"
MONITOR_SUBSCRIPTION = "factorymind-monitor"
STREAMED = ("telemetry", "errors")  # the tables that arrive as events; the rest is context
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# --- messages -------------------------------------------------------------------------------

def encode(event: Mapping) -> bytes:
    return json.dumps(event, separators=(",", ":")).encode()


def decode(data: bytes) -> dict:
    return json.loads(data)


def _events(table: str, df: pd.DataFrame) -> list[dict]:
    out = []
    for row in df.to_dict("records"):
        event = {"type": table, **row}
        event["datetime"] = pd.Timestamp(row["datetime"]).strftime(_TIME_FORMAT)
        event["machineID"] = int(row["machineID"])
        out.append(event)
    return out


# --- splitting a bundle into history + replay ----------------------------------------------

def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, date_format=_TIME_FORMAT).encode()


def split_bundle(files: Mapping[str, bytes], replay_hours: int,
                 ) -> tuple[dict[str, bytes], list[dict]]:
    """(history bundle, events): everything up to `replay_hours` before the end of the data,
    and the telemetry and error readings after that, oldest first (errors before telemetry
    within an hour, as a machine logs an error before its next hourly reading)."""
    if replay_hours < 1:
        raise ValueError("replay_hours must be at least 1")
    tables = load_bundle(files).tables
    end = tables["telemetry"]["datetime"].max()
    cut = end - pd.Timedelta(hours=replay_hours)
    history = {f"{name}.csv": _csv(df) for name, df in tables.items() if name not in STREAMED}
    events = []
    for name in STREAMED:
        df = tables[name]
        history[f"{name}.csv"] = _csv(df[df["datetime"] <= cut])
        events += _events(name, df[df["datetime"] > cut])
    if "manifest.json" in files:
        history["manifest.json"] = files["manifest.json"]
    events.sort(key=lambda e: (e["datetime"], e["type"] != "errors", e["machineID"]))
    return history, events


def replay(events: Iterable[dict], publish: Callable[[bytes, dict], None],
           seconds_per_hour: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> int:
    """Publish events hour by hour, pausing `seconds_per_hour` between hours. Returns the
    number published. `publish(data, attributes)`."""
    count, hour = 0, None
    for event in events:
        if hour is not None and event["datetime"] != hour and seconds_per_hour > 0:
            sleep(seconds_per_hour)
        hour = event["datetime"]
        publish(encode(event), {"type": event["type"], "machine_id": str(event["machineID"]),
                                "datetime": event["datetime"]})
        count += 1
    return count


# --- the monitor ----------------------------------------------------------------------------

@dataclass(frozen=True)
class HourStatus:
    as_of: str
    risk: dict[int, str]  # machine -> risk level
    alert: Alert | None  # machines that newly reached the alert level this hour


def alert_message(alert: Alert, report: AnalysisReport, as_of: str) -> dict:
    """The JSON published to the alerts topic: the alert plus each machine's loss range."""
    machines = []
    for m in alert.machines:
        c = report.cascade[m.machine_id]
        machines.append({**asdict(m), "loss_p10": c.loss_p10, "loss_p50": c.loss_p50,
                         "loss_p90": c.loss_p90})
    return {"as_of": as_of, "subject": subject(alert), "min_level": alert.min_level,
            "total_expected_loss": alert.total_expected_loss, "machines": machines}


class LiveMonitor:
    """Re-analyses the plant as readings arrive and raises an alert when a machine turns risky.

    `history` is a bundle (file name -> bytes) holding the data before the stream starts,
    including the context tables (machines, maintenance, schedule, inventory, orders).
    """

    def __init__(self, history: Mapping[str, bytes], min_level: str = "high",
                 assumptions: Mapping[str, float] | None = None,
                 on_status: Callable[[HourStatus, AnalysisReport], None] | None = None):
        if min_level not in LEVELS:
            raise ValueError(f"min_level must be one of {sorted(LEVELS)}")
        tables = load_bundle(history).tables
        self.context = {name: df for name, df in tables.items() if name not in STREAMED}
        self.manifest = history.get("manifest.json")
        self.data = {name: tables[name].copy() for name in STREAMED}
        self.machines = set(int(m) for m in tables["machines"]["machineID"])
        self.min_level = min_level
        self.assumptions = assumptions
        self.on_status = on_status
        self.last_hour = tables["telemetry"]["datetime"].max()
        self._pending: dict[pd.Timestamp, dict[str, list[dict]]] = defaultdict(
            lambda: {name: [] for name in STREAMED})
        self._seen: set[tuple] = set()
        self._alerted: set[int] = set()
        self.report: AnalysisReport | None = None

    def ingest(self, event: Mapping) -> list[HourStatus]:
        """Add one reading. Returns the status of every hour this completed (usually 0 or 1).
        Duplicates (Pub/Sub delivers at least once) and readings older than the analysed
        history are ignored."""
        table = event["type"] if event["type"] in STREAMED else None
        if table is None:
            raise ValueError(f"unknown event type {event['type']!r}")
        at = pd.Timestamp(event["datetime"])
        key = (table, int(event["machineID"]), at, event.get("errorID"))
        if key in self._seen or at <= self.last_hour:
            return []
        self._seen.add(key)
        row = {k: v for k, v in event.items() if k != "type"}
        row["datetime"] = at
        self._pending[at][table].append(row)
        return self._drain()

    def _complete(self, hour: pd.Timestamp) -> bool:
        reported = {r["machineID"] for r in self._pending[hour]["telemetry"]}
        return self.machines <= reported

    def _drain(self) -> list[HourStatus]:
        out = []
        while self._pending:
            hour = min(self._pending)
            if not self._complete(hour):
                break
            batch = self._pending.pop(hour)
            for name in STREAMED:
                if batch[name]:
                    self.data[name] = pd.concat([self.data[name], pd.DataFrame(batch[name])],
                                                ignore_index=True)
            self.last_hour = hour
            out.append(self._evaluate(hour))
        return out

    def bundle(self) -> dict[str, bytes]:
        files = {f"{name}.csv": _csv(df) for name, df in {**self.context, **self.data}.items()}
        if self.manifest:
            files["manifest.json"] = self.manifest
        return files

    def _evaluate(self, hour: pd.Timestamp) -> HourStatus:
        report = analyze(self.bundle(), self.assumptions)
        self.report = report
        risky = {m.machine_id for m in report.machines
                 if LEVELS[m.risk_level] >= LEVELS[self.min_level]}
        new = risky - self._alerted
        self._alerted = (self._alerted & risky) | new  # re-arm machines that calmed down
        alert = build_alert(report, self.min_level)
        if alert is not None:
            alert = Alert([m for m in alert.machines if m.machine_id in new], self.min_level)
        status = HourStatus(as_of=hour.strftime(_TIME_FORMAT),
                            risk={m.machine_id: m.risk_level for m in report.machines},
                            alert=alert if alert and alert.machines else None)
        if self.on_status:
            self.on_status(status, report)
        return status


# --- transports -----------------------------------------------------------------------------

class LocalBus:
    """In-memory topics: publish appends, subscribers are called synchronously."""

    def __init__(self):
        self.messages: dict[str, list[tuple[bytes, dict]]] = defaultdict(list)
        self._subscribers: dict[str, list[Callable[[bytes, dict], None]]] = defaultdict(list)

    def publish(self, topic: str, data: bytes, attributes: Mapping[str, str] | None = None):
        self.messages[topic].append((data, dict(attributes or {})))
        for callback in self._subscribers[topic]:
            callback(data, dict(attributes or {}))

    def subscribe(self, topic: str, callback: Callable[[bytes, dict], None]) -> None:
        self._subscribers[topic].append(callback)


class PubSubBus:
    """Google Cloud Pub/Sub (or its emulator, via PUBSUB_EMULATOR_HOST)."""

    def __init__(self, project: str):
        from google.cloud import pubsub_v1  # imported here so the app works without it

        self.project = project
        self._publisher = pubsub_v1.PublisherClient()
        self._subscriber = pubsub_v1.SubscriberClient()

    def topic_path(self, topic: str) -> str:
        return self._publisher.topic_path(self.project, topic)

    def publish(self, topic: str, data: bytes, attributes: Mapping[str, str] | None = None):
        """Publish and wait until Pub/Sub has stored the message; returns its ID."""
        return self._publisher.publish(self.topic_path(topic), data,
                                       **dict(attributes or {})).result(timeout=60)

    def subscribe(self, subscription: str, callback: Callable[[bytes, dict], None],
                  timeout: float | None = None) -> None:
        """Stream messages to `callback` until `timeout` seconds pass (None: forever).
        A message is acknowledged only after the callback returns, so a crash redelivers it."""
        path = self._subscriber.subscription_path(self.project, subscription)

        def handle(message):
            callback(message.data, dict(message.attributes))
            message.ack()

        # One callback at a time: the monitor keeps state, so it must not run concurrently.
        from concurrent.futures import ThreadPoolExecutor

        from google.cloud.pubsub_v1.subscriber.scheduler import ThreadScheduler
        from google.cloud.pubsub_v1.types import FlowControl
        future = self._subscriber.subscribe(
            path, handle, flow_control=FlowControl(max_messages=1000),
            scheduler=ThreadScheduler(executor=ThreadPoolExecutor(max_workers=1)))
        with self._subscriber:
            try:
                future.result(timeout=timeout)
            except TimeoutError:
                future.cancel()
                future.result()
            except BaseException:
                future.cancel()
                raise
