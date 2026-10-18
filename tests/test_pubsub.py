"""Live monitoring over Pub/Sub: replaying real readings through the monitor (in memory)."""

import json
import random
from pathlib import Path

import pandas as pd
import pytest

from factorymind.bundle import load_bundle
from factorymind.engine import analyze
from factorymind.pubsub import (ALERTS_TOPIC, TELEMETRY_TOPIC, LiveMonitor, LocalBus,
                                alert_message, decode, encode, replay, split_bundle)

DEMO = Path(__file__).parent.parent / "demo_bundles" / "comp2_failure_building.zip"
REPLAY_HOURS = 20  # machine 90 turns high-risk 12h before the end of the data


@pytest.fixture(scope="module")
def files():
    return {DEMO.name: DEMO.read_bytes()}


@pytest.fixture(scope="module")
def split(files):
    return split_bundle(files, REPLAY_HOURS)


def run(history, events):
    statuses = []
    monitor = LiveMonitor(history, on_status=lambda s, r: statuses.append(s))
    for event in events:
        monitor.ingest(event)
    return monitor, statuses


@pytest.fixture(scope="module")
def streamed(split):
    return run(*split)


# --- splitting ------------------------------------------------------------------------------

def test_split_keeps_every_reading_exactly_once(files, split):
    history, events = split
    full = load_bundle(files).tables
    past = load_bundle(history).tables
    streamed_tel = sum(e["type"] == "telemetry" for e in events)
    assert len(past["telemetry"]) + streamed_tel == len(full["telemetry"])
    assert len(past["errors"]) + sum(e["type"] == "errors" for e in events) == len(full["errors"])
    assert len({e["datetime"] for e in events}) == REPLAY_HOURS
    # Context tables are untouched.
    pd.testing.assert_frame_equal(past["orders"], full["orders"])


def test_events_are_in_time_order_with_errors_first(split):
    _, events = split
    keys = [(e["datetime"], e["type"] != "errors") for e in events]
    assert keys == sorted(keys)


def test_events_round_trip_as_json(split):
    event = split[1][0]
    assert decode(encode(event)) == event


# --- replay ---------------------------------------------------------------------------------

def test_replay_publishes_everything_and_paces_by_hour(split):
    _, events = split
    sent, sleeps = [], []
    count = replay(events, lambda d, a: sent.append((d, a)), seconds_per_hour=2,
                   sleep=sleeps.append)
    assert count == len(sent) == len(events)
    assert sleeps == [2] * (REPLAY_HOURS - 1)
    assert sent[0][1]["type"] in ("telemetry", "errors")
    assert all(isinstance(v, str) for _, attrs in sent for v in attrs.values())


# --- the monitor ----------------------------------------------------------------------------

def test_one_status_per_streamed_hour(streamed):
    _, statuses = streamed
    assert len(statuses) == REPLAY_HOURS


def test_machine_90_alerts_once_when_it_turns_high(streamed):
    _, statuses = streamed
    alerts = [s for s in statuses if s.alert]
    assert len(alerts) == 1
    first = alerts[0]
    assert [m.machine_id for m in first.alert.machines] == [90]
    before = statuses[statuses.index(first) - 1]
    assert before.risk[90] != "high" and first.risk[90] == "high"
    # It stays high afterwards without alerting again.
    assert all(s.risk[90] == "high" for s in statuses[statuses.index(first):])


def test_streamed_result_matches_analysing_the_full_upload(files, streamed):
    monitor, _ = streamed
    full = analyze(files)
    assert ([(m.machine_id, m.risk_level) for m in monitor.report.machines]
            == [(m.machine_id, m.risk_level) for m in full.machines])
    assert monitor.report.cascade[90].expected_loss == full.cascade[90].expected_loss


def test_duplicates_and_out_of_order_delivery_give_the_same_result(split, streamed):
    history, events = split
    rng = random.Random(0)
    messy = events + rng.sample(events, 40)  # Pub/Sub may redeliver
    rng.shuffle(messy)
    _, statuses = run(history, messy)
    _, clean = streamed
    assert [(s.as_of, s.risk) for s in statuses] == [(s.as_of, s.risk) for s in clean]
    assert [s.as_of for s in statuses if s.alert] == [s.as_of for s in clean if s.alert]


def test_unknown_event_type_is_rejected(split):
    monitor = LiveMonitor(split[0])
    with pytest.raises(ValueError):
        monitor.ingest({"type": "weather", "machineID": 1, "datetime": "2015-05-19 00:00:00"})


# --- end to end over a bus ------------------------------------------------------------------

def test_pipeline_over_a_bus_publishes_a_json_alert_with_the_loss_range(split):
    history, events = split
    bus = LocalBus()

    def on_status(status, report):
        if status.alert:
            bus.publish(ALERTS_TOPIC, encode(alert_message(status.alert, report, status.as_of)))

    monitor = LiveMonitor(history, on_status=on_status)
    bus.subscribe(TELEMETRY_TOPIC, lambda d, a: monitor.ingest(decode(d)))
    replay(events, lambda d, a: bus.publish(TELEMETRY_TOPIC, d, a), seconds_per_hour=0)

    assert len(bus.messages[TELEMETRY_TOPIC]) == len(events)
    [(data, _)] = bus.messages[ALERTS_TOPIC]
    alert = json.loads(data)
    assert alert["subject"].startswith("FactoryMind: 1 machine at risk")
    m = alert["machines"][0]
    assert m["machine_id"] == 90 and m["top_component"] == "comp2"
    assert m["loss_p10"] <= m["loss_p50"] <= m["loss_p90"]
