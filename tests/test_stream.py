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

"""Real-time risk monitoring and Slack alerts, offline (a fake Slack transport)."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from factorymind.alerts import format_slack_event, slack_notifier
from factorymind.stream import AlertLedger, RiskMonitor, replay as risk_replay

DEMOS = Path(__file__).parent.parent / "demo_bundles"
T0 = datetime(2015, 5, 19, 6)


def bundle(name):
    path = DEMOS / f"{name}.zip"
    return {path.name: path.read_bytes()}


# --- the ledger -------------------------------------------------------------------------------

def test_one_alert_per_episode():
    ledger = AlertLedger(timedelta(hours=6))
    assert ledger.should_alert("m90", T0)
    ledger.record("m90", T0)
    assert not ledger.should_alert("m90", T0 + timedelta(hours=30))  # still risky: quiet


def test_recovery_re_arms_but_not_within_the_cooldown():
    ledger = AlertLedger(timedelta(hours=6))
    ledger.record(("m90", "comp2"), T0)
    ledger.clear(lambda k: k[0] == "m90")  # recovered
    assert not ledger.should_alert(("m90", "comp2"), T0 + timedelta(hours=2))  # flapping
    assert ledger.should_alert(("m90", "comp2"), T0 + timedelta(hours=7))
    assert ledger.sent_at(("m90", "comp2")) == T0


# --- streaming the demo bundles -------------------------------------------------------------------

@pytest.fixture(scope="module")
def comp2_stream():
    sent = []
    events = risk_replay(bundle("comp2_failure_building"), hours=24,
                         notify=slack_notifier("https://hooks.example/x",
                                               post=lambda url, body: sent.append((url, body))))
    return events, sent


def test_streaming_comp2_raises_exactly_one_alert_a_day_ahead(comp2_stream):
    events, sent = comp2_stream
    assert len(events) == 1 and len(sent) == 1
    e = events[0]
    assert (e.machine_id, e.component, e.risk_level) == (90, "comp2", "high")
    # The real failure is at 2015-05-20 06:00: the alert comes 24h earlier, from streamed data.
    assert e.at == datetime(2015, 5, 19, 6)
    assert e.failure_probability > 0.99 and e.drivers and e.expected_loss > 0
    assert e.best_action == "Replace comp2, machine 47 covers" and e.net_benefit > 0
    assert e.delivery_error is None


def test_the_slack_message_says_what_and_what_to_do(comp2_stream):
    (url, body), = comp2_stream[1]
    payload = json.loads(body)
    assert url == "https://hooks.example/x"
    assert "Machine 90" in payload["text"] and "comp2" in payload["text"] and ">99%" in payload["text"]
    text = json.dumps(payload["blocks"])
    assert "Replace comp2, machine 47 covers" in text and "Expected loss" in text
    assert "rotate" in text  # a driver


def test_healthy_fleet_raises_no_alerts():
    assert risk_replay(bundle("healthy_fleet"), hours=24) == []


def test_two_machines_at_risk_each_alert_once():
    events = risk_replay(bundle("two_machines_at_risk"), hours=24)
    assert sorted((e.machine_id, e.component) for e in events) == [(92, "comp3"), (95, "comp4")]


def test_a_failed_delivery_is_reported_not_retried_every_hour():
    def broken(event):
        raise RuntimeError("Slack is down")

    events = risk_replay(bundle("comp2_failure_building"), hours=24, notify=broken)
    assert len(events) == 1 and events[0].delivery_error == "Slack is down"


def test_bad_level_is_rejected():
    with pytest.raises(ValueError):
        RiskMonitor(bundle("healthy_fleet"), min_level="low")


def test_format_slack_event_has_fallback_text(comp2_stream):
    payload = format_slack_event(comp2_stream[0][0])
    assert payload["text"] and payload["blocks"][0]["type"] == "header"


# --- deployment -------------------------------------------------------------------------------------

def test_deploy_passes_alert_secrets_only_through_secret_manager():
    script = (Path(__file__).parent.parent / "scripts" / "deploy.sh").read_text()
    env_loop = script[script.index("for name in"):script.index("done")]
    assert "SLACK_WEBHOOK_URL" not in env_loop and "SMTP_PASSWORD" not in env_loop
    assert "--set-secrets" in script and "factorymind-slack-webhook" in script
