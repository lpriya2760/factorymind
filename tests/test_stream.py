"""Real-time risk monitoring and Slack alerts, offline (a fake Slack transport)."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from factorymind.alerts import format_slack_event, slack_notifier
from factorymind.stream import AlertLedger, RiskMonitor, replay

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
    events = replay(bundle("comp2_failure_building"), hours=24,
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
    assert replay(bundle("healthy_fleet"), hours=24) == []


def test_two_machines_at_risk_each_alert_once():
    events = replay(bundle("two_machines_at_risk"), hours=24)
    assert sorted((e.machine_id, e.component) for e in events) == [(92, "comp3"), (95, "comp4")]


def test_a_failed_delivery_is_reported_not_retried_every_hour():
    def broken(event):
        raise RuntimeError("Slack is down")

    events = replay(bundle("comp2_failure_building"), hours=24, notify=broken)
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
