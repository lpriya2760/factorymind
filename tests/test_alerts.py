"""Slack and email alerts: built from the AnalysisReport, sent through fake transports."""

import json
from pathlib import Path

import pytest

from factorymind.alerts import (AlertConfig, build_alert, format_email, format_slack,
                                send_alert)
from factorymind.engine import analyze

DEMOS = Path(__file__).parent.parent / "demo_bundles"


def demo(name):
    path = DEMOS / f"{name}.zip"
    return analyze({path.name: path.read_bytes()})


@pytest.fixture(scope="module")
def at_risk():
    return demo("comp2_failure_building")


@pytest.fixture(scope="module")
def healthy():
    return demo("healthy_fleet")


EMAIL_ENV = {"ALERT_EMAIL_TO": "ops@example.com, lead@example.com",
             "ALERT_EMAIL_FROM": "factorymind@example.com",
             "SMTP_HOST": "smtp.example.com", "SMTP_USER": "bot", "SMTP_PASSWORD": "pw"}


class FakeSMTP:
    def __init__(self, host, port, fail=False):
        self.host, self.port, self.fail = host, port, fail
        self.tls = False
        self.login_args = None
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.tls = True

    def login(self, user, password):
        self.login_args = (user, password)

    def send_message(self, msg):
        if self.fail:
            raise OSError("connection refused")
        self.sent.append(msg)


# --- building the alert ----------------------------------------------------------------------

def test_alert_lists_high_risk_machines_with_loss_and_best_action(at_risk):
    alert = build_alert(at_risk)
    high = [m for m in at_risk.machines if m.risk_level == "high"]
    assert [m.machine_id for m in alert.machines] == [m.machine_id for m in high]
    first = alert.machines[0]
    assert first.expected_loss == at_risk.cascade[first.machine_id].expected_loss
    best = at_risk.interventions[first.machine_id][0]
    assert first.best_action == best.label
    assert first.net_benefit == best.net_benefit


def test_no_alert_when_nothing_is_at_risk(healthy):
    assert build_alert(healthy) is None


def test_medium_threshold_includes_medium_machines(at_risk):
    medium_up = [m.machine_id for m in at_risk.machines if m.risk_level in ("medium", "high")]
    assert [m.machine_id for m in build_alert(at_risk, "medium").machines] == medium_up


def test_unknown_level_is_rejected(at_risk):
    with pytest.raises(ValueError):
        build_alert(at_risk, "urgent")


# --- formatting ------------------------------------------------------------------------------

def test_slack_payload_has_fallback_text_and_one_section_per_machine(at_risk):
    alert = build_alert(at_risk)
    payload = format_slack(alert)
    assert "at risk" in payload["text"]
    sections = [b for b in payload["blocks"] if b["type"] == "section"]
    assert len(sections) == len(alert.machines)
    assert f"Machine {alert.machines[0].machine_id}" in sections[0]["text"]["text"]


def test_email_has_subject_recipients_and_body(at_risk):
    alert = build_alert(at_risk)
    msg = format_email(alert, "a@example.com", ["b@example.com", "c@example.com"])
    assert msg["Subject"].startswith("FactoryMind:")
    assert msg["To"] == "b@example.com, c@example.com"
    assert f"Machine {alert.machines[0].machine_id}" in msg.get_content()


# --- configuration ---------------------------------------------------------------------------

def test_config_from_env():
    config = AlertConfig.from_env({"SLACK_WEBHOOK_URL": "https://hooks.example/x", **EMAIL_ENV})
    assert config.channels == ["Slack", "email"]
    assert config.email_to == ["ops@example.com", "lead@example.com"]
    assert config.smtp_port == 587
    assert config.min_level == "high"


def test_nothing_configured_means_no_channels():
    assert AlertConfig.from_env({}).channels == []


def test_email_needs_recipient_sender_and_host():
    assert not AlertConfig.from_env({"ALERT_EMAIL_TO": "ops@example.com"}).email_enabled


def test_bad_min_level_is_rejected():
    with pytest.raises(ValueError):
        AlertConfig.from_env({"ALERT_MIN_LEVEL": "low"})


# --- sending ---------------------------------------------------------------------------------

def test_send_to_slack_and_email(at_risk):
    alert = build_alert(at_risk)
    config = AlertConfig.from_env({"SLACK_WEBHOOK_URL": "https://hooks.example/x", **EMAIL_ENV})
    posts, servers = [], []

    def smtp_factory(host, port):
        servers.append(FakeSMTP(host, port))
        return servers[-1]

    results = send_alert(alert, config, post=lambda url, body: posts.append((url, body)),
                         smtp_factory=smtp_factory)

    assert results == {"Slack": None, "email": None}
    url, body = posts[0]
    assert url == "https://hooks.example/x"
    assert json.loads(body) == format_slack(alert)
    smtp = servers[0]
    assert (smtp.host, smtp.port, smtp.tls, smtp.login_args) == (
        "smtp.example.com", 587, True, ("bot", "pw"))
    assert smtp.sent[0]["To"] == "ops@example.com, lead@example.com"


def test_one_failing_channel_does_not_stop_the_other(at_risk):
    alert = build_alert(at_risk)
    config = AlertConfig.from_env({"SLACK_WEBHOOK_URL": "https://hooks.example/x", **EMAIL_ENV})

    def broken_post(url, body):
        raise RuntimeError("Slack webhook returned HTTP 404")

    sent = []

    def smtp_factory(host, port):
        sent.append(FakeSMTP(host, port))
        return sent[-1]

    results = send_alert(alert, config, post=broken_post, smtp_factory=smtp_factory)
    assert results == {"Slack": "Slack webhook returned HTTP 404", "email": None}
    assert len(sent[0].sent) == 1
