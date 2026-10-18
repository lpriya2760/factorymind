"""Slack and email alerts for machines at risk.

build_alert() turns an AnalysisReport into an Alert (or None when no machine is at risk);
send_alert() delivers it to every configured channel. Channels are configured with
environment variables, so nothing secret lives in the code:

- Slack: SLACK_WEBHOOK_URL (an incoming-webhook URL)
- Email: ALERT_EMAIL_TO (comma-separated), ALERT_EMAIL_FROM, SMTP_HOST, SMTP_PORT (default
  587 with STARTTLS; 465 uses implicit TLS), SMTP_USER, SMTP_PASSWORD
- ALERT_MIN_LEVEL: "high" (default) or "medium"

Only the standard library is used, and the senders take injectable transports so the tests
run offline.
"""

from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Callable, Mapping

from factorymind.report import AnalysisReport

LEVELS = {"low": 0, "medium": 1, "high": 2}
LEVEL_EMOJI = {"high": "🔴", "medium": "🟠", "low": "🟢"}


@dataclass(frozen=True)
class MachineAlert:
    machine_id: int
    model: str
    risk_level: str
    top_component: str
    failure_probability: float  # top component, next 24h
    expected_loss: float  # USD, from the cascade
    best_action: str
    action_cost: float
    net_benefit: float  # 0 when the best action is to do nothing


@dataclass(frozen=True)
class Alert:
    machines: list[MachineAlert]  # riskiest first
    min_level: str

    @property
    def total_expected_loss(self) -> float:
        return sum(m.expected_loss for m in self.machines)


@dataclass(frozen=True)
class AlertConfig:
    slack_webhook_url: str | None = None
    email_to: list[str] = field(default_factory=list)
    email_from: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    min_level: str = "high"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AlertConfig":
        env = os.environ if env is None else env
        min_level = env.get("ALERT_MIN_LEVEL", "high").strip().lower()
        if min_level not in ("high", "medium"):
            raise ValueError("ALERT_MIN_LEVEL must be 'high' or 'medium'")
        return cls(
            slack_webhook_url=env.get("SLACK_WEBHOOK_URL") or None,
            email_to=[a.strip() for a in env.get("ALERT_EMAIL_TO", "").split(",") if a.strip()],
            email_from=env.get("ALERT_EMAIL_FROM") or env.get("SMTP_USER") or None,
            smtp_host=env.get("SMTP_HOST") or None,
            smtp_port=int(env.get("SMTP_PORT") or 587),
            smtp_user=env.get("SMTP_USER") or None,
            smtp_password=env.get("SMTP_PASSWORD") or None,
            min_level=min_level,
        )

    @property
    def slack_enabled(self) -> bool:
        return bool(self.slack_webhook_url)

    @property
    def email_enabled(self) -> bool:
        return bool(self.email_to and self.email_from and self.smtp_host)

    @property
    def channels(self) -> list[str]:
        return [name for name, on in (("Slack", self.slack_enabled),
                                      ("email", self.email_enabled)) if on]


def build_alert(report: AnalysisReport, min_level: str = "high") -> Alert | None:
    """The machines at or above min_level, with what a failure would cost and what to do.

    Returns None when no machine qualifies, so callers never send an empty alert.
    """
    if min_level not in LEVELS:
        raise ValueError(f"min_level must be one of {sorted(LEVELS)}")
    machines = []
    for m in report.machines:
        if LEVELS[m.risk_level] < LEVELS[min_level]:
            continue
        best = report.interventions[m.machine_id][0]
        acting = best.lever != "do_nothing"
        machines.append(MachineAlert(
            machine_id=m.machine_id,
            model=m.model,
            risk_level=m.risk_level,
            top_component=m.top_component,
            failure_probability=m.max_risk,
            expected_loss=report.cascade[m.machine_id].expected_loss,
            best_action=best.label if acting else "Do nothing (no action pays for itself)",
            action_cost=best.cost if acting else 0.0,
            net_benefit=best.net_benefit if acting else 0.0,
        ))
    return Alert(machines=machines, min_level=min_level) if machines else None


def _pct(p: float) -> str:
    return ">99%" if p > 0.99 else "<1%" if p < 0.01 else f"{p:.0%}"


def _machine_line(m: MachineAlert) -> str:
    action = m.best_action
    if m.net_benefit > 0:
        action += f" (costs ${m.action_cost:,.0f}, net benefit ${m.net_benefit:,.0f})"
    return (f"Machine {m.machine_id} ({m.model}): {m.top_component} is {_pct(m.failure_probability)}"
            f" likely to fail in the next 24h. Expected loss ${m.expected_loss:,.0f}. "
            f"Best action: {action}.")


def subject(alert: Alert) -> str:
    n = len(alert.machines)
    return (f"FactoryMind: {n} machine{'s' if n != 1 else ''} at risk, "
            f"${alert.total_expected_loss:,.0f} expected loss")


def format_slack(alert: Alert) -> dict:
    """An incoming-webhook payload: plain `text` for notifications, blocks for the message."""
    blocks: list[dict] = [{"type": "header", "text": {"type": "plain_text",
                                                       "text": "🏭 " + subject(alert)}}]
    for m in alert.machines:
        blocks.append({"type": "section", "text": {
            "type": "mrkdwn", "text": f"{LEVEL_EMOJI[m.risk_level]} {_machine_line(m)}"}})
    blocks.append({"type": "context", "elements": [{
        "type": "mrkdwn",
        "text": f"Alerting on {alert.min_level} risk and above. Open FactoryMind for the "
                "cascade and every option."}]})
    return {"text": subject(alert), "blocks": blocks}


def format_email(alert: Alert, sender: str, recipients: list[str]) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject(alert)
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    lines = [f"{LEVEL_EMOJI[m.risk_level]} {_machine_line(m)}" for m in alert.machines]
    msg.set_content(
        "FactoryMind found machines at risk of failing in the next 24 hours.\n\n"
        + "\n\n".join(lines)
        + f"\n\nAlerting on {alert.min_level} risk and above. Open FactoryMind for the cascade "
          "and every option.\n")
    return msg


Post = Callable[[str, bytes], None]
SmtpFactory = Callable[[str, int], smtplib.SMTP]


def _urllib_post(url: str, body: bytes) -> None:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - URL is operator config
        if resp.status >= 300:
            raise RuntimeError(f"Slack webhook returned HTTP {resp.status}")


def send_slack(alert: Alert, webhook_url: str, post: Post = _urllib_post) -> None:
    post(webhook_url, json.dumps(format_slack(alert)).encode())


def _smtp_connect(host: str, port: int) -> smtplib.SMTP:
    if port == 465:  # implicit TLS
        return smtplib.SMTP_SSL(host, port, timeout=10)
    return smtplib.SMTP(host, port, timeout=10)


def send_email(alert: Alert, config: AlertConfig,
               smtp_factory: SmtpFactory = _smtp_connect) -> None:
    msg = format_email(alert, config.email_from, config.email_to)
    with smtp_factory(config.smtp_host, config.smtp_port) as smtp:
        if config.smtp_port not in (25, 465):
            smtp.starttls()
        if config.smtp_user and config.smtp_password:
            smtp.login(config.smtp_user, config.smtp_password)
        smtp.send_message(msg)


def send_alert(alert: Alert, config: AlertConfig, post: Post = _urllib_post,
               smtp_factory: SmtpFactory | None = None) -> dict[str, str | None]:
    """Send to every configured channel. Returns channel -> None on success or the error text,
    so one failing channel never stops the other."""
    results: dict[str, str | None] = {}
    if config.slack_enabled:
        try:
            send_slack(alert, config.slack_webhook_url, post)
            results["Slack"] = None
        except Exception as exc:  # noqa: BLE001 - reported per channel
            results["Slack"] = str(exc)
    if config.email_enabled:
        try:
            send_email(alert, config, smtp_factory or _smtp_connect)
            results["email"] = None
        except Exception as exc:  # noqa: BLE001 - reported per channel
            results["email"] = str(exc)
    return results
