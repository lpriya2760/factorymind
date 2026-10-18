"""Watch the telemetry stream on Pub/Sub and publish an alert when a machine turns high-risk.

Pulls readings from the factorymind-monitor subscription, re-runs FactoryMind's analysis each
time every machine has reported for the next hour, and publishes a JSON alert to the
factorymind-alerts topic the first time a machine reaches the alert level. With --notify it
also sends the alert to the Slack/email channels configured for the app (see the README).

    uv run --group stream python scripts/monitor.py            # then, in another terminal:
    uv run --group stream python scripts/replay_publisher.py
"""

import argparse
import json

from stream_common import add_common_args, project_id, read_bundle, status_line

from factorymind.alerts import AlertConfig, send_alert
from factorymind.stream import (ALERTS_TOPIC, MONITOR_SUBSCRIPTION, LiveMonitor, PubSubBus,
                                alert_message, decode, encode, split_bundle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--subscription", default=MONITOR_SUBSCRIPTION)
    parser.add_argument("--alerts-topic", default=ALERTS_TOPIC)
    parser.add_argument("--min-level", choices=["high", "medium"], default="high")
    parser.add_argument("--notify", action="store_true",
                        help="also send alerts to the configured Slack/email channels")
    parser.add_argument("--timeout", type=float, default=None,
                        help="stop after this many seconds (default: run until Ctrl-C)")
    args = parser.parse_args()

    project = project_id(args.project)
    history, _ = split_bundle(read_bundle(args.bundle), args.replay_hours)
    bus = PubSubBus(project)
    config = AlertConfig.from_env() if args.notify else None

    def on_status(status, report):
        print(status_line(status), flush=True)
        if status.alert is None:
            return
        message = alert_message(status.alert, report, status.as_of)
        message_id = bus.publish(args.alerts_topic, encode(message),
                                 {"min_level": status.alert.min_level, "as_of": status.as_of})
        print(f"  🔔 ALERT published to {args.alerts_topic} (id {message_id}): "
              f"{message['subject']}")
        print("    " + json.dumps(message["machines"][0]))
        if config and config.channels:
            for channel, error in send_alert(status.alert, config).items():
                print(f"    {channel}: {'sent' if error is None else 'failed: ' + error}")

    monitor = LiveMonitor(history, min_level=args.min_level, on_status=on_status)
    print(f"Monitoring projects/{project}/subscriptions/{args.subscription} "
          f"({len(monitor.machines)} machines, alerting on {args.min_level} risk). "
          "Ctrl-C to stop.", flush=True)
    try:
        bus.subscribe(args.subscription, lambda data, attrs: monitor.ingest(decode(data)),
                      timeout=args.timeout)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
