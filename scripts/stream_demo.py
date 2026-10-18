"""The whole Pub/Sub pipeline in one process, in memory: no Google Cloud account needed.

Same replay, same monitor, same alert messages as replay_publisher.py + monitor.py, but the
topics are in-memory. Use it to rehearse the demo or when there's no network.

    uv run python scripts/stream_demo.py --seconds-per-hour 0.5
"""

import argparse
import json

from stream_common import add_common_args, read_bundle, status_line

from factorymind.pubsub import (ALERTS_TOPIC, TELEMETRY_TOPIC, LiveMonitor, LocalBus,
                                alert_message, decode, encode, replay, split_bundle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, project=False)
    parser.add_argument("--seconds-per-hour", type=float, default=0.0)
    parser.add_argument("--min-level", choices=["high", "medium"], default="high")
    args = parser.parse_args()

    history, events = split_bundle(read_bundle(args.bundle), args.replay_hours)
    bus = LocalBus()

    def on_status(status, report):
        print(status_line(status), flush=True)
        if status.alert:
            message = alert_message(status.alert, report, status.as_of)
            bus.publish(ALERTS_TOPIC, encode(message), {"as_of": status.as_of})
            print(f"  🔔 ALERT → {ALERTS_TOPIC}: {message['subject']}")
            print("    " + json.dumps(message["machines"][0]))

    monitor = LiveMonitor(history, min_level=args.min_level, on_status=on_status)
    bus.subscribe(TELEMETRY_TOPIC, lambda data, attrs: monitor.ingest(decode(data)))
    print(f"Replaying {len(events)} real readings through {TELEMETRY_TOPIC} (in memory)…")
    replay(events, lambda data, attrs: bus.publish(TELEMETRY_TOPIC, data, attrs),
           args.seconds_per_hour)
    print(f"Done: {len(bus.messages[ALERTS_TOPIC])} alert(s) published.")


if __name__ == "__main__":
    main()
