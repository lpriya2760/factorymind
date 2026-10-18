"""Publish real sensor readings to Pub/Sub, hour by hour, as if the machines were live.

Replays the last --replay-hours of a bundle's telemetry and error readings (the real Azure
Predictive Maintenance data) to the factorymind-telemetry topic, one simulated hour every
--seconds-per-hour seconds. Start scripts/monitor.py first (or any time: the subscription
keeps the messages).

    uv run --group stream python scripts/replay_publisher.py --seconds-per-hour 1
"""

import argparse

from stream_common import add_common_args, project_id, read_bundle

from factorymind.stream import TELEMETRY_TOPIC, PubSubBus, replay, split_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--seconds-per-hour", type=float, default=1.0,
                        help="real seconds per simulated hour (default 1)")
    parser.add_argument("--topic", default=TELEMETRY_TOPIC)
    args = parser.parse_args()

    project = project_id(args.project)
    _, events = split_bundle(read_bundle(args.bundle), args.replay_hours)
    bus = PubSubBus(project)
    hours = len({e["datetime"] for e in events})
    print(f"Publishing {len(events)} readings ({hours} hours) to "
          f"projects/{project}/topics/{args.topic}, {args.seconds_per_hour:g}s per hour…")
    last = None

    def publish(data, attributes):
        nonlocal last
        bus.publish(args.topic, data, attributes)
        if attributes["datetime"] != last:
            last = attributes["datetime"]
            print(f"  {last}", flush=True)

    count = replay(events, publish, args.seconds_per_hour)
    print(f"Done: {count} messages published.")


if __name__ == "__main__":
    main()
