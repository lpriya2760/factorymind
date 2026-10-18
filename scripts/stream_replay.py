"""Stream a bundle into FactoryMind's risk monitor hour by hour, as if the readings were live,
and send a Slack alert the moment a component becomes risky.

    uv run python scripts/stream_replay.py demo_bundles/comp2_failure_building.zip
    uv run python scripts/stream_replay.py demo_bundles/comp2_failure_building.zip --dry-run

The bundle's last --hours of telemetry and errors are streamed; everything before seeds the
monitor. Slack needs SLACK_WEBHOOK_URL (in .env locally; see the README). --dry-run prints the
alerts instead of sending them.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from factorymind.alerts import format_slack_event, slack_notifier  # noqa: E402
from factorymind.stream import replay  # noqa: E402


def read_bundle(path: Path) -> dict[str, bytes]:
    if path.is_dir():
        return {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
    if path.suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as zf:
            return {Path(n).name: zf.read(n) for n in zf.namelist() if not n.endswith("/")}
    sys.exit(f"{path} is neither a bundle folder nor a .zip")


def print_hour(interval: float):
    def on_hour(at, monitor, new):
        machine, (component, p, level) = max(monitor.scores.items(), key=lambda kv: kv[1][1])
        print(f"{at:%Y-%m-%d %H:%M}  riskiest: machine {machine} {component} {p:6.1%} ({level})")
        for e in new:
            print(f"   🚨 ALERT machine {e.machine_id} {e.component} "
                  f"{e.failure_probability:.1%}: {e.best_action}"
                  + (f" [delivery failed: {e.delivery_error}]" if e.delivery_error else ""))
        if interval:
            time.sleep(interval)
    return on_hour


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path, help="a demo bundle .zip or folder")
    parser.add_argument("--hours", type=int, default=72, help="final hours to stream (72)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="seconds between streamed hours (0.5)")
    parser.add_argument("--min-level", choices=["high", "medium"], default="high")
    parser.add_argument("--dry-run", action="store_true", help="print alerts, don't send")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if args.dry_run:
        def notify(event):
            print(json.dumps(format_slack_event(event)["blocks"][1], indent=1))
    else:
        url = os.environ.get("SLACK_WEBHOOK_URL")
        if not url:
            sys.exit("Set SLACK_WEBHOOK_URL (see the README's Slack setup), or use --dry-run.")
        notify = slack_notifier(url)
    events = replay(read_bundle(args.bundle), args.hours, notify, args.min_level,
                    on_hour=print_hour(args.interval))
    sent = [e for e in events if not e.delivery_error]
    print(f"\n{len(events)} alert(s) raised, {len(sent)} delivered.")


if __name__ == "__main__":
    main()
