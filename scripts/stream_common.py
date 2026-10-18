"""Shared command-line plumbing for the Pub/Sub scripts (replay_publisher, monitor, stream_demo)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factorymind.alerts import LEVEL_EMOJI  # noqa: E402

DEFAULT_BUNDLE = ROOT / "demo_bundles" / "comp2_failure_building.zip"


def add_common_args(parser: argparse.ArgumentParser, project: bool = True) -> None:
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE,
                        help="a bundle .zip or folder (default: the comp2_failure_building demo)")
    parser.add_argument("--replay-hours", type=int, default=36,
                        help="how many final hours of the data to stream (default 36). The "
                             "publisher and the monitor must use the same value.")
    if project:
        parser.add_argument("--project", default=None,
                            help="GCP project (default: GOOGLE_CLOUD_PROJECT or gcloud config)")


def read_bundle(path: Path) -> dict[str, bytes]:
    if path.is_dir():
        return {p.name: p.read_bytes() for p in sorted(path.iterdir()) if p.is_file()}
    return {path.name: path.read_bytes()}


def project_id(arg: str | None) -> str:
    project = arg or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        try:
            project = subprocess.run(["gcloud", "config", "get-value", "project"],
                                     capture_output=True, text=True).stdout.strip()
        except FileNotFoundError:
            project = ""
    if not project:
        sys.exit("No GCP project: pass --project or run `gcloud config set project <ID>`.")
    return project


def status_line(status) -> str:
    """One line per analysed hour: a dot per machine, coloured by risk."""
    dots = " ".join(f"{LEVEL_EMOJI[level]}{machine}" for machine, level in sorted(status.risk.items()))
    return f"{status.as_of}  {dots}"
