"""Generate Gemini's recommendations for the demo bundles and ship them with the app.

The demo then never waits on Gemini, even right after Cloud Run scales from zero. Re-run after
changing the prompts, the analysis or the demo data (a test fails until you do):

    uv run python scripts/cache_recommendations.py [--workers 4] [--missing]

--missing keeps the answers already shipped and only asks Gemini for the ones not yet cached.

Uses whatever Gemini configuration is in the environment (AI Studio key or Vertex AI).
Only answers whose every number passed the guardrail are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factorymind.agent import (PROMPT_VERSION, SHIPPED_CACHE, cache_key,  # noqa: E402
                               recommend, recommendation_to_dict)
from factorymind.replay import files_until, risk_timeline  # noqa: E402

DEMO = ROOT / "demo_bundles"
# Assumption changes shown in the demo, besides the defaults.
EXTRA_SCENARIOS = [("comp2_failure_building", 90, {"downtime_cost_per_hour": 300})]
# Replays that cross the default threshold: the plant as it was at the crossing hour.
REPLAY_SCENARIOS = [("comp2_failure_building", 90)]
ATTEMPTS = 3


def jobs():
    for bundle in sorted(p for p in DEMO.iterdir() if p.is_dir()):
        files = {p.name: p.read_bytes() for p in bundle.iterdir() if p.is_file()}
        machines = [int(line.split(",")[0])
                    for line in (bundle / "machines.csv").read_text().splitlines()[1:]]
        for m in machines:
            yield bundle.name, files, m, {}
        for name, m, assumptions in EXTRA_SCENARIOS:
            if name == bundle.name:
                yield bundle.name, files, m, assumptions
        for name, m in REPLAY_SCENARIOS:
            if name == bundle.name:
                crossing = risk_timeline(files).first_crossing(m)
                yield (f"{bundle.name} replayed to {crossing.at:%Y-%m-%d %H:%M}",
                       files_until(files, crossing.at), m, {})


def answer(job):
    name, files, machine_id, assumptions = job
    for attempt in range(1, ATTEMPTS + 1):
        try:
            rec = recommend(files, machine_id, assumptions, use_cache=False)
        except Exception as error:  # noqa: BLE001 - overloaded or rate-limited: wait and retry
            print(f"  {name} machine {machine_id}: attempt {attempt} failed ({error})"[:160])
            time.sleep(20 * attempt)
            continue
        if not rec.unverified_numbers:
            return job, rec
        print(f"  {name} machine {machine_id}: unverified {rec.unverified_numbers}, retrying")
    return job, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--missing", action="store_true",
                        help="keep the shipped answers; only ask for the ones not cached yet")
    args = parser.parse_args()

    entries, meta, failed = {}, {}, []
    todo = list(jobs())
    if args.missing and SHIPPED_CACHE.exists():
        shipped = json.loads(SHIPPED_CACHE.read_text())
        if shipped["prompt_version"] == PROMPT_VERSION:
            entries, meta = shipped["entries"], shipped["meta"]
            todo = [j for j in todo if cache_key(j[1], j[2], j[3]) not in entries]
    with ThreadPoolExecutor(args.workers) as pool:
        for (name, files, machine_id, assumptions), rec in pool.map(answer, todo):
            label = f"{name} machine {machine_id}" + (f" {assumptions}" if assumptions else "")
            if rec is None:
                failed.append(label)
                print(f"FAILED {label}")
                continue
            key = cache_key(files, machine_id, assumptions)
            entries[key] = recommendation_to_dict(rec)
            meta[key] = {"bundle": name, "machine_id": machine_id, "assumptions": assumptions}
            print(f"ok {label}: {rec.model}, {len(rec.tool_calls)} tool calls"
                  + (", after a rewrite" if rec.retried else ""))

    SHIPPED_CACHE.write_text(json.dumps({
        "prompt_version": PROMPT_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entries": entries, "meta": meta,
    }, indent=1) + "\n")
    print(f"wrote {len(entries)} answers to {SHIPPED_CACHE.relative_to(ROOT)}")
    if failed:
        sys.exit(f"no verified answer for: {', '.join(failed)}; re-run to fill them in")


if __name__ == "__main__":
    main()
