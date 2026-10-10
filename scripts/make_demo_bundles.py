"""Cut demo bundles from the Azure Predictive Maintenance dataset.

Each bundle has 10 machines × 30 days of real telemetry, errors and maintenance, plus synthetic
business data (schedule, inventory, orders) from factorymind.synthetic. A manifest.json records
provenance and the scenario, including which failures happen in the 24h after the bundle ends
(ground truth for the demo; the app ignores it).

Usage (after downloading the Kaggle dataset into data/raw/):
    uv run python scripts/make_demo_bundles.py [--raw data/raw] [--out demo_bundles] [--seed 7]
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factorymind.bundle import MAX_BUNDLE_BYTES  # noqa: E402
from factorymind.engine import analyze  # noqa: E402
from factorymind.synthetic import generate_business_data, to_csv_bytes  # noqa: E402

N_MACHINES = 10
WINDOW = pd.Timedelta(days=30)
HORIZON = pd.Timedelta(hours=24)  # the model predicts failures this far ahead
LEAD = pd.Timedelta(hours=12)  # how long after the bundle ends the comp2 failure happens
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class Scenario:
    name: str
    description: str
    end: pd.Timestamp
    at_risk: list[int]  # machines with a failure shortly after `end`
    # Other machines must have no failure this long after `end`.
    quiet_after: pd.Timedelta = HORIZON


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("demo_bundles"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    raw = load_raw(args.raw)
    rng = np.random.default_rng(args.seed)
    scenarios = [
        comp2_failure_building(raw, rng),
        two_machines_at_risk(raw, rng),
        healthy_fleet(raw),
    ]
    args.out.mkdir(exist_ok=True)
    for scenario in scenarios:
        machines = choose_machines(raw, scenario, rng)
        files = build_bundle(raw, scenario, machines, args.seed)
        write_bundle(args.out, scenario.name, files)
        report = analyze(files)  # raises if the bundle is invalid
        size = sum(len(b) for b in files.values())
        assert size < MAX_BUNDLE_BYTES, f"{scenario.name} is {size} bytes"
        print(f"{scenario.name}: {len(report.machines)} machines, ends {scenario.end}, "
              f"{size / 2**20:.1f} MB")


def load_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ["telemetry", "errors", "maint", "failures", "machines"]:
        path = raw_dir / f"PdM_{name}.csv"
        if not path.exists():
            sys.exit(f"Missing {path}. Download the Kaggle dataset "
                     f"arnabbiswas1/microsoft-azure-predictive-maintenance into {raw_dir}/.")
        df = pd.read_csv(path)
        if "datetime" in df:
            df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
        tables[name] = df
    return tables


def failures_between(raw, start, end, machine_ids=None) -> pd.DataFrame:
    f = raw["failures"]
    mask = (f["datetime"] > start) & (f["datetime"] <= end)
    if machine_ids is not None:
        mask &= f["machineID"].isin(machine_ids)
    return f[mask]


def is_clean(raw, machine_id, start, end) -> bool:
    """No failure on this machine between start and end."""
    return failures_between(raw, start, end, [machine_id]).empty


def first_telemetry(raw) -> pd.Timestamp:
    return raw["telemetry"]["datetime"].min()


def comp2_failure_building(raw, rng) -> Scenario:
    earliest = first_telemetry(raw) + WINDOW + LEAD
    candidates = [
        f for f in raw["failures"].itertuples()
        if f.failure == "comp2" and f.datetime >= earliest
        and is_clean(raw, f.machineID, f.datetime - LEAD - WINDOW, f.datetime - pd.Timedelta(seconds=1))
    ]
    if not candidates:
        sys.exit("No comp2 failure with a clean 30-day history found.")
    f = candidates[int(rng.integers(len(candidates)))]
    end = f.datetime - LEAD
    return Scenario(
        "comp2_failure_building",
        f"Machine {f.machineID}'s comp2 fails {LEAD / pd.Timedelta(hours=1):.0f}h after the "
        "data ends. Its risk should stand out; the rest of the fleet is healthy.",
        end, [int(f.machineID)],
    )


def two_machines_at_risk(raw, rng) -> Scenario:
    earliest = first_telemetry(raw) + WINDOW + pd.Timedelta(hours=6)
    f = raw["failures"][raw["failures"]["datetime"] >= earliest].sort_values("datetime")
    pairs = []
    rows = list(f.itertuples())
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if b.datetime - a.datetime > pd.Timedelta(hours=14):
                break
            if a.machineID == b.machineID or a.failure == b.failure:
                continue
            end = a.datetime - pd.Timedelta(hours=6)
            start = end - WINDOW
            if all(is_clean(raw, m, start, end) for m in (a.machineID, b.machineID)):
                pairs.append((a, b, end))
    if not pairs:
        sys.exit("No pair of failures on different machines close together found.")
    a, b, end = pairs[int(rng.integers(len(pairs)))]
    return Scenario(
        "two_machines_at_risk",
        f"Machine {a.machineID} ({a.failure}) and machine {b.machineID} ({b.failure}) both fail "
        "within 24h of the data ending. Shows prioritising between two risks.",
        end, [int(a.machineID), int(b.machineID)],
    )


def healthy_fleet(raw) -> Scenario:
    # Mid-year, well inside the data; choose_machines picks machines with no nearby failures.
    end = pd.Timestamp("2015-06-30 18:00:00")
    return Scenario(
        "healthy_fleet",
        "No machine fails during the data or the 14 days after it. A calm baseline to compare "
        "with.",
        end, [], quiet_after=pd.Timedelta(days=14),
    )


def choose_machines(raw, scenario: Scenario, rng) -> list[int]:
    """The at-risk machines, one healthy machine of the same model for each (so rerouting is
    possible), then random healthy machines up to N_MACHINES."""
    start = scenario.end - WINDOW
    models = raw["machines"].set_index("machineID")["model"]
    failing = set(failures_between(raw, start, scenario.end + scenario.quiet_after)["machineID"])
    healthy = sorted(m for m in models.index if m not in failing)

    chosen = list(scenario.at_risk)
    for m in scenario.at_risk:
        same_model = [h for h in healthy if models[h] == models[m] and h not in chosen]
        if same_model:
            chosen.append(int(rng.choice(same_model)))
    rest = [h for h in healthy if h not in chosen]
    chosen += [int(h) for h in rng.choice(rest, size=N_MACHINES - len(chosen), replace=False)]
    return sorted(chosen)


def build_bundle(raw, scenario: Scenario, machines: list[int], seed: int) -> dict[str, bytes]:
    start, end = scenario.end - WINDOW, scenario.end

    def rows(name, since=None):
        df = raw[name][raw[name]["machineID"].isin(machines)]
        if "datetime" in df:
            mask = df["datetime"] <= end
            if since is not None:
                mask &= df["datetime"] > since
            df = df[mask]
        return df.sort_values([c for c in ("datetime", "machineID") if c in df])

    machine_tables = {
        "telemetry": rows("telemetry", since=start),
        "errors": rows("errors", since=start),
        "maint": rows("maint"),  # full history, for "hours since component replaced"
        "machines": rows("machines"),
    }
    files = to_csv_bytes(machine_tables)
    files |= to_csv_bytes(generate_business_data(machine_tables["machines"], end, seed))

    upcoming = failures_between(raw, end, end + HORIZON, machines)
    files["manifest.json"] = (json.dumps({
        "provenance": {**{t: "real" for t in machine_tables},
                       **{t: "synthetic" for t in ("production_schedule", "inventory", "orders")}},
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "window_start": start.strftime(DATE_FORMAT),
            "as_of": end.strftime(DATE_FORMAT),
            "upcoming_failures": [
                {"machineID": int(f.machineID), "comp": f.failure,
                 "datetime": f.datetime.strftime(DATE_FORMAT)}
                for f in upcoming.itertuples()
            ],
        },
        "generated_by": f"scripts/make_demo_bundles.py --seed {seed}",
    }, indent=2) + "\n").encode()
    return files


def write_bundle(out: Path, name: str, files: dict[str, bytes]) -> None:
    folder = out / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in sorted(files):
            (folder / filename).write_bytes(files[filename])
            # A fixed timestamp keeps the zip byte-identical between runs.
            zf.writestr(zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0)),
                        files[filename], zipfile.ZIP_DEFLATED)
    (out / f"{name}.zip").write_bytes(buf.getvalue())


if __name__ == "__main__":
    main()
