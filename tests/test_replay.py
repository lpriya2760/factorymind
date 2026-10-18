"""Replay: stepping through a bundle hour by hour gives the same risk as analysing it then."""

from pathlib import Path

import pandas as pd
import pytest

from factorymind.bundle import load_bundle
from factorymind.engine import analyze
from factorymind.replay import files_until, risk_timeline

DEMOS = Path(__file__).parent.parent / "demo_bundles"
COMP2_CROSSES = pd.Timestamp("2026-10-18 06:00")  # the real failure is 24h later


def bundle(name):
    path = DEMOS / f"{name}.zip"
    return {path.name: path.read_bytes()}


@pytest.fixture(scope="module")
def comp2():
    return risk_timeline(bundle("comp2_failure_building"), hours=48)


def test_replays_the_last_hours_one_hour_at_a_time(comp2):
    assert len(comp2) == 48
    assert comp2.hours[-1] == pd.Timestamp("2026-10-18 18:00")
    assert all(b - a == pd.Timedelta(hours=1) for a, b in zip(comp2.hours, comp2.hours[1:]))
    assert len(comp2.machines_at(0)) == 10


def test_comp2_risk_rises_then_crosses_the_threshold(comp2):
    crossing = comp2.first_crossing(90)
    assert (crossing.at, crossing.component) == (COMP2_CROSSES, "comp2")
    assert crossing.probability > 0.99
    history = comp2.history(90, crossing.step)
    assert history["comp2"].iloc[:-1].max() < 0.01  # low until the hour it crosses
    assert len(history) == crossing.step + 1
    assert comp2.machines_at(crossing.step)[0].machine_id == 90


def test_the_threshold_is_configurable(comp2):
    assert comp2.first_crossing(90, 0.9).at == COMP2_CROSSES
    assert comp2.first_crossing(90, 0.001).at < COMP2_CROSSES  # the early drift
    assert comp2.first_crossing(44) is None
    with pytest.raises(ValueError):
        comp2.first_crossing(90, 0)


@pytest.mark.parametrize("step_back", [0, 1])
def test_each_hour_matches_analysing_the_data_up_to_then(comp2, step_back):
    """No lookahead: the replayed score is what a prediction at that hour would have said."""
    step = comp2.first_crossing(90).step - step_back
    report = analyze(files_until(bundle("comp2_failure_building"), comp2.hours[step]))
    replayed = {m.machine_id: m for m in comp2.machines_at(step)}
    for m in report.machines:
        assert m.component_risk == pytest.approx(replayed[m.machine_id].component_risk)
        assert m.top_drivers == replayed[m.machine_id].top_drivers


def test_files_until_drops_what_had_not_happened_yet():
    files = bundle("comp2_failure_building")
    cut = load_bundle(files_until(files, COMP2_CROSSES)).tables
    full = load_bundle(files).tables
    for name in ("telemetry", "errors", "maint"):
        assert cut[name]["datetime"].max() <= COMP2_CROSSES
    assert cut["telemetry"]["datetime"].max() == COMP2_CROSSES
    pd.testing.assert_frame_equal(cut["orders"], full["orders"])


def test_a_healthy_fleet_never_crosses():
    timeline = risk_timeline(bundle("healthy_fleet"), hours=48)
    assert all(timeline.first_crossing(m) is None for m in timeline.machines.index)
