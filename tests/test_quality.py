"""Quality projection through analyze(), on the committed demo bundles."""

import io
import statistics
from pathlib import Path

import pandas as pd
import pytest

from factorymind import quality
from factorymind.engine import analyze

DEMO = Path(__file__).parent.parent / "demo_bundles"
BUNDLES = ["comp2_failure_building", "two_machines_at_risk", "healthy_fleet"]


def bundle(name: str) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (DEMO / name).iterdir() if p.is_file()}


@pytest.fixture(scope="module")
def comp2_report():
    return analyze(bundle("comp2_failure_building"))


def test_every_machine_has_a_projection_over_the_horizon(comp2_report):
    assert set(comp2_report.quality) == {m.machine_id for m in comp2_report.machines}
    for q in comp2_report.quality.values():
        assert q.horizon_hours == list(range(49))
        assert len(q.defect_rate) == 49


def test_custom_horizon():
    q = analyze(bundle("healthy_fleet"), quality_horizon_hours=12).quality
    assert all(p.horizon_hours[-1] == 12 for p in q.values())


def test_invalid_horizon():
    with pytest.raises(ValueError):
        analyze(bundle("healthy_fleet"), quality_horizon_hours=0)


@pytest.mark.parametrize("name", BUNDLES)
def test_defect_rate_never_decreases_and_stays_in_bounds(name):
    for q in analyze(bundle(name)).quality.values():
        rates = q.defect_rate
        assert all(a <= b for a, b in zip(rates, rates[1:]))
        assert quality.BASE_RATE <= rates[0] and rates[-1] <= quality.MAX_RATE


def test_older_components_mean_more_scrap():
    files = bundle("comp2_failure_building")
    maint = pd.read_csv(io.BytesIO(files["maint.csv"]), parse_dates=["datetime"])
    before = analyze(files).quality[90].scrap_cost

    on_90 = maint["machineID"] == 90
    maint.loc[on_90, "datetime"] -= pd.Timedelta(days=120)
    files["maint.csv"] = maint.to_csv(index=False).encode()
    after = analyze(files).quality[90]
    assert after.scrap_cost > before
    assert after.component_age_hours > 0


def test_scrap_cost_scales_with_the_scrap_cost_assumption():
    files = bundle("comp2_failure_building")
    base = analyze(files).quality[90].scrap_cost
    doubled = analyze(files, assumptions={"scrap_cost_per_unit": 30}).quality[90].scrap_cost
    free = analyze(files, assumptions={"scrap_cost_per_unit": 0}).quality[90].scrap_cost
    assert doubled == pytest.approx(2 * base, rel=1e-3)
    assert free == 0


def test_degrading_machine_projects_worse_quality_than_the_fleet(comp2_report):
    q90 = comp2_report.quality[90]
    fleet = statistics.median(q.defect_rate[-1] for q in comp2_report.quality.values())
    assert q90.drift_sensor == "rotate"
    assert q90.defect_rate[-1] > 2 * fleet
    assert q90.errors_24h >= 1


def test_healthy_fleet_stays_low():
    assert all(q.defect_rate[-1] < 0.1 for q in analyze(bundle("healthy_fleet")).quality.values())
