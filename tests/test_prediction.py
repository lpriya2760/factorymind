"""Failure prediction through analyze(), on the committed demo bundles (real Azure PdM data)."""

import socket
from pathlib import Path

import pytest

from factorymind.engine import analyze

DEMO = Path(__file__).parent.parent / "demo_bundles"


def bundle(name: str) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (DEMO / name).iterdir() if p.is_file()}


@pytest.fixture(scope="module")
def comp2_report():
    return analyze(bundle("comp2_failure_building"))


def test_degrading_comp2_machine_ranks_first(comp2_report):
    top = comp2_report.machines[0]
    assert top.machine_id == 90
    assert top.top_component == "comp2"
    assert top.risk_level == "high"


def test_drivers_explain_the_comp2_risk(comp2_report):
    drivers = comp2_report.machines[0].top_drivers
    assert 1 <= len(drivers) <= 3
    assert all(d.impact >= 0.1 for d in drivers)
    assert not any(d.description.startswith("0×") for d in drivers)
    assert any("rotate" in d.feature or "comp2" in d.feature or "error" in d.feature
               for d in drivers)
    assert drivers == sorted(drivers, key=lambda d: -d.impact)


def test_rest_of_fleet_is_low_risk(comp2_report):
    assert all(m.risk_level == "low" for m in comp2_report.machines[1:])


def test_machines_are_sorted_by_highest_component_risk(comp2_report):
    risks = [m.max_risk for m in comp2_report.machines]
    assert risks == sorted(risks, reverse=True)


def test_probabilities_are_valid(comp2_report):
    for m in comp2_report.machines:
        assert set(m.component_risk) == {"comp1", "comp2", "comp3", "comp4"}
        assert all(0 <= p <= 1 for p in m.component_risk.values())
        assert sum(m.component_risk.values()) <= 1 + 1e-6


def test_two_machines_at_risk_rank_top_two():
    machines = analyze(bundle("two_machines_at_risk")).machines
    top_two = {(m.machine_id, m.top_component) for m in machines[:2]}
    assert top_two == {(92, "comp3"), (95, "comp4")}
    assert all(m.risk_level == "high" for m in machines[:2])


def test_healthy_fleet_has_no_high_risk():
    assert all(m.risk_level != "high" for m in analyze(bundle("healthy_fleet")).machines)


def test_prediction_makes_no_network_calls(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network call during analysis")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert analyze(bundle("healthy_fleet")).machines
