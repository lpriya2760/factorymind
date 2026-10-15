"""Editable plant assumptions flow through the whole analysis."""

from pathlib import Path

import pytest

from factorymind.agent import AnalysisSession, make_tools
from factorymind.assumptions import ASSUMPTION_INFO, DEFAULT_ASSUMPTIONS
from factorymind.engine import analyze

DEMO = Path(__file__).parent.parent / "demo_bundles"


@pytest.fixture(scope="module")
def files():
    return {p.name: p.read_bytes() for p in (DEMO / "comp2_failure_building").iterdir()
            if p.is_file()}


def test_every_assumption_can_be_shown_and_explained():
    assert set(ASSUMPTION_INFO) == set(DEFAULT_ASSUMPTIONS)
    assert all(info.label and info.unit and info.help for info in ASSUMPTION_INFO.values())


def test_doubling_the_downtime_cost_increases_a_failing_machines_expected_loss(files):
    base = DEFAULT_ASSUMPTIONS["downtime_cost_per_hour"]
    normal = analyze(files)
    doubled = analyze(files, assumptions={"downtime_cost_per_hour": 2 * base})
    assert doubled.assumptions_used["downtime_cost_per_hour"] == 2 * base
    assert doubled.cascade[90].expected_loss > normal.cascade[90].expected_loss
    assert doubled.cascade[90].if_failure.downtime_cost == 2 * normal.cascade[90].if_failure.downtime_cost


def test_edits_reach_the_actions_and_their_ranking(files):
    normal = analyze(files).interventions[90]
    pricey_parts = analyze(files, assumptions={"component_replacement_cost": 5000}).interventions[90]
    replace = {o.lever: o for o in pricey_parts}["replace_component"]
    assert replace.cost > {o.lever: o for o in normal}["replace_component"].cost


def test_gemini_sees_the_edited_assumptions(files):
    tools = {t.__name__: t for t in make_tools(
        AnalysisSession(files, {"downtime_cost_per_hour": 9000}))}
    assert tools["get_assumptions"]()["downtime_cost_per_hour"] == 9000
