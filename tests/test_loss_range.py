"""The Monte Carlo loss range (P10/P50/P90) on the cascade (issue #11)."""

from pathlib import Path

import pandas as pd
import pytest

from factorymind.assumptions import resolve_assumptions
from factorymind.cascade import (MC_SAMPLES, PREDICTION_WINDOW_H, build_cascades, loss_range,
                                 prepare_plant, run_scenarios)
from factorymind.engine import analyze
from tests.test_cascade import healthy_quality, tiny_tables

DEMOS = Path(__file__).parent.parent / "demo_bundles"


def demo(name, seed=0):
    path = DEMOS / f"{name}.zip"
    return analyze({path.name: path.read_bytes()}, seed=seed)


@pytest.fixture(scope="module")
def comp2():
    return demo("comp2_failure_building")


def tiny_plant(qty=300):
    return prepare_plant(tiny_tables(qty), pd.Timestamp("2015-01-01 00:00"), healthy_quality(),
                         resolve_assumptions())


# --- the range itself -----------------------------------------------------------------------

def test_percentiles_are_ordered_for_every_demo_machine():
    for name in ("comp2_failure_building", "two_machines_at_risk", "healthy_fleet"):
        for c in demo(name).cascade.values():
            assert c.loss_p10 <= c.loss_p50 <= c.loss_p90
            assert c.loss_samples == MC_SAMPLES


def test_same_seed_gives_an_identical_report(comp2):
    assert demo("comp2_failure_building").to_dict() == comp2.to_dict()


def test_seed_changes_the_draws():
    plant = tiny_plant()
    ranges = {loss_range(plant, 1, 0.5, seed=s) for s in range(5)}
    assert len(ranges) > 1


def test_no_failure_chance_means_no_spread():
    c = build_cascades(tiny_plant(), {1: 0.0})[1]
    assert c.loss_p10 == c.loss_p50 == c.loss_p90 == c.if_no_failure.total_loss


def test_certain_failure_spreads_over_the_failure_hour():
    plant = tiny_plant()
    outcomes = sorted(run_scenarios(plant, 1, 1.0, failure_at=h)[0].total_loss
                      for h in range(PREDICTION_WINDOW_H))
    p10, p50, p90 = loss_range(plant, 1, 1.0)
    assert outcomes[0] <= p10 <= p50 <= p90 <= outcomes[-1]
    # Every percentile is a loss that actually happens, not an interpolation.
    assert {p10, p50, p90} <= set(outcomes)


def test_even_odds_split_between_failing_and_running():
    plant = tiny_plant()
    fails = run_scenarios(plant, 1, 0.5, failure_at=12)[0].total_loss
    ok = run_scenarios(plant, 1, 0.5)[1].total_loss
    p10, _, p90 = loss_range(plant, 1, 0.5)
    assert p10 == ok  # the cheaper outcome: keeps running
    assert p90 >= fails * 0.9  # a failure somewhere in the window


def test_rare_failure_leaves_every_percentile_at_the_running_loss(comp2):
    c = comp2.cascade[44]  # ~1% failure chance
    assert c.failure_probability < 0.1
    assert c.loss_p10 == c.loss_p50 == c.loss_p90 == c.if_no_failure.total_loss
    assert c.expected_loss > c.loss_p90  # the tail risk sits in the expected value


# --- the demo story -------------------------------------------------------------------------

def test_machine_90_range_brackets_its_expected_loss(comp2):
    c = comp2.cascade[90]
    assert c.loss_p10 < c.expected_loss < c.loss_p90
    assert c.loss_p10 < c.loss_p90  # failure timing matters
