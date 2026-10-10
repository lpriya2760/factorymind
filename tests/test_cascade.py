"""The cascade: hand-checked order simulation, cost assembly, and analyze() on the demo bundles."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factorymind.assumptions import resolve_assumptions
from factorymind.cascade import CASCADE_HORIZON_H, FAILURE_AT_H, build_cascades, simulate_orders
from factorymind.engine import analyze
from factorymind.quality import BASE_RATE
from factorymind.report import QualityProjection

DEMO = Path(__file__).parent.parent / "demo_bundles"


def orders(*rows):
    return pd.DataFrame(rows, columns=["order_id", "customer", "product", "qty", "due_hours",
                                       "penalty_per_day", "priority"])


# Stock 100, safety stock 50, 10 good units an hour. A needs 150 at hour 5; B needs 100 at hour 10.
A = ("A", "Apex", "P", 150, 5.0, 240.0, "normal")
B = ("B", "Bluefin", "P", 100, 10.0, 240.0, "normal")


def test_hand_calculation_without_downtime():
    # A ships on time at hour 5 (100 + 5×10 = 150), leaving 0, so the position drops below 50.
    # B then waits for 100 more units: ships at hour 15, 5h late, one day's penalty.
    late, breach = simulate_orders(orders(A, B), 100, 50, np.full(30, 10.0))
    assert [(o.order_id, o.hours_late, o.penalty, o.shipped) for o in late] == [("B", 5, 240, True)]
    assert breach == 5


def test_hand_calculation_with_downtime():
    # No output in hours 2–5. At hour 5 stock is 120 < 150: A waits (position 120 − 150 < 50).
    # A ships at hour 9 (4h late); B waits for 100 new units and ships at hour 19 (9h late).
    production = np.full(30, 10.0)
    production[2:6] = 0
    late, breach = simulate_orders(orders(A, B), 100, 50, production)
    assert [(o.order_id, o.hours_late, o.penalty) for o in late] == [("A", 4, 240), ("B", 9, 240)]
    assert breach == 5


def test_orders_still_waiting_at_the_end_count_as_late():
    late, _ = simulate_orders(orders(A, B), 100, 50, np.full(12, 10.0))
    assert [(o.order_id, o.hours_late, o.shipped) for o in late] == [("B", 2, False)]


def test_penalty_counts_part_days_and_uses_the_multiplier():
    # 25h late is two (part) days; the multiplier doubles it.
    late, _ = simulate_orders(orders(("C", "Cedar", "P", 100, 0.0, 100.0, "high")), 0, 0,
                              np.full(30, 4.0), penalty_multiplier=2)
    assert (late[0].hours_late, late[0].penalty) == (25, 400)


def test_no_shortfall_means_no_late_orders_or_breach():
    late, breach = simulate_orders(orders(A, B), 1000, 50, np.full(30, 10.0))
    assert late == [] and breach is None


# --- cost assembly (build_cascades) ----------------------------------------------------------

def tiny_tables(qty: int):
    return {
        "production_schedule": pd.DataFrame({"machine_id": [1], "product": ["P"],
                                             "units_per_hour": [10.0]}),
        "inventory": pd.DataFrame({"product": ["P"], "on_hand": [100], "safety_stock": [50]}),
        "orders": pd.DataFrame({
            "order_id": ["A"], "customer": ["Apex"], "product": ["P"], "qty": [qty],
            "due_date": [pd.Timestamp("2015-01-02 00:00")],  # 24h after as_of
            "penalty_per_day": [500.0], "priority": ["high"],
        }),
    }


def healthy_quality():
    hours = list(range(CASCADE_HORIZON_H + 1))
    return {1: QualityProjection(hours, [BASE_RATE] * len(hours), 0.0, 10.0, 0.0, 0.0, "volt",
                                 0.0, 0)}


def cascade(p: float, qty: int = 100):
    return build_cascades(tiny_tables(qty), pd.Timestamp("2015-01-01 00:00"), {1: p},
                          healthy_quality(), resolve_assumptions())[1]


def test_zero_failure_probability_means_zero_loss():
    assert cascade(p=0).expected_loss == 0


def test_certain_failure_costs_downtime_and_repair():
    a = resolve_assumptions()
    c = cascade(p=1)
    assert c.lost_units == 10 * a["repair_hours"]
    assert c.if_failure.downtime_cost == a["downtime_cost_per_hour"] * a["repair_hours"]
    assert c.if_failure.repair_cost == (a["component_replacement_cost"]
                                        + a["maintenance_labour_cost_per_hour"] * a["repair_hours"])
    assert c.expected_loss == c.if_failure.total_loss


def test_failure_makes_a_tight_order_late():
    # 100 in stock + 24h × 9.95 good units ≈ 339 by hour 24 normally; the 8h outage loses ~80.
    c = cascade(p=0.5, qty=300)
    assert c.if_no_failure.late_orders == []
    assert [o.order_id for o in c.if_failure.late_orders] == ["A"]
    assert c.if_failure.penalties == 500
    assert c.expected_loss == pytest.approx(0.5 * c.if_failure.total_loss
                                            + 0.5 * c.if_no_failure.total_loss, abs=0.01)


# --- through analyze() -------------------------------------------------------------------------

def bundle(name):
    return {p.name: p.read_bytes() for p in (DEMO / name).iterdir() if p.is_file()}


@pytest.fixture(scope="module")
def comp2_report():
    return analyze(bundle("comp2_failure_building"))


def test_every_machine_has_a_cascade(comp2_report):
    assert set(comp2_report.cascade) == {m.machine_id for m in comp2_report.machines}


def test_machine_90_failure_cascades_to_late_orders(comp2_report):
    c = comp2_report.cascade[90]
    assert c.failure_probability > 0.99 and c.failure_at_hours == FAILURE_AT_H
    assert c.product == "P-200"
    f = c.if_failure
    assert f.late_orders and f.penalties > 0
    assert f.inventory_breach_hours is not None
    assert f.inventory_breach_hours <= min(o.due_hours + o.hours_late for o in f.late_orders)


def test_machine_90_has_the_largest_expected_loss(comp2_report):
    losses = {m: c.expected_loss for m, c in comp2_report.cascade.items()}
    assert max(losses, key=losses.get) == 90


def test_healthy_fleet_ships_on_time_without_failures():
    for c in analyze(bundle("healthy_fleet")).cascade.values():
        assert c.if_no_failure.late_orders == []
        assert c.if_no_failure.inventory_breach_hours is None


def test_penalty_multiplier_flows_through():
    report = analyze(bundle("comp2_failure_building"), assumptions={"penalty_multiplier": 0})
    assert report.cascade[90].if_failure.penalties == 0
