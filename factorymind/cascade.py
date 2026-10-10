"""The cascade: how a machine's possible failure spreads to quality, inventory, orders and money.

For each machine, two scenarios are simulated hour by hour over CASCADE_HORIZON_H, starting at
the end of the uploaded data:

- **If it fails:** it fails at FAILURE_AT_H (the middle of the 24h prediction window) and is down
  for `repair_hours`. A repair restores normal quality (BASE_RATE defects).
- **If it doesn't fail:** it keeps running, with its projected defect rate.

In both, every machine making the same product adds its *good* output each hour
(units_per_hour × (1 − defect rate)) to the product's inventory. Orders for the product ship
first in, first out by due date as soon as there is enough stock (safety stock may be used). The
safety stock is breached when the inventory position (stock minus orders due but not yet shipped)
falls below it. An
order not shipped by its due date is late; its penalty is penalty_per_day × days late (part days
count as a day) × penalty_multiplier.

    loss if it fails      = downtime cost + repair cost + penalties + excess scrap
    loss if it doesn't    = penalties + excess scrap
    expected loss         = p × loss if it fails + (1 − p) × loss if it doesn't

where p is the probability that any component fails in the next 24h, and excess scrap is this
machine's scrap above the normal BASE_RATE (normal scrap isn't a loss caused by anything).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from factorymind.quality import BASE_RATE
from factorymind.report import Cascade, LateOrder, QualityProjection, ScenarioOutcome

CASCADE_HORIZON_H = 168  # simulate a week ahead
FAILURE_AT_H = 12  # middle of the 24h prediction window


def simulate_orders(orders: pd.DataFrame, on_hand: float, safety_stock: float,
                    production: np.ndarray, penalty_multiplier: float = 1.0,
                    ) -> tuple[list[LateOrder], float | None]:
    """Ship orders from stock, hour by hour.

    `orders` has order_id, customer, product, qty, due_hours, penalty_per_day, priority.
    `production[t]` is the good output finished during hour t, in stock from hour t + 1.
    Returns the late orders and the first hour the inventory position (stock minus orders due
    but not yet shipped) is below safety stock, or None.
    """
    queue = orders.assign(_rank=orders["priority"].map({"high": 0, "normal": 1, "low": 2})
                          .fillna(1)).sort_values(["due_hours", "_rank", "order_id"])
    pending = list(queue.itertuples())
    stock, breach, late = float(on_hand), None, []
    horizon = len(production)
    for t in range(horizon + 1):
        while pending and pending[0].due_hours <= t and stock >= pending[0].qty:
            order = pending.pop(0)
            stock -= order.qty
            if t > order.due_hours:
                late.append(_late(order, t, True, penalty_multiplier))
        backlog = sum(o.qty for o in pending if o.due_hours <= t)
        if breach is None and stock - backlog < safety_stock:
            breach = float(t)
        if t < horizon:
            stock += production[t]
    late += [_late(o, horizon, False, penalty_multiplier) for o in pending if o.due_hours < horizon]
    return late, breach


def _late(order, shipped_at: float, shipped: bool, penalty_multiplier: float) -> LateOrder:
    hours_late = max(0.0, shipped_at - order.due_hours)
    return LateOrder(
        order_id=str(order.order_id), customer=str(order.customer), product=str(order.product),
        due_hours=float(order.due_hours), hours_late=float(hours_late),
        penalty=round(order.penalty_per_day * math.ceil(hours_late / 24) * penalty_multiplier, 2),
        shipped=shipped,
    )


def build_cascades(tables: dict[str, pd.DataFrame], as_of: pd.Timestamp,
                   failure_probability: dict[int, float],
                   quality: dict[int, QualityProjection],
                   assumptions: dict[str, float]) -> dict[int, Cascade]:
    """A cascade for every machine. `quality` must cover at least CASCADE_HORIZON_H hours."""
    schedule = tables["production_schedule"]
    inventory = tables["inventory"].groupby("product")[["on_hand", "safety_stock"]].sum()
    orders = tables["orders"].assign(
        due_hours=(tables["orders"]["due_date"] - as_of) / pd.Timedelta(hours=1))

    h = CASCADE_HORIZON_H
    good_output = {}  # machine -> good units per hour, if it keeps running
    for row in schedule.itertuples():
        defect = np.asarray(quality[row.machine_id].defect_rate[:h]) if row.machine_id in quality \
            else np.full(h, BASE_RATE)
        good_output.setdefault(row.product, {})[row.machine_id] = row.units_per_hour * (1 - defect)

    repair_h = assumptions["repair_hours"]
    down = slice(FAILURE_AT_H, min(h, FAILURE_AT_H + math.ceil(repair_h)))
    scrap_cost = assumptions["scrap_cost_per_unit"]
    penalty_multiplier = assumptions["penalty_multiplier"]

    out = {}
    for machine_id, p in failure_probability.items():
        rows = schedule[schedule["machine_id"] == machine_id]
        product = (str(rows.sort_values("units_per_hour").iloc[-1]["product"])
                   if len(rows) else None)
        units = float(rows["units_per_hour"].sum())
        excess = (np.asarray(quality[machine_id].defect_rate[:h]) - BASE_RATE).clip(min=0) \
            * units * scrap_cost

        def outcome(fails: bool) -> ScenarioOutcome:
            late, breach = [], None
            if product is not None:
                production = sum(good_output[product].values())
                if fails:
                    mine = good_output[product][machine_id].copy()
                    mine[down] = 0
                    mine[down.stop:] = units * (1 - BASE_RATE)
                    production = production - good_output[product][machine_id] + mine
                stock = inventory.loc[product] if product in inventory.index else None
                late, breach = simulate_orders(
                    orders[orders["product"] == product],
                    stock["on_hand"] if stock is not None else 0.0,
                    stock["safety_stock"] if stock is not None else 0.0,
                    production, penalty_multiplier)
            penalties = round(sum(o.penalty for o in late), 2)
            scrap = float(excess[:down.start].sum() if fails else excess.sum())
            downtime_cost = assumptions["downtime_cost_per_hour"] * repair_h if fails else 0.0
            repair_cost = (assumptions["component_replacement_cost"]
                           + assumptions["maintenance_labour_cost_per_hour"] * repair_h
                           if fails else 0.0)
            return ScenarioOutcome(
                late_orders=late, penalties=penalties, excess_scrap_cost=round(scrap, 2),
                downtime_cost=round(downtime_cost, 2), repair_cost=round(repair_cost, 2),
                inventory_breach_hours=breach,
                total_loss=round(penalties + scrap + downtime_cost + repair_cost, 2),
            )

        if_failure, if_no_failure = outcome(True), outcome(False)
        out[machine_id] = Cascade(
            product=product, failure_probability=round(p, 4), failure_at_hours=FAILURE_AT_H,
            downtime_hours=repair_h, lost_units=round(units * repair_h, 1),
            if_failure=if_failure, if_no_failure=if_no_failure,
            expected_loss=round(p * if_failure.total_loss + (1 - p) * if_no_failure.total_loss, 2),
        )
    return out
