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

The expected loss uses one failure time. loss_range() adds the spread: seeded Monte Carlo draws
of whether and when in the 24h window the machine fails, summarised as P10/P50/P90 (ADR 0005).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from factorymind.quality import BASE_RATE
from factorymind.report import Cascade, LateOrder, QualityProjection, ScenarioOutcome

CASCADE_HORIZON_H = 168  # simulate a week ahead
PREDICTION_WINDOW_H = 24  # the failure model predicts a failure within this window
FAILURE_AT_H = 12  # middle of the 24h prediction window
MC_SAMPLES = 1000  # Monte Carlo draws behind the loss range


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


@dataclass
class Plant:
    """Everything the cascade needs about the plant, prepared once per analysis."""
    orders: pd.DataFrame  # with due_hours from the end of the data
    inventory: pd.DataFrame  # on_hand, safety_stock by product
    product: dict[int, str | None]  # each machine's (main) product
    units: dict[int, float]  # each machine's scheduled units per hour
    makers: dict[str, list[int]]  # machines making each product
    defect: dict[int, np.ndarray]  # each machine's projected defect rate, hour by hour
    assumptions: dict[str, float]

    def good_output(self, machine_id: int) -> np.ndarray:
        """Good units per hour if the machine keeps running as it is."""
        return self.units[machine_id] * (1 - self.defect[machine_id])

    def excess_scrap(self, machine_id: int) -> np.ndarray:
        """USD per hour of this machine's scrap above the normal defect rate."""
        return ((self.defect[machine_id] - BASE_RATE).clip(min=0) * self.units[machine_id]
                * self.assumptions["scrap_cost_per_unit"])


def prepare_plant(tables: dict[str, pd.DataFrame], as_of: pd.Timestamp,
                  quality: dict[int, QualityProjection], assumptions: dict[str, float]) -> Plant:
    """`quality` must cover at least CASCADE_HORIZON_H hours."""
    schedule = tables["production_schedule"]
    h = CASCADE_HORIZON_H
    product, units, makers, defect = {}, {}, {}, {}
    for machine_id in quality:
        rows = schedule[schedule["machine_id"] == machine_id]
        product[machine_id] = (str(rows.sort_values("units_per_hour").iloc[-1]["product"])
                               if len(rows) else None)
        units[machine_id] = float(rows["units_per_hour"].sum())
        defect[machine_id] = np.asarray(quality[machine_id].defect_rate[:h])
        if product[machine_id] is not None:
            makers.setdefault(product[machine_id], []).append(machine_id)
    return Plant(
        orders=tables["orders"].assign(
            due_hours=(tables["orders"]["due_date"] - as_of) / pd.Timedelta(hours=1)),
        inventory=tables["inventory"].groupby("product")[["on_hand", "safety_stock"]].sum(),
        product=product, units=units, makers=makers, defect=defect, assumptions=assumptions,
    )


def run_scenarios(plant: Plant, machine_id: int, p: float,
                  production: dict[int, np.ndarray] | None = None,
                  excess_scrap: np.ndarray | None = None,
                  failure_at: int = FAILURE_AT_H,
                  ) -> tuple[ScenarioOutcome, ScenarioOutcome, float]:
    """Simulate `machine_id` failing at `failure_at` (probability p) or not.

    `production` (good units per hour of each machine making the product) and `excess_scrap`
    (USD per hour attributed to this machine) default to everyone running as they are; the
    intervention simulator passes changed ones. Returns (if it fails, if not, expected loss).
    """
    a = plant.assumptions
    product = plant.product[machine_id]
    if production is None:
        production = {m: plant.good_output(m) for m in plant.makers.get(product, [])}
    if excess_scrap is None:
        excess_scrap = plant.excess_scrap(machine_id)
    repair_h = a["repair_hours"]
    down = slice(failure_at, min(CASCADE_HORIZON_H, failure_at + math.ceil(repair_h)))

    def outcome(fails: bool) -> ScenarioOutcome:
        late, breach = [], None
        if product is not None:
            total = sum(production.values())
            if fails:
                mine = production[machine_id].copy()
                mine[down] = 0
                mine[down.stop:] = plant.units[machine_id] * (1 - BASE_RATE)  # repaired
                total = total - production[machine_id] + mine
            stock = plant.inventory.loc[product] if product in plant.inventory.index else None
            late, breach = simulate_orders(
                plant.orders[plant.orders["product"] == product],
                stock["on_hand"] if stock is not None else 0.0,
                stock["safety_stock"] if stock is not None else 0.0,
                total, a["penalty_multiplier"])
        penalties = round(sum(o.penalty for o in late), 2)
        scrap = float(excess_scrap[:down.start].sum() if fails else excess_scrap.sum())
        downtime_cost = a["downtime_cost_per_hour"] * repair_h if fails else 0.0
        repair_cost = (a["component_replacement_cost"]
                       + a["maintenance_labour_cost_per_hour"] * repair_h if fails else 0.0)
        return ScenarioOutcome(
            late_orders=late, penalties=penalties, excess_scrap_cost=round(scrap, 2),
            downtime_cost=round(downtime_cost, 2), repair_cost=round(repair_cost, 2),
            inventory_breach_hours=breach,
            total_loss=round(penalties + scrap + downtime_cost + repair_cost, 2),
        )

    if_failure, if_no_failure = outcome(True), outcome(False)
    expected = round(p * if_failure.total_loss + (1 - p) * if_no_failure.total_loss, 2)
    return if_failure, if_no_failure, expected


def loss_range(plant: Plant, machine_id: int, p: float, seed: int = 0,
               production: dict[int, np.ndarray] | None = None,
               excess_scrap: np.ndarray | None = None,
               samples: int = MC_SAMPLES,
               no_failure_loss: float | None = None) -> tuple[float, float, float]:
    """P10, P50 and P90 of the week's loss, from `samples` seeded Monte Carlo draws.

    Each draw decides whether the machine fails (with probability p) and, if it does, in which
    hour of the 24h prediction window (each hour equally likely); every other input is the same
    as the expected-value cascade. Percentiles use the inverted-CDF method, so each one is a loss
    that actually occurs in a draw rather than an interpolation between "fails" and "keeps
    running".

    A failure's loss depends only on its hour, so each drawn hour is simulated once. When fewer
    than 10% of draws fail, every percentile is the keeps-running loss whichever way the failure
    losses fall, so none are simulated. `no_failure_loss` skips re-simulating that scenario when
    the caller already has it.
    """
    p = min(max(p, 0.0), 1.0)
    rng = np.random.default_rng([seed, machine_id])
    fails = rng.random(samples) < p
    when = rng.integers(0, PREDICTION_WINDOW_H, samples)
    if no_failure_loss is None:
        no_failure_loss = run_scenarios(plant, machine_id, p, production, excess_scrap)[1].total_loss
    if fails.sum() * 10 < samples:
        return (round(float(no_failure_loss), 2),) * 3
    fail_loss = np.zeros(PREDICTION_WINDOW_H)
    for h in np.unique(when[fails]):
        fail_loss[h] = run_scenarios(plant, machine_id, p, production, excess_scrap,
                                     failure_at=int(h))[0].total_loss
    losses = np.where(fails, fail_loss[when], no_failure_loss)
    p10, p50, p90 = np.percentile(losses, [10, 50, 90], method="inverted_cdf")
    return round(float(p10), 2), round(float(p50), 2), round(float(p90), 2)


def build_cascades(plant: Plant, failure_probability: dict[int, float],
                   seed: int = 0) -> dict[int, Cascade]:
    """The do-nothing cascade for every machine, with its Monte Carlo loss range."""
    out = {}
    for machine_id, p in failure_probability.items():
        if_failure, if_no_failure, expected = run_scenarios(plant, machine_id, p)
        p10, p50, p90 = loss_range(plant, machine_id, p, seed,
                                   no_failure_loss=if_no_failure.total_loss)
        repair_h = plant.assumptions["repair_hours"]
        out[machine_id] = Cascade(
            product=plant.product[machine_id], failure_probability=round(p, 4),
            failure_at_hours=FAILURE_AT_H, downtime_hours=repair_h,
            lost_units=round(plant.units[machine_id] * repair_h, 1),
            if_failure=if_failure, if_no_failure=if_no_failure, expected_loss=expected,
            loss_p10=p10, loss_p50=p50, loss_p90=p90, loss_samples=MC_SAMPLES,
        )
    return out
