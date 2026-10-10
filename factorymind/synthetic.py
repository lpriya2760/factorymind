"""Seeded generator for the synthetic business context of a bundle.

Public machine datasets have no production schedule, inventory or customer orders, so we
generate them consistently with a bundle's machines. Machines of the same model make the same
product, so work can be rerouted between them. Orders are sized so the plant runs close to
capacity: losing a machine for a shift makes some orders late, which is what the cascade shows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Typical units per hour by machine model; unknown models get the default.
_BASE_RATE = {"model1": 30.0, "model2": 35.0, "model3": 40.0, "model4": 45.0}
_DEFAULT_RATE = 35.0

# Fictional customers.
CUSTOMERS = [
    "Apex Auto", "Bluefin Marine", "Cedar Robotics", "Delta Pumps", "Everline Packaging",
    "Fjord Appliances", "Granite Tools", "Harbor Rail",
]
PRIORITIES = ["high", "normal", "low"]


def product_for(model: str) -> str:
    """model3 -> P-300; any other model name -> P-<name>."""
    suffix = model.removeprefix("model")
    return f"P-{suffix}00" if suffix.isdigit() else f"P-{model}"


def generate_business_data(machines: pd.DataFrame, as_of: pd.Timestamp,
                           seed: int) -> dict[str, pd.DataFrame]:
    """Production schedule, inventory and orders for `machines` (columns machineID, model).

    Orders fall due 1–5 days after `as_of`, the end of the bundle's telemetry. The same inputs
    and seed always give the same tables.
    """
    rng = np.random.default_rng(seed)
    machines = machines.sort_values("machineID")

    schedule = pd.DataFrame({
        "machine_id": machines["machineID"].astype("int64").to_numpy(),
        "product": [product_for(m) for m in machines["model"]],
        "units_per_hour": [
            round(_BASE_RATE.get(m, _DEFAULT_RATE) * rng.uniform(0.9, 1.1), 1)
            for m in machines["model"]
        ],
    })

    inventory_rows, order_rows = [], []
    as_of = pd.Timestamp(as_of)
    for product, group in schedule.groupby("product", sort=True):
        capacity = group["units_per_hour"].sum()  # units per hour across the product's machines
        on_hand = int(capacity * rng.uniform(12, 30))
        safety_stock = int(on_hand * rng.uniform(0.25, 0.4))
        inventory_rows.append((product, on_hand, safety_stock))

        # Size orders so cumulative demand by each due date nearly uses up the stock above
        # safety level plus everything the machines can make before then.
        due_days = np.sort(rng.integers(1, 6, size=int(rng.integers(2, 5))))
        promised = 0
        for days in due_days:
            due = as_of.normalize() + pd.Timedelta(days=int(days))
            hours = (due - as_of) / pd.Timedelta(hours=1)
            target = (on_hand - safety_stock) + capacity * hours * rng.uniform(0.85, 0.98)
            qty = max(10, int(round((target - promised) / 10)) * 10)
            promised += qty
            order_rows.append((
                product, qty, due,
                float(round(rng.uniform(200, 2000), -1)),
                str(rng.choice(PRIORITIES, p=[0.3, 0.5, 0.2])),
                str(rng.choice(CUSTOMERS)),
            ))

    inventory = pd.DataFrame(inventory_rows, columns=["product", "on_hand", "safety_stock"])
    orders = pd.DataFrame(
        order_rows, columns=["product", "qty", "due_date", "penalty_per_day", "priority", "customer"]
    ).sort_values(["due_date", "product"], kind="stable").reset_index(drop=True)
    orders.insert(0, "order_id", [f"O-{i + 1:04d}" for i in range(len(orders))])
    orders = orders[["order_id", "customer", "product", "qty", "due_date", "penalty_per_day",
                     "priority"]]

    return {"production_schedule": schedule, "inventory": inventory, "orders": orders}


def to_csv_bytes(tables: dict[str, pd.DataFrame]) -> dict[str, bytes]:
    """Serialise tables to CSV with fixed formatting, so the same tables give the same bytes."""
    return {
        f"{name}.csv": df.to_csv(index=False, lineterminator="\n",
                                 date_format="%Y-%m-%d %H:%M:%S").encode()
        for name, df in tables.items()
    }
