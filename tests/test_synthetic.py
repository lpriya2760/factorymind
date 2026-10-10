"""Tests for the synthetic business-data generator and the committed demo bundles."""

import json
from pathlib import Path

import pandas as pd
import pytest

from factorymind.bundle import MAX_BUNDLE_BYTES
from factorymind.engine import analyze
from factorymind.synthetic import generate_business_data, to_csv_bytes

ROOT = Path(__file__).parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "tiny_bundle"
DEMO_BUNDLES = sorted(p for p in (ROOT / "demo_bundles").glob("*") if p.is_dir())

AS_OF = pd.Timestamp("2015-01-01 08:00:00")
MACHINES = pd.DataFrame({
    "machineID": [3, 1, 2, 4, 5],
    "model": ["model3", "model3", "model4", "model1", "model2"],
})


def generate(seed=7, machines=MACHINES):
    return generate_business_data(machines, AS_OF, seed)


def test_same_seed_gives_byte_identical_csvs():
    assert to_csv_bytes(generate(seed=7)) == to_csv_bytes(generate(seed=7))


def test_different_seeds_differ():
    assert to_csv_bytes(generate(seed=7)) != to_csv_bytes(generate(seed=8))


def test_every_machine_has_a_schedule_row():
    schedule = generate()["production_schedule"]
    assert sorted(schedule["machine_id"]) == [1, 2, 3, 4, 5]
    assert (schedule["units_per_hour"] > 0).all()


def test_machines_of_the_same_model_make_the_same_product():
    schedule = generate()["production_schedule"].set_index("machine_id")
    assert schedule.loc[1, "product"] == schedule.loc[3, "product"] == "P-300"


def test_every_scheduled_product_has_inventory_and_orders():
    t = generate()
    products = set(t["production_schedule"]["product"])
    assert set(t["inventory"]["product"]) == products
    assert set(t["orders"]["product"]) == products


def test_inventory_and_orders_are_sane():
    t = generate()
    inv, orders = t["inventory"], t["orders"]
    assert (inv["safety_stock"] < inv["on_hand"]).all()
    assert (orders["qty"] > 0).all()
    assert (orders["due_date"] > AS_OF).all()
    assert (orders["due_date"] <= AS_OF + pd.Timedelta(days=6)).all()
    assert orders["order_id"].is_unique


def test_orders_keep_the_plant_near_capacity():
    """By each due date, demand uses the stock above safety level plus 90–95% of capacity, so
    normal running ships on time but downtime would eat into safety stock."""
    t = generate()
    capacity = t["production_schedule"].groupby("product")["units_per_hour"].sum()
    for product, inv in t["inventory"].set_index("product").iterrows():
        orders = t["orders"][t["orders"]["product"] == product].sort_values("due_date")
        for due, demand in zip(orders["due_date"], orders["qty"].cumsum()):
            hours = (due - AS_OF) / pd.Timedelta(hours=1)
            made = capacity[product] * hours
            spare = inv["on_hand"] - inv["safety_stock"]
            assert spare + 0.88 * made < demand < spare + 0.97 * made


def test_generated_data_passes_bundle_validation_as_synthetic():
    files = {p.name: p.read_bytes() for p in FIXTURE.glob("*.csv")
             if p.stem in {"telemetry", "errors", "maint", "machines"}}
    machines = pd.read_csv(FIXTURE / "machines.csv")
    files |= to_csv_bytes(generate_business_data(machines, AS_OF, seed=1))
    report = analyze(files)
    assert len(report.machines) == 2
    assert report.data_provenance["orders"] == "synthetic"


@pytest.mark.parametrize("bundle", DEMO_BUNDLES, ids=lambda p: p.name)
def test_demo_bundle_is_valid_and_small_enough(bundle):
    files = {p.name: p.read_bytes() for p in bundle.iterdir() if p.is_file()}
    assert sum(len(b) for b in files.values()) < MAX_BUNDLE_BYTES
    report = analyze(files)
    assert len(report.machines) == 10
    assert report.data_provenance["telemetry"] == "real"
    assert report.data_provenance["orders"] == "synthetic"


def test_a_demo_bundle_has_a_comp2_failure_building():
    if not DEMO_BUNDLES:
        pytest.skip("demo bundles not generated yet")
    upcoming = [f for b in DEMO_BUNDLES
                for f in json.loads((b / "manifest.json").read_text())["scenario"]["upcoming_failures"]]
    assert any(f["comp"] == "comp2" for f in upcoming)
