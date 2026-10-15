"""Plant cost assumptions used by the cascade and intervention simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

DEFAULT_ASSUMPTIONS: dict[str, float] = {
    "downtime_cost_per_hour": 2500.0,  # USD of lost contribution per hour a machine is down
    "repair_hours": 8.0,  # unplanned downtime after a component failure
    "component_replacement_cost": 1200.0,  # USD per component, parts only
    "maintenance_labour_cost_per_hour": 90.0,  # USD
    "scrap_cost_per_unit": 15.0,  # USD per defective unit
    "penalty_multiplier": 1.0,  # scales every order's penalty_per_day
    "planned_replacement_hours": 3.0,  # planned part swap: parts staged, no secondary damage
    "full_service_hours": 6.0,  # replace all four components and recalibrate
    "reroute_setup_cost": 500.0,  # USD to move a job to another machine
    "spare_capacity_share": 0.5,  # how far above its scheduled rate another machine can run
}



@dataclass(frozen=True)
class AssumptionInfo:
    label: str
    unit: str
    step: float
    help: str
    max_value: float | None = None


# How each assumption is shown and explained in the app.
ASSUMPTION_INFO: dict[str, AssumptionInfo] = {
    "downtime_cost_per_hour": AssumptionInfo(
        "Downtime cost", "$ per hour", 100.0,
        "Lost contribution for every hour a machine's output isn't made anywhere: unplanned "
        "repairs, planned stops and slowdowns all use it."),
    "repair_hours": AssumptionInfo(
        "Repair time after a failure", "hours", 1.0,
        "How long a machine is down after an unplanned component failure.", 72.0),
    "component_replacement_cost": AssumptionInfo(
        "Component cost", "$ per part", 100.0, "Parts cost of replacing one component."),
    "maintenance_labour_cost_per_hour": AssumptionInfo(
        "Maintenance labour", "$ per hour", 10.0,
        "Technician cost for repairs, planned swaps and services."),
    "scrap_cost_per_unit": AssumptionInfo(
        "Scrap cost", "$ per defective unit", 1.0,
        "What a defective unit costs; drives the quality losses."),
    "penalty_multiplier": AssumptionInfo(
        "Late-order penalty multiplier", "×", 0.1,
        "Scales every order's penalty per day late (1 = as in the orders file).", 10.0),
    "planned_replacement_hours": AssumptionInfo(
        "Planned part swap", "hours", 0.5,
        "Planned stop to replace one component (parts staged, no secondary damage).", 72.0),
    "full_service_hours": AssumptionInfo(
        "Full service", "hours", 0.5,
        "Planned stop to replace all four components and recalibrate.", 72.0),
    "reroute_setup_cost": AssumptionInfo(
        "Reroute setup cost", "$", 50.0, "One-off cost of moving a job to another machine."),
    "spare_capacity_share": AssumptionInfo(
        "Spare capacity on another machine", "share of its rate", 0.05,
        "How far above its scheduled rate another machine can run to cover (0.5 = +50%).", 2.0),
}


def resolve_assumptions(overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    """Defaults with overrides applied. Rejects unknown keys and negative values."""
    resolved = dict(DEFAULT_ASSUMPTIONS)
    for key, value in (overrides or {}).items():
        if key not in DEFAULT_ASSUMPTIONS:
            raise ValueError(f"unknown assumption {key!r}")
        if value < 0:
            raise ValueError(f"assumption {key!r} must not be negative")
        resolved[key] = float(value)
    return resolved
