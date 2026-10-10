"""Plant cost assumptions used by the cascade and intervention simulator."""

from __future__ import annotations

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
