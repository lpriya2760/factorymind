"""The AnalysisReport: the one structure the UI and the agent tools read.

Later tickets add quality, cascade and intervention sections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Driver:
    """A feature pushing a machine's risk up, in words the operator understands."""
    feature: str
    description: str
    impact: float  # contribution to the top component's log-odds; larger = stronger


@dataclass(frozen=True)
class MachineSummary:
    machine_id: int
    model: str
    age: int  # years
    component_risk: dict[str, float]  # comp1..comp4 -> probability of failing in the next 24h
    top_drivers: list[Driver]  # strongest first
    risk_level: str  # "low" | "medium" | "high"

    @property
    def top_component(self) -> str:
        return max(self.component_risk, key=self.component_risk.get)

    @property
    def max_risk(self) -> float:
        return max(self.component_risk.values())


@dataclass(frozen=True)
class QualityProjection:
    """Defect rate if the machine keeps running, hour by hour (see factorymind.quality)."""
    horizon_hours: list[int]  # 0, 1, ..., horizon
    defect_rate: list[float]  # share of units defective at each hour
    scrap_cost: float  # USD over the whole horizon
    # The inputs behind the projection, so it can be explained.
    units_per_hour: float
    component_age_hours: float  # since the oldest component was replaced
    drift: float  # largest relative gap from the machine's normal sensor level
    drift_sensor: str
    drift_growth_per_hour: float
    errors_24h: int


@dataclass(frozen=True)
class LateOrder:
    order_id: str
    customer: str
    product: str
    due_hours: float  # from the end of the data
    hours_late: float
    penalty: float  # USD
    shipped: bool  # False if it still hadn't shipped when the simulation ended


@dataclass(frozen=True)
class ScenarioOutcome:
    """What happens to the machine's product over the cascade horizon in one scenario."""
    late_orders: list[LateOrder]
    penalties: float
    excess_scrap_cost: float  # this machine's scrap above the normal defect rate
    downtime_cost: float
    repair_cost: float
    inventory_breach_hours: float | None  # when stock first falls below safety stock
    total_loss: float


@dataclass(frozen=True)
class Cascade:
    """How a machine's possible failure spreads to quality, inventory, orders and money
    (see factorymind.cascade)."""
    product: str | None
    failure_probability: float  # any component, next 24h
    failure_at_hours: float  # assumed time of failure
    downtime_hours: float
    lost_units: float  # not produced while the machine is down
    if_failure: ScenarioOutcome
    if_no_failure: ScenarioOutcome
    expected_loss: float  # weighted by failure_probability


@dataclass(frozen=True)
class AnalysisReport:
    machines: list[MachineSummary]  # riskiest first
    quality: dict[int, QualityProjection]  # by machine_id
    cascade: dict[int, Cascade]  # by machine_id
    assumptions_used: dict[str, float]
    seed: int
    data_provenance: dict[str, str]  # table name -> "real" | "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)
