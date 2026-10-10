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
class AnalysisReport:
    machines: list[MachineSummary]  # riskiest first
    quality: dict[int, QualityProjection]  # by machine_id
    assumptions_used: dict[str, float]
    seed: int
    data_provenance: dict[str, str]  # table name -> "real" | "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)
