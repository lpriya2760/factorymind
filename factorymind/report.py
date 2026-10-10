"""The AnalysisReport: the one structure the UI and the agent tools read.

Later tickets add risk, quality, cascade and intervention sections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MachineSummary:
    machine_id: int
    model: str
    age: int  # years


@dataclass(frozen=True)
class AnalysisReport:
    machines: list[MachineSummary]
    assumptions_used: dict[str, float]
    seed: int
    data_provenance: dict[str, str]  # table name -> "real" | "synthetic"

    def to_dict(self) -> dict:
        return asdict(self)
