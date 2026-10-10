"""The analysis engine: the single entry point the UI and agent tools call. It never calls Gemini."""

from __future__ import annotations

from typing import Mapping

from factorymind.assumptions import resolve_assumptions
from factorymind.bundle import load_bundle
from factorymind.report import AnalysisReport, MachineSummary


def analyze(
    files: Mapping[str, bytes],
    assumptions: Mapping[str, float] | None = None,
    seed: int = 0,
) -> AnalysisReport:
    """Validate an uploaded bundle (file name -> contents) and analyse it.

    Raises BundleValidationError if the bundle is malformed, and ValueError for bad assumptions.
    """
    assumptions_used = resolve_assumptions(assumptions)
    bundle = load_bundle(files)
    machines = bundle.tables["machines"].sort_values("machineID")
    return AnalysisReport(
        machines=[
            MachineSummary(machine_id=int(m.machineID), model=str(m.model), age=int(m.age))
            for m in machines.itertuples()
        ],
        assumptions_used=assumptions_used,
        seed=seed,
        data_provenance=bundle.provenance,
    )
