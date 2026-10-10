"""The analysis engine: the single entry point the UI and agent tools call. It never calls Gemini."""

from __future__ import annotations

from typing import Mapping

from factorymind.assumptions import resolve_assumptions
from factorymind.bundle import load_bundle
from factorymind.features import latest_features
from factorymind.predict import assess
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

    features = latest_features(bundle.tables)
    machines = bundle.tables["machines"].set_index("machineID")
    summaries = [
        MachineSummary(
            machine_id=int(machine_id),
            model=str(machines.at[machine_id, "model"]),
            age=int(machines.at[machine_id, "age"]),
            component_risk=a.component_risk,
            top_drivers=a.top_drivers,
            risk_level=a.risk_level,
        )
        for machine_id, a in zip(features["machineID"], assess(features))
    ]
    summaries.sort(key=lambda m: (-m.max_risk, m.machine_id))

    return AnalysisReport(
        machines=summaries,
        assumptions_used=assumptions_used,
        seed=seed,
        data_provenance=bundle.provenance,
    )
