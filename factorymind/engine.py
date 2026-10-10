"""The analysis engine: the single entry point the UI and agent tools call. It never calls Gemini."""

from __future__ import annotations

from typing import Mapping

from factorymind.assumptions import resolve_assumptions
from factorymind.bundle import load_bundle
from factorymind.cascade import CASCADE_HORIZON_H, build_cascades, prepare_plant
from factorymind.features import build_features
from factorymind.interventions import simulate_interventions
from factorymind.predict import assess
from factorymind.quality import project_quality
from factorymind.report import AnalysisReport, MachineSummary

DEFAULT_QUALITY_HORIZON_H = 48


def analyze(
    files: Mapping[str, bytes],
    assumptions: Mapping[str, float] | None = None,
    seed: int = 0,
    quality_horizon_hours: int = DEFAULT_QUALITY_HORIZON_H,
) -> AnalysisReport:
    """Validate an uploaded bundle (file name -> contents) and analyse it.

    Raises BundleValidationError if the bundle is malformed, and ValueError for bad arguments.
    """
    if quality_horizon_hours < 1:
        raise ValueError("quality_horizon_hours must be at least 1")
    assumptions_used = resolve_assumptions(assumptions)
    bundle = load_bundle(files)

    features = build_features(bundle.tables)
    latest = features.groupby("machineID", sort=True).tail(1).reset_index(drop=True)
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
        for machine_id, a in zip(latest["machineID"], assess(latest))
    ]
    summaries.sort(key=lambda m: (-m.max_risk, m.machine_id))

    schedule = bundle.tables["production_schedule"]
    scrap_cost = assumptions_used["scrap_cost_per_unit"]
    week = project_quality(features, schedule, scrap_cost, CASCADE_HORIZON_H)
    plant = prepare_plant(bundle.tables, as_of=bundle.tables["telemetry"]["datetime"].max(),
                          quality=week, assumptions=assumptions_used)
    cascades = build_cascades(
        plant, {m.machine_id: sum(m.component_risk.values()) for m in summaries})

    return AnalysisReport(
        machines=summaries,
        quality=project_quality(features, schedule, scrap_cost, quality_horizon_hours),
        cascade=cascades,
        interventions=simulate_interventions(plant, features, summaries, week, cascades),
        assumptions_used=assumptions_used,
        seed=seed,
        data_provenance=bundle.provenance,
    )
