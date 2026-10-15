"""The intervention simulator: what each possible action would do, by re-running the cascade.

For each machine, every action changes the cascade's inputs (the machine's output, its failure
chance, its defect rate, and for rerouting another machine's output), then the same two-scenario
simulation as doing nothing is re-run (factorymind.cascade.run_scenarios):

- **Replace the at-risk component**: a planned stop of `planned_replacement_hours` now. The part's
  age resets, the sensor that component affects returns to this machine's normal level and recent
  errors clear; the failure chance is then **re-predicted by the failure model**.
- **Full service now**: a planned stop of `full_service_hours`; all four components replaced and
  all sensors back to normal; failure chance re-predicted by the model.
- **Reroute while replacing**: as replacing the component, but the healthiest machine of the same
  model covers this machine's output during the stop, up to `spare_capacity_share` of its rate.
- **Slow down**: 25% less output for 24h. The failure chance is reduced by an *assumed* stress
  effect, not by the model: in this data low rotation is a symptom of comp2 failing, so feeding a
  slower speed to the model would wrongly raise the risk.

One accounting rule for every action: each hour of a machine's output that isn't made anywhere
costs `downtime_cost_per_hour`. Net benefit = loss avoided (versus doing nothing) − the action's
cost. Actions are ranked by net benefit; ties go to the less disruptive action.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from factorymind.cascade import CASCADE_HORIZON_H, Plant, run_scenarios
from factorymind.features import COMPONENTS, ERRORS, FEATURES, WINDOWS_H
from factorymind.predict import assess
from factorymind.quality import BASE_RATE, defect_curve
from factorymind.report import Cascade, Intervention, MachineSummary, QualityProjection

# In the Azure PdM data each component's failures show up in one sensor.
COMPONENT_SENSOR = {"comp1": "volt", "comp2": "rotate", "comp3": "pressure", "comp4": "vibration"}
SLOWDOWN = 0.25  # share of output given up
SLOWDOWN_HOURS = 24  # the prediction window
STRESS_EXPONENT = 2  # assumed: failure hazard scales with (speed)²
DISRUPTION = {"do_nothing": 0, "reduce_speed": 1, "replace_component": 2, "reroute": 3,
              "maintain_now": 4}


def simulate_interventions(plant: Plant, features: pd.DataFrame,
                           machines: list[MachineSummary],
                           quality: dict[int, QualityProjection],
                           cascades: dict[int, Cascade]) -> dict[int, list[Intervention]]:
    """Ranked interventions for every machine. `features` has every reading (build_features);
    `quality` covers at least CASCADE_HORIZON_H hours."""
    by_id = {m.machine_id: m for m in machines}
    history = {int(m): rows.sort_values("datetime") for m, rows in features.groupby("machineID")}
    return {m.machine_id: _for_machine(plant, m, by_id, history, quality, cascades[m.machine_id])
            for m in machines}


def _for_machine(plant, machine, by_id, history, quality, cascade) -> list[Intervention]:
    a = plant.assumptions
    m = machine.machine_id
    p = sum(machine.component_risk.values())
    baseline = cascade.expected_loss
    units = plant.units[m]
    product = plant.product[m]
    makers = plant.makers.get(product, [])
    hours = np.arange(CASCADE_HORIZON_H)
    rows = history[m]
    q = quality[m]
    comp = machine.top_component
    options = []

    def add(lever, label, params, cost, p_after, risks_after, basis, note,
            production=None, excess=None):
        after = baseline if lever == "do_nothing" else run_scenarios(
            plant, m, p_after, production, excess)[2]
        options.append(dict(
            lever=lever, label=label, params=params, cost=round(cost, 2),
            failure_probability_after=round(p_after, 4),
            component_risk_after={c: round(v, 4) for c, v in risks_after.items()},
            effect_basis=basis, expected_loss_after=round(after, 2),
            loss_avoided=round(baseline - after, 2),
            net_benefit=round(baseline - after - cost, 2),
            disruption=DISRUPTION[lever], note=note,
        ))

    def production_with(mine: np.ndarray, extra: dict[int, np.ndarray] | None = None):
        out = {k: plant.good_output(k) for k in makers}
        if m in out:
            out[m] = mine
        for k, v in (extra or {}).items():
            out[k] = out[k] + v
        return out

    add("do_nothing", "Do nothing", {}, 0.0, p, machine.component_risk, "model",
        "Keep running as it is.")

    # Slow down: assumed effect on the failure hazard; fewer units, so less scrap too.
    speed = np.where(hours < SLOWDOWN_HOURS, 1 - SLOWDOWN, 1.0)
    p_slow = 1 - (1 - p) ** ((1 - SLOWDOWN) ** STRESS_EXPONENT)
    add("reduce_speed", f"Slow down {SLOWDOWN:.0%} for {SLOWDOWN_HOURS}h",
        {"speed_reduction": SLOWDOWN, "hours": SLOWDOWN_HOURS},
        a["downtime_cost_per_hour"] * SLOWDOWN * SLOWDOWN_HOURS, p_slow,
        {c: v * p_slow / p if p else 0.0 for c, v in machine.component_risk.items()},
        "assumption",
        f"Assumed: running {SLOWDOWN:.0%} slower cuts stress, so the failure hazard falls to "
        f"{(1 - SLOWDOWN) ** STRESS_EXPONENT:.0%} of today's. The model can't predict this: in "
        "this data low rotation is a symptom of failure, not a cure.",
        production_with(plant.good_output(m) * speed), plant.excess_scrap(m) * speed)

    # Planned work now: re-predict risk from a machine back to normal, then simulate the stop.
    def planned(components, stop_hours):
        sensors = [COMPONENT_SENSOR[c] for c in components]
        risks = assess(_reset(rows, components, sensors))[0].component_risk
        ages = {c: (0.0 if c in components else rows.iloc[-1][f"hours_since_{c}"])
                for c in COMPONENTS}
        fixed = q.drift_sensor in sensors
        defect = defect_curve(max(ages.values()), 0.0 if fixed else q.drift,
                              0.0 if fixed else q.drift_growth_per_hour, 0,
                              CASCADE_HORIZON_H - 1)
        running = hours >= math.ceil(stop_hours)
        mine = np.where(running, units * (1 - defect), 0.0)
        excess = np.where(running, (defect - BASE_RATE).clip(min=0) * units
                          * a["scrap_cost_per_unit"], 0.0)
        return risks, sum(risks.values()), mine, excess, ~running

    stop = a["planned_replacement_hours"]
    labour = a["maintenance_labour_cost_per_hour"]
    part = a["component_replacement_cost"]
    risks, p_new, mine, excess, stopped = planned([comp], stop)
    sensor = COMPONENT_SENSOR[comp]
    add("replace_component", f"Replace {comp} now", {"component": comp},
        part + labour * stop + a["downtime_cost_per_hour"] * stop, p_new, risks, "model",
        f"Planned {stop:g}h swap of {comp} now: its age resets and {sensor} returns to this "
        "machine's normal level. Failure chance re-predicted by the model.",
        production_with(mine), excess)

    # Reroute: same swap, with the healthiest same-model machine covering the stop.
    candidates = [k for k in makers if k != m and by_id[k].model == machine.model]
    if candidates:
        target = min(candidates, key=lambda k: (sum(by_id[k].component_risk.values()),
                                                plant.defect[k][0], k))
        moved = min(units, a["spare_capacity_share"] * plant.units[target])
        cover = np.where(stopped, moved, 0.0)
        cover_scrap = cover * (plant.defect[target] - BASE_RATE).clip(min=0) \
            * a["scrap_cost_per_unit"]
        uncovered = (1 - moved / units) if units else 1.0
        own = plant.units[target]
        # The target keeps its own schedule and makes the moved units on top, with spare capacity.
        add("reroute", f"Replace {comp}, machine {target} covers",
            {"component": comp, "target_machine": target, "stop_hours": stop,
             "units_per_hour_moved": round(moved, 1),
             "target_own_units_per_hour": round(own, 1),
             "target_keeps_its_own_schedule": True,
             "target_units_per_hour_while_covering": round(own + moved, 1),
             "spare_capacity_share": a["spare_capacity_share"],
             "units_per_hour_not_made": round(units - moved, 1)},
            part + labour * stop + a["reroute_setup_cost"]
            + a["downtime_cost_per_hour"] * stop * uncovered,
            p_new, risks, "model",
            f"As replacing {comp}, but machine {target} (same model, lowest risk) keeps its own "
            f"{own:.0f} units/h schedule and makes {moved:.0f} of this machine's {units:.0f} "
            f"units/h on top, using spare capacity (up to "
            f"{a['spare_capacity_share']:.0%} above its rate), during the {stop:g}h stop. The "
            f"other {units - moved:.0f} units/h aren't made anywhere and cost downtime.",
            production_with(mine, {target: cover * (1 - plant.defect[target])}),
            excess + cover_scrap)

    service = a["full_service_hours"]
    risks, p_new, mine, excess, _ = planned(COMPONENTS, service)
    add("maintain_now", "Full service now", {"hours": service},
        part * len(COMPONENTS) + labour * service + a["downtime_cost_per_hour"] * service,
        p_new, risks, "model",
        f"Planned {service:g}h service now: all four components replaced and every sensor back to "
        "this machine's normal level. Failure chance re-predicted by the model.",
        production_with(mine), excess)

    options.sort(key=lambda o: (-o["net_benefit"], o["disruption"]))
    return [Intervention(rank=i + 1, **o) for i, o in enumerate(options)]


def _reset(rows: pd.DataFrame, components: list[str], sensors: list[str]) -> pd.DataFrame:
    """The latest reading as it would look after the work: replaced components are new, the
    affected sensors read this machine's normal (median) levels, and recent errors are cleared."""
    row = rows.iloc[-1].copy()
    for c in components:
        row[f"hours_since_{c}"] = 0.0
    for s in sensors:
        for stat in ("mean", "std"):
            for w in WINDOWS_H:
                row[f"{s}_{stat}_{w}h"] = rows[f"{s}_{stat}_{w}h"].median()
    for e in ERRORS:
        row[f"{e}_count_24h"] = 0
    frame = pd.DataFrame([row])
    return frame.astype({f: float for f in FEATURES})
