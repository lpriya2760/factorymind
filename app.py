"""FactoryMind Streamlit UI. All analysis goes through factorymind.engine.analyze()."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from factorymind.bundle import MAX_BUNDLE_BYTES, TABLES, BundleValidationError
from factorymind import quality
from factorymind.agent import recommend
from factorymind.assumptions import ASSUMPTION_INFO, DEFAULT_ASSUMPTIONS
from factorymind.cascade import CASCADE_HORIZON_H, FAILURE_AT_H
from factorymind.engine import DEFAULT_QUALITY_HORIZON_H, analyze
from factorymind.predict import MODEL_DIR

# Local development reads GOOGLE_API_KEY from .env; Cloud Run uses Vertex AI settings instead.
load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(page_title="FactoryMind", page_icon="🏭", layout="wide")
st.title("FactoryMind")
st.caption(
    "Predict machine failures, trace their cascade to quality, inventory and orders, "
    "and find the smallest action that prevents the largest loss."
)

uploads = st.file_uploader(
    f"Upload a bundle: the {len(TABLES)} CSV files (or one .zip), plus an optional manifest.json. "
    f"Max {MAX_BUNDLE_BYTES // 2**20} MB.",
    type=["csv", "zip", "json"],
    accept_multiple_files=True,
)

with st.expander("Expected files"):
    st.dataframe(
        pd.DataFrame([
            {"file": f"{name}.csv", "columns": ", ".join(spec.columns),
             "assumed provenance": spec.provenance}
            for name, spec in TABLES.items()
        ]),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        'Kaggle names such as `PdM_telemetry.csv` also work. To change provenance, add a '
        '`manifest.json` like `{"provenance": {"orders": "real"}}`.'
    )



def assumption_panel() -> dict[str, float]:
    """Sidebar inputs for the plant assumptions; returns their current values."""
    with st.sidebar:
        st.header("Plant assumptions")
        st.caption("Every money figure and the ranking of actions use these. Change one and the "
                   "whole analysis re-runs.")
        if st.button("Reset to defaults"):
            for key, value in DEFAULT_ASSUMPTIONS.items():
                st.session_state[f"assume_{key}"] = value
        values = {}
        for key, info in ASSUMPTION_INFO.items():
            default = DEFAULT_ASSUMPTIONS[key]
            st.session_state.setdefault(f"assume_{key}", default)
            values[key] = st.number_input(
                f"{info.label} ({info.unit})", min_value=0.0, max_value=info.max_value,
                step=info.step, format="%.0f" if info.step >= 1 else "%.2f",
                key=f"assume_{key}", help=f"{info.help} Default: {default:g}.")
        changed = {k: v for k, v in values.items() if v != DEFAULT_ASSUMPTIONS[k]}
        if changed:
            st.info("Changed from default: " + "; ".join(
                f"{ASSUMPTION_INFO[k].label.lower()} {v:g} (default {DEFAULT_ASSUMPTIONS[k]:g})"
                for k, v in changed.items()))
    return values


assumptions = assumption_panel()

if not uploads:
    st.info("Upload a bundle to start.")
    st.stop()

@st.cache_data(show_spinner="Analysing…", max_entries=8)
def run_analysis(files: tuple[tuple[str, bytes], ...], quality_horizon_hours: int,
                 assumptions: tuple[tuple[str, float], ...]):
    return analyze(dict(files), dict(assumptions), quality_horizon_hours=quality_horizon_hours)


# The horizon slider is drawn further down; read its value first so the analysis uses it.
st.session_state.setdefault("horizon", DEFAULT_QUALITY_HORIZON_H)
try:
    report = run_analysis(tuple((u.name, u.getvalue()) for u in uploads),
                          st.session_state["horizon"], tuple(sorted(assumptions.items())))
except BundleValidationError as exc:
    st.error(f"The bundle has {len(exc.errors)} problem(s). Fix them and upload again.")
    st.dataframe(pd.DataFrame([asdict(e) for e in exc.errors]), hide_index=True, width="stretch")
    st.stop()

synthetic = [name for name, p in report.data_provenance.items() if p == "synthetic"]
if synthetic:
    st.warning("Synthetic inputs: " + ", ".join(f"`{name}`" for name in synthetic))

LEVEL_BADGES = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}


def synthetic_badge(*tables: str) -> str:
    """A badge for figures built on synthetic inputs."""
    return (" :orange-badge[synthetic data]"
            if any(report.data_provenance.get(t) == "synthetic" for t in tables) else "")
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]

st.subheader(f"Failure risk in the next 24h ({len(report.machines)} machines)")
high = [m for m in report.machines if m.risk_level == "high"]
if high:
    st.error("  \n".join(
        f"**Machine {m.machine_id}**: {m.top_component} is "
        f"{'>99%' if m.max_risk > 0.99 else f'{m.max_risk:.0%}'} likely to fail in the next 24h"
        for m in high
    ))

st.dataframe(
    pd.DataFrame([
        {
            "Risk": LEVEL_BADGES[m.risk_level],
            "Machine": m.machine_id,
            "Model": m.model,
            "Age": m.age,
            # For low-risk machines the "riskiest" part and its reasons are noise.
            "Riskiest part": m.top_component if m.risk_level != "low" else "—",
            **{c: m.component_risk[c] * 100 for c in COMPONENTS},
            "Why": (" · ".join(d.description for d in m.top_drivers)
                    if m.risk_level != "low" else ""),
        }
        for m in report.machines
    ]),
    hide_index=True,
    width="stretch",
    column_config={
        c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%.1f%%")
        for c in COMPONENTS
    } | {"Why": st.column_config.TextColumn("Why", width="large")},
)

with st.expander("About the failure model"):
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    st.markdown(
        "A BigQuery ML boosted-tree classifier, trained on "
        f"{metrics['train_period']} of the Azure Predictive Maintenance data and tested on "
        f"{metrics['test_period']}, which it never saw. Its features are rolling 3h and 24h "
        "sensor averages and variability, error counts, time since each part was replaced, and "
        "machine model and age. It runs inside this app, with no external calls."
    )
    st.dataframe(
        pd.DataFrame(metrics["components"]).T.rename_axis("component").reset_index(),
        hide_index=True,
    )
    st.caption(
        "The public dataset is simulated, and every failure is preceded by a clear sensor "
        "anomaly, so these held-out scores are higher than you'd expect in a real plant."
    )

st.subheader("Machine detail")
by_id = {m.machine_id: m for m in report.machines}
machine_id = st.selectbox(
    "Machine", list(by_id),
    format_func=lambda i: f"Machine {i} · {LEVEL_BADGES[by_id[i].risk_level]} failure risk",
)
st.slider("Hours ahead", min_value=6, max_value=168, step=6, key="horizon")

st.markdown("#### Quality if it keeps running")
q = report.quality[machine_id]
horizon = q.horizon_hours[-1]
now_col, later_col, scrap_col = st.columns(3)
now_col.metric("Defect rate now", f"{q.defect_rate[0]:.1%}")
later_col.metric(f"Defect rate in {horizon}h", f"{q.defect_rate[-1]:.1%}",
                 delta=f"{(q.defect_rate[-1] - q.defect_rate[0]) * 100:+.1f} pts",
                 delta_color="inverse")
scrap_col.metric(f"Scrap cost over {horizon}h", f"${q.scrap_cost:,.0f}")
st.line_chart(
    pd.DataFrame({"Defect rate (%)": [r * 100 for r in q.defect_rate]},
                 index=pd.Index(q.horizon_hours, name="Hours from now")),
    y_label="Defect rate (%)",
)
growth = (f", widening {q.drift_growth_per_hour * 24:.1%} a day"
          if q.drift_growth_per_hour > 0 else "")
st.caption(
    f"Why: oldest component replaced {q.component_age_hours / 24:.0f} days ago · "
    f"{q.drift_sensor} is {q.drift:.1%} away from this machine's normal level{growth} · "
    f"{q.errors_24h} error(s) in the last 24h · {q.units_per_hour:.0f} units/hour scheduled · "
    f"scrap costs \\${report.assumptions_used['scrap_cost_per_unit']:,.0f} a unit"
)

with st.expander("How quality is projected"):
    st.markdown(
        "Public machine datasets have no defect records, so FactoryMind uses a transparent "
        "formula instead of a learned model. The defect rate *h* hours from now is:"
    )
    st.latex(
        r"\text{defect}(h) = \min\Big(\text{MAX\_RATE},\ \text{BASE\_RATE}"
        r" + \text{WEAR\_WEIGHT}\cdot\min(\text{wear}(h), \text{WEAR\_CAP})^2"
        r" + \text{DRIFT\_WEIGHT}\cdot\min\big(\max(\text{drift}(h) - \text{DRIFT\_DEADBAND}, 0),"
        r" \text{DRIFT\_CAP}\big) + \text{ERROR\_WEIGHT}\cdot\text{errors}_{24h}\Big)"
    )
    st.markdown(
        "- **wear(h)** = (hours since the oldest component was replaced + *h*) ÷ TYPICAL_LIFE_H\n"
        "- **drift** = the largest gap between a sensor's 24h average and this machine's own "
        "normal level (its median across the uploaded data), as a share of normal\n"
        "- **drift(h)** = drift now + its growth per hour over the last 24h × min(*h*, "
        "TREND_HORIZON_H): a day's trend is projected two days ahead, then held flat. "
        "Drift never shrinks.\n"
        "- **scrap cost** = Σ over the hours of units per hour × defect(h) × scrap cost per unit"
    )
    st.dataframe(
        pd.DataFrame([{"parameter": k, "value": v, "meaning": m}
                      for k, (v, m) in quality.PARAMETERS.items()]),
        hide_index=True, width="stretch",
    )

st.markdown("#### What a failure would cause")
c = report.cascade[machine_id]
f, ok = c.if_failure, c.if_no_failure
p = c.failure_probability
p_text = ">99%" if p > 0.99 else "<1%" if p < 0.01 else f"{p:.0%}"


def breach_text(hours):
    return f"below safety stock in {hours:.0f}h" if hours is not None else "stays above safety stock"


chain = st.columns(5)
with chain[0].container(border=True):
    st.markdown("**⚙️ Machine**")
    st.metric("Failure in 24h", p_text)
    st.caption(f"If it fails: down {c.downtime_hours:g}h from about hour {c.failure_at_hours:g}")
with chain[1].container(border=True):
    st.markdown("**🔧 Quality**")
    st.metric("Excess scrap if it fails", f"${f.excess_scrap_cost:,.0f}")
    st.caption(f"\\${ok.excess_scrap_cost:,.0f} if it keeps running for {CASCADE_HORIZON_H}h; "
               "a repair restores normal quality")
with chain[2].container(border=True):
    st.markdown(f"**📦 Inventory ({c.product or 'no product'})**"
                + synthetic_badge("inventory", "production_schedule"))
    st.metric("Units not made", f"{c.lost_units:,.0f}")
    st.caption(f"If it fails: {breach_text(f.inventory_breach_hours)}. "
               f"If not: {breach_text(ok.inventory_breach_hours)}.")
with chain[3].container(border=True):
    st.markdown("**🚚 Orders**" + synthetic_badge("orders"))
    st.metric("Late if it fails", f"{len(f.late_orders)}")
    st.caption(f"\\${f.penalties:,.0f} penalties if it fails. "
               f"If not: {len(ok.late_orders)} late, \\${ok.penalties:,.0f}.")
with chain[4].container(border=True):
    st.markdown("**💰 Loss**")
    st.metric("Expected loss", f"${c.expected_loss:,.0f}")
    st.caption(f"\\${f.total_loss:,.0f} if it fails · \\${ok.total_loss:,.0f} if not, "
               f"weighted by the {p_text} failure chance")

def scenario_column(o):
    return [str(len(o.late_orders)), breach_text(o.inventory_breach_hours).capitalize()] + [
        f"${x:,.0f}" for x in (o.penalties, o.excess_scrap_cost, o.downtime_cost, o.repair_cost,
                               o.total_loss)]


st.dataframe(
    pd.DataFrame(
        {"If it fails": scenario_column(f), "If it keeps running": scenario_column(ok)},
        index=["Late orders", "Safety stock", "Penalties", "Excess scrap", "Downtime",
               "Repair", "Total loss"],
    ),
    width="stretch",
)

if f.late_orders:
    st.markdown("**Orders late if it fails**")
    st.dataframe(
        pd.DataFrame([
            {"Order": o.order_id, "Customer": o.customer, "Product": o.product,
             "Due": f"in {o.due_hours:.0f}h",
             "Hours late": o.hours_late if o.shipped else f"{o.hours_late:.0f}+ (not shipped)",
             "Penalty ($)": o.penalty}
            for o in f.late_orders
        ]),
        hide_index=True, width="stretch",
    )

with st.expander("How the cascade is simulated"):
    st.markdown(
        f"Hour 0 is the end of the uploaded data. FactoryMind simulates the next "
        f"{CASCADE_HORIZON_H}h hour by hour, twice:\n\n"
        f"- **If it fails**: it fails at hour {FAILURE_AT_H} (the middle of the 24h prediction "
        "window) and is down for the repair time. Repair restores normal quality.\n"
        "- **If it keeps running**: it keeps producing at its projected defect rate.\n\n"
        "Every machine making the same product adds its *good* output each hour. Orders ship "
        "first in, first out by due date when there is enough stock; safety stock can be used. "
        "Safety stock is breached when stock minus orders due but not yet shipped falls below "
        "it. A late order's penalty is penalty per day × days late (part days count) × the "
        "penalty multiplier.\n\n"
        "**Loss if it fails** = downtime + repair (part and labour) + penalties + excess scrap. "
        "**Loss if it keeps running** = penalties + excess scrap. **Expected loss** weights "
        "them by the failure chance. Excess scrap is scrap above the normal "
        f"{quality.BASE_RATE:.1%} defect rate."
    )
    st.caption("Costs use the plant assumptions in the sidebar: " + ", ".join(
        f"{k.replace('_', ' ')} {v:g}" for k, v in report.assumptions_used.items()))

st.markdown("#### What to do about it")
options = report.interventions[machine_id]
best = options[0]
if best.lever == "do_nothing":
    alt = options[1]
    effect = (f"to avoid \\${alt.loss_avoided:,.0f} of expected loss" if alt.loss_avoided > 0
              else "and wouldn't reduce the expected loss")
    st.success(f"**Best: do nothing.** No action pays for itself: the best alternative, "
               f"*{alt.label.lower()}*, would cost \\${alt.cost:,.0f} {effect}.")
else:
    st.success(f"**Best: {best.label}.** Costs \\${best.cost:,.0f} and avoids "
               f"\\${best.loss_avoided:,.0f} of expected loss: a net benefit of "
               f"**\\${best.net_benefit:,.0f}**. {best.note}")

DISRUPTION_LABELS = ["None", "Low", "Medium", "Medium-high", "High"]


def money(x: float) -> str:
    return f"−${-x:,.0f}" if x < 0 else f"${x:,.0f}"


st.dataframe(
    pd.DataFrame([
        {
            "Rank": o.rank,
            "Action": o.label,
            "Cost": money(o.cost),
            "Failure chance after": (
                (">99%" if o.failure_probability_after > 0.99
                 else "<1%" if o.failure_probability_after < 0.01
                 else f"{o.failure_probability_after:.0%}")
                + (" (assumed)" if o.effect_basis == "assumption" else "")),
            "Expected loss after": money(o.expected_loss_after),
            "Loss avoided": money(o.loss_avoided),
            "Net benefit": money(o.net_benefit),
            "Disruption": DISRUPTION_LABELS[o.disruption],
        }
        for o in options
    ]),
    hide_index=True, width="stretch",
)

with st.expander("How each action is simulated"):
    st.markdown(
        "Every action changes the inputs and re-runs the same failure/no-failure cascade as "
        "doing nothing. **Net benefit = loss avoided − the action's cost**. Ties go to the less "
        "disruptive action. One accounting rule applies everywhere: each hour of a machine's "
        "output that isn't made anywhere costs the downtime cost per hour.\n\n"
        + "\n".join(f"- **{o.label}**: {o.note}" for o in options)
    )

st.markdown("#### 🤖 Gemini's recommendation")
bundle_key = hashlib.sha256(b"".join(u.getvalue() for u in uploads)).hexdigest()[:16]
recs = st.session_state.setdefault("recommendations", {})
rec_key = (bundle_key, machine_id, tuple(sorted(assumptions.items())))
if st.button("Ask Gemini what to do", key=f"ask-{machine_id}", type="primary"):
    with st.spinner("Gemini is checking the risk, the cascade and the options…"):
        try:
            recs[rec_key] = recommend({u.name: u.getvalue() for u in uploads}, machine_id,
                                      assumptions)
        except Exception as exc:  # noqa: BLE001 - shown to the operator, not swallowed
            st.error(f"Gemini couldn't answer right now: {exc}")
rec = recs.get(rec_key)
if rec is None:
    st.caption("Gemini reads the same analysis through tools and explains the best action. "
               "Every figure it quotes is checked against what the tools returned.")
else:
    with st.container(border=True):
        st.markdown(rec.text.replace("$", "\\$"))
    if rec.unverified_numbers:
        st.warning("Couldn't verify these figures against the tool results: "
                   + ", ".join(rec.unverified_numbers))
    else:
        st.caption(f"✅ Every figure was checked against the tool results"
                   f"{' after one rewrite' if rec.retried else ''} · {rec.model}")
    with st.expander(f"Tools Gemini used ({len(rec.tool_calls)})"):
        for call in rec.tool_calls:
            args = ", ".join(f"{k}={v}" for k, v in call.args.items())
            st.markdown(f"**`{call.name}({args})`**")
            st.json(call.result, expanded=False)
