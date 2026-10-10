"""FactoryMind Streamlit UI. All analysis goes through factorymind.engine.analyze()."""

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from factorymind.bundle import MAX_BUNDLE_BYTES, TABLES, BundleValidationError
from factorymind import quality
from factorymind.engine import DEFAULT_QUALITY_HORIZON_H, analyze
from factorymind.predict import MODEL_DIR

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

if not uploads:
    st.info("Upload a bundle to start.")
    st.stop()

@st.cache_data(show_spinner="Analysing…", max_entries=8)
def run_analysis(files: tuple[tuple[str, bytes], ...], quality_horizon_hours: int):
    return analyze(dict(files), quality_horizon_hours=quality_horizon_hours)


# The horizon slider is drawn further down; read its value first so the analysis uses it.
st.session_state.setdefault("horizon", DEFAULT_QUALITY_HORIZON_H)
try:
    report = run_analysis(tuple((u.name, u.getvalue()) for u in uploads),
                          st.session_state["horizon"])
except BundleValidationError as exc:
    st.error(f"The bundle has {len(exc.errors)} problem(s). Fix them and upload again.")
    st.dataframe(pd.DataFrame([asdict(e) for e in exc.errors]), hide_index=True, width="stretch")
    st.stop()

synthetic = [name for name, p in report.data_provenance.items() if p == "synthetic"]
if synthetic:
    st.warning("Synthetic inputs: " + ", ".join(f"`{name}`" for name in synthetic))

LEVEL_BADGES = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}
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
    f"scrap costs ${report.assumptions_used['scrap_cost_per_unit']:,.0f} a unit"
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
        "- **drift(h)** = drift now + its growth per hour over the last 24h × *h* (never shrinks)\n"
        "- **scrap cost** = Σ over the hours of units per hour × defect(h) × scrap cost per unit"
    )
    st.dataframe(
        pd.DataFrame([{"parameter": k, "value": v, "meaning": m}
                      for k, (v, m) in quality.PARAMETERS.items()]),
        hide_index=True, width="stretch",
    )
