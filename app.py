"""FactoryMind Streamlit UI. All analysis goes through factorymind.engine.analyze()."""

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from factorymind.bundle import MAX_BUNDLE_BYTES, TABLES, BundleValidationError
from factorymind.engine import analyze
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

try:
    report = analyze({u.name: u.getvalue() for u in uploads})
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
