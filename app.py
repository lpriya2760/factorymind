"""FactoryMind Streamlit UI. All analysis goes through factorymind.engine.analyze()."""

from dataclasses import asdict

import pandas as pd
import streamlit as st

from factorymind.bundle import MAX_BUNDLE_BYTES, TABLES, BundleValidationError
from factorymind.engine import analyze

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

st.subheader(f"Machines ({len(report.machines)})")
st.dataframe(
    pd.DataFrame([asdict(m) for m in report.machines]).rename(
        columns={"machine_id": "Machine", "model": "Model", "age": "Age (years)"}
    ),
    hide_index=True,
    width="stretch",
)
