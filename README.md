# FactoryMind

Predict machine failures, trace their cascade to quality, inventory and customer orders, simulate interventions, and have Gemini recommend the smallest action that prevents the largest loss.

Built for the Google AI Builder Cup 2026 (Manufacturing). Spec: [#1](https://github.com/lpriya2760/factorymind/issues/1).

## Run locally

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`). uv installs Python 3.12 and the dependencies.

```bash
uv run streamlit run app.py
```

Upload a bundle: try the tiny one in `tests/fixtures/tiny_bundle/`.

## Test

```bash
uv run pytest
```

Tests run against the analysis engine's single entry point, `analyze(files, assumptions, seed) -> AnalysisReport`, using fixture bundles. They need no network.

## Bundle format

| File | Columns | Assumed provenance |
|---|---|---|
| `telemetry.csv` | datetime, machineID, volt, rotate, pressure, vibration | real |
| `errors.csv` | datetime, machineID, errorID | real |
| `maint.csv` | datetime, machineID, comp | real |
| `machines.csv` | machineID, model, age | real |
| `production_schedule.csv` | machine_id, product, units_per_hour | synthetic |
| `inventory.csv` | product, on_hand, safety_stock | synthetic |
| `orders.csv` | order_id, customer, product, qty, due_date, penalty_per_day, priority | synthetic |

- **Machine data:** the four machine tables come from the public [Azure Predictive Maintenance dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance). Kaggle file names like `PdM_telemetry.csv` are accepted.
- **Upload options:** upload the files individually or as one `.zip`, up to 30 MB in total.
- **Provenance:** add a `manifest.json` such as `{"provenance": {"orders": "real"}}` to override the assumed provenance.

## Deploy to Cloud Run

Requirements:
- the [gcloud CLI](https://cloud.google.com/sdk/docs/install), signed in with `gcloud auth login`
- a project with billing enabled, selected with `gcloud config set project <PROJECT_ID>`

Then run:

```bash
./scripts/deploy.sh
```

The script deploys from source to `asia-southeast1` with `min-instances=0` and `max-instances=2` (1 vCPU, 1 GiB) to stay inside free-trial credits. Override the region with `REGION=...`.
