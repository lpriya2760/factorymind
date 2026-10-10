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

## Demo bundles

`demo_bundles/` holds ready-to-upload bundles. Each has 10 machines × 30 days of real Azure PdM data plus synthetic business data, and comes as a folder and as a `.zip`:

| Bundle | Scenario |
|---|---|
| `comp2_failure_building` | One machine's comp2 fails 12h after the data ends; the rest of the fleet is healthy |
| `two_machines_at_risk` | Two machines fail on different components within 24h of the data ending |
| `healthy_fleet` | No failures, a calm baseline |

Each bundle's `manifest.json` records which tables are real or synthetic, and which failures actually happen in the 24h after the data ends. The app ignores that ground truth; it's there so you know what the demo should show.

**Synthetic business data** (`factorymind/synthetic.py`) is generated from a seed, so the same seed always gives the same files:
- Machines of the same model make the same product, so work can be rerouted between them.
- Orders fall due 1–5 days after the data ends, and are sized so the plant runs close to capacity. Losing a machine for a shift makes some orders late.

**To regenerate the bundles:**
1. Download the [Kaggle dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) and unzip it into `data/raw/` (git-ignored).
2. Run:

```bash
uv run python scripts/make_demo_bundles.py --seed 7
```

## Deploy to Cloud Run

Requirements:
- the [gcloud CLI](https://cloud.google.com/sdk/docs/install), signed in with `gcloud auth login`
- a project with billing enabled, selected with `gcloud config set project <PROJECT_ID>`

Then run:

```bash
./scripts/deploy.sh
```

The script deploys from source to `asia-southeast1` with `min-instances=0` and `max-instances=2` (1 vCPU, 1 GiB) to stay inside free-trial credits. Override the region with `REGION=...`.
