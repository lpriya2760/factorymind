# FactoryMind

Predict machine failures, trace their cascade to quality, inventory and customer orders, simulate interventions, and have Gemini recommend the smallest action that prevents the largest loss.

Built for the Google AI Builder Cup 2026 (Manufacturing). Spec: [#1](https://github.com/lpriya2760/factorymind/issues/1).

## Run locally

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`). uv installs Python 3.12 and the dependencies. On a Mac, XGBoost also needs the OpenMP runtime (`brew install libomp`).

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
| `healthy_fleet` | No machine fails during the data or in the 14 days after it, a calm baseline |

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

## Failure model

FactoryMind predicts which component (comp1–comp4), if any, fails on each machine **in the next 24 hours**.

**Features** (`factorymind/features.py`, shared by training and the app). Each is computed at every machine's latest reading:
- rolling 3h and 24h average and standard deviation of voltage, rotation, pressure and vibration
- counts of each error type in the last 24h
- hours since each component was last replaced
- machine model and age

**Model:** a BigQuery ML `BOOSTED_TREE_CLASSIFIER` with class weighting, since only about 2% of readings precede a failure.
- **Training data:** every 3rd hour of Jan–Aug 2015.
- **Test data:** Sep–Dec 2015, which the model never saw. There's a one-day gap between the two, so no 24h prediction window crosses from training into the test period.
- **Running it:** the trained model is exported, converted to XGBoost JSON and runs inside the app with no network calls.
- **Explanations:** the "Why" column comes from each prediction's per-feature contributions (SHAP values), compared with the rest of the uploaded fleet.

**Held-out results** (`factorymind/model/metrics.json`):

| Component | AUC | Precision | Recall | Failures in test window |
|---|---|---|---|---|
| comp1 | 1.00 | 0.92 | 1.00 | 408 |
| comp2 | 1.00 | 1.00 | 1.00 | 686 |
| comp3 | 1.00 | 0.97 | 1.00 | 296 |
| comp4 | 1.00 | 0.99 | 1.00 | 380 |

The "failures" column counts test readings (sampled every 3h) that fall within 24h of a failure.

**What the warning looks like:** predicted risk is about 0 more than 36h before a failure. It starts climbing 24–30h before, and sits at about 0.99 for the whole final 24h.

**Why the scores are so high:** the Azure dataset is simulated, and every failure is preceded by a clean, consistent sensor anomaly starting about a day before. A real plant's data is noisier, so expect lower scores. The pipeline is the same either way; retrain it on your own history.

**Risk levels:** *high* means the riskiest component has a probability of 0.5 or more; *medium* means 0.2 or more. On the test period, "high" flags 1.9% of readings with 0.98 precision and 1.00 recall for "a failure in the next 24h".

**To retrain** (needs gcloud signed in and the Kaggle data in `data/raw/`):

```bash
uv run python scripts/train_model.py
```

The script:
1. builds the features and labels
2. uploads the training rows to BigQuery (dataset `factorymind`)
3. trains the model there, which takes about 20 minutes, mostly BigQuery ML start-up
4. exports it through a Cloud Storage bucket, converts it to JSON and rewrites `factorymind/model/`

Add `--skip-train` to only re-evaluate the current model.

## Quality projection

For each machine, FactoryMind projects the **defect rate and scrap cost if it keeps running** for the next *N* hours (6–168, chosen in the app). Public machine datasets have no defect records, so this is a transparent formula, not a learned model (`factorymind/quality.py`):

```
defect(h) = min(MAX_RATE,
                BASE_RATE
                + WEAR_WEIGHT  × min(wear(h), WEAR_CAP)²
                + DRIFT_WEIGHT × min(max(drift(h) − DRIFT_DEADBAND, 0), DRIFT_CAP)
                + ERROR_WEIGHT × errors in the last 24h)
```

- **wear(h)** = (hours since the oldest component was replaced + h) ÷ TYPICAL_LIFE_H
- **drift** = the largest gap between a sensor's 24h average and the machine's own normal level (its median 24h average across the uploaded data), as a share of normal
- **drift(h)** = drift now + its growth per hour over the last 24h × h. Drift never shrinks, so the defect rate never falls.
- **scrap cost** = the sum over hours of units per hour (from the production schedule) × defect(h) × scrap cost per unit (a plant assumption, $15 by default)

| Parameter | Value | Meaning |
|---|---|---|
| BASE_RATE | 0.5% | defect rate of a healthy machine |
| TYPICAL_LIFE_H | 4,320 (180 days) | typical component life |
| WEAR_WEIGHT | 1% | added defect rate at one typical life; grows with wear² |
| WEAR_CAP | 2 | wear beyond two lifetimes adds nothing more |
| DRIFT_DEADBAND | 2% | sensor gaps below this are normal noise |
| DRIFT_WEIGHT | 0.5 | added defect rate per unit of drift beyond the deadband |
| DRIFT_CAP | 50% | drift beyond the deadband is capped here |
| ERROR_WEIGHT | 0.5% | added defect rate per error in the last 24h |
| MAX_RATE | 50% | the defect rate never exceeds this |

**Example from `comp2_failure_building`, machine 90:**
- Its rotation is 15.8% below normal and the gap is widening, and it logged 2 errors in the last 24h.
- The defect rate goes from 8.5% now to 13.9% in 48h, which is $3,013 of scrap. The fleet median is 2.3%.

Quality can also degrade *without* an imminent failure. Machine 72 in the same bundle runs with pressure 24% above normal and projects 11.6% defects, although it doesn't fail for months.

All four sensors count toward drift. The spec mentioned only vibration and pressure, but comp2 problems show up in rotation.

**Why this formula, and how the parameters were chosen:** see [ADR 0001](docs/adr/0001-quality-projection-formula.md). It covers the options considered, the rationale for each term, which parameters were tuned against the data and which are judgment calls, the limitations, and how to replace the formula with a model fitted to real defect data.

## Deploy to Cloud Run

Requirements:
- the [gcloud CLI](https://cloud.google.com/sdk/docs/install), signed in with `gcloud auth login`
- a project with billing enabled, selected with `gcloud config set project <PROJECT_ID>`

Then run:

```bash
./scripts/deploy.sh
```

The script deploys from source to `asia-southeast1` with `min-instances=0` and `max-instances=2` (1 vCPU, 1 GiB) to stay inside free-trial credits. Override the region with `REGION=...`.
