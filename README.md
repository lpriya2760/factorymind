# FactoryMind

Predict machine failures, trace their cascade to quality, inventory and customer orders, simulate interventions, and have Gemini recommend the smallest action that prevents the largest loss.

Built for the Google AI Builder Cup 2026 (Manufacturing). Spec: [#1](https://github.com/lpriya2760/factorymind/issues/1).

## Run locally

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`). uv installs Python 3.12 and the dependencies. On a Mac, XGBoost also needs the OpenMP runtime (`brew install libomp`).

```bash
uv run streamlit run app.py
```

Click one of the **demo bundles** next to the uploader, or upload your own files. Each demo can also be downloaded as a zip to see the format.

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
- Orders fall due 1–5 days after the data ends. Safety stock is 2–4 hours of production. By each due date, demand uses the stock above safety level plus 90–95% of what the machines can make. So normal running ships on time and keeps safety stock intact, while losing a machine for a shift eats into safety stock and can make orders late.

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
- **drift(h)** = drift now + its growth per hour over the last 24h × min(h, TREND_HORIZON_H). One day of trend is projected two days ahead, then held flat. Drift never shrinks, so the defect rate never falls.
- **scrap cost** = the sum over hours of units per hour (from the production schedule) × defect(h) × scrap cost per unit (a plant assumption, $15 by default)

| Parameter | Value | Meaning |
|---|---|---|
| BASE_RATE | 0.5% | defect rate of a healthy machine |
| TYPICAL_LIFE_H | 4,320 (180 days) | typical component life |
| WEAR_WEIGHT | 1% | added defect rate at one typical life; grows with wear² |
| WEAR_CAP | 2 | wear beyond two lifetimes adds nothing more |
| DRIFT_DEADBAND | 4% | sensor gaps below this are normal noise |
| DRIFT_WEIGHT | 0.5 | added defect rate per unit of drift beyond the deadband |
| DRIFT_CAP | 50% | drift beyond the deadband is capped here |
| TREND_HORIZON_H | 48 | the last 24h drift trend is projected this many hours, then held flat |
| ERROR_WEIGHT | 0.5% | added defect rate per error in the last 24h |
| MAX_RATE | 50% | the defect rate never exceeds this |

**Example from `comp2_failure_building`, machine 90:**
- Its rotation is 15.8% below normal and the gap is widening, and it logged 2 errors in the last 24h.
- The defect rate goes from 7.5% now to 12.9% in 48h, which is $2,746 of scrap. The fleet median is 1.3%.

Quality can also degrade *without* an imminent failure. Machine 72 in the same bundle runs with pressure 24% above normal and projects 10.6% defects, although it doesn't fail for months.

All four sensors count toward drift. The spec mentioned only vibration and pressure, but comp2 problems show up in rotation.

**Why this formula, and how the parameters were chosen:** see [ADR 0001](docs/adr/0001-quality-projection-formula.md). It covers the options considered, the rationale for each term, which parameters were tuned against the data and which are judgment calls, the limitations, and how to replace the formula with a model fitted to real defect data.

## Cascade: what a failure would cause

For each machine, FactoryMind simulates its product **hour by hour for a week** in two scenarios (`factorymind/cascade.py`):

- **If it fails:** it stops at hour 12 (the middle of the 24h prediction window) for the repair time, and the repair restores normal quality.
- **If it keeps running:** it produces at its projected defect rate.

**How a scenario is simulated:**
- Every machine making the product adds its *good* output each hour.
- Orders ship first in, first out by due date. Safety stock can be used. It counts as breached when stock minus orders due but not yet shipped falls below it.
- Late orders pay penalty per day × days late (part days count) × the penalty multiplier.

**The losses:**
- **Loss if it fails** = downtime + repair (part and labour) + penalties + excess scrap
- **Loss if it keeps running** = penalties + excess scrap
- **Expected loss** = p × the first + (1 − p) × the second

Excess scrap is scrap above the normal 0.5% rate.

**Example: machine 90 in `comp2_failure_building`** (>99% failure chance). If it fails:
- 297 units aren't made, and P-200 falls below safety stock at 30h.
- Order O-0001 ships late, with a $1,990 penalty.
- Total loss: $24,420.

If it keeps running, its degrading quality costs $10,890 of excess scrap. Expected loss: **$24,412**.

**Why it's built this way:** see [ADR 0002](docs/adr/0002-cascade-simulation.md). It covers each modelling choice and the two demo-data fixes the cascade needed.

## What to do: the intervention simulator

For each machine, FactoryMind evaluates five options. Each one **re-runs the cascade with changed inputs** (`factorymind/interventions.py`):

| Action | Failure chance after | Cost of the action |
|---|---|---|
| Do nothing | the model's prediction | $0 |
| Replace the at-risk part (3h planned stop) | re-predicted by the model, with the part new and its sensor back to normal | part + labour + lost output |
| Reroute while replacing: a same-model machine covers up to 50% of its rate | re-predicted by the model | part + labour + setup + uncovered output |
| Full service (6h, all four parts) | re-predicted by the model | 4 parts + labour + lost output |
| Slow down 25% for 24h | **assumed** stress effect, labelled in the app | lost output |

**Ranking:** net benefit = loss avoided − the action's cost. Ties go to the less disruptive action. Every hour of output not made anywhere costs the downtime cost per hour.

**Example, machine 90:** the best action is **replace comp2 while machine 47 covers**. It costs $6,074 and avoids $24,249 of expected loss, a net benefit of **$18,176**. Healthy machines get "do nothing".

**Why it's built this way:** see [ADR 0003](docs/adr/0003-intervention-simulator.md). It covers why a replacement also resets the sensor features, and why slowing down can't use the model, because low rotation is a symptom of failure.

## Plant assumptions

Every money figure and the ranking of actions depend on these. Edit them in the app's sidebar: the whole analysis, including Gemini's tools, re-runs straight away without re-uploading. **Reset to defaults** restores them, and changed values are listed.

| Assumption | Default | Meaning |
|---|---|---|
| Downtime cost | 2500 $ per hour | Lost contribution for every hour a machine's output isn't made anywhere: unplanned repairs, planned stops and slowdowns all use it. |
| Repair time after a failure | 8 hours | How long a machine is down after an unplanned component failure. |
| Component cost | 1200 $ per part | Parts cost of replacing one component. |
| Maintenance labour | 90 $ per hour | Technician cost for repairs, planned swaps and services. |
| Scrap cost | 15 $ per defective unit | What a defective unit costs; drives the quality losses. |
| Late-order penalty multiplier | 1 × | Scales every order's penalty per day late (1 = as in the orders file). |
| Planned part swap | 3 hours | Planned stop to replace one component (parts staged, no secondary damage). |
| Full service | 6 hours | Planned stop to replace all four components and recalibrate. |
| Reroute setup cost | 500 $ | One-off cost of moving a job to another machine. |
| Spare capacity on another machine | 0.5 share of its rate | How far above its scheduled rate another machine can run to cover (0.5 = +50%). |

**Try it on `comp2_failure_building`.** Lower the downtime cost from $2,500 to $300 an hour:
- Machine 90's expected loss falls from $24,412 to $6,822.
- The best action changes from *replace comp2, machine 47 covers* to *replace comp2 now*, because with downtime that cheap the $500 reroute setup no longer pays.

Figures built on synthetic inputs (inventory, production schedule, orders) carry a **synthetic data** badge.

## Gemini's recommendation

In the machine view, **Ask Gemini what to do** runs an agent built with Google ADK (`factorymind/agent.py`). Gemini calls seven tools over the analysis to see what matters, then writes a plain-language recommendation:
- `get_risk_overview`
- `get_machine_risk`
- `project_quality`
- `get_cascade`
- `simulate_interventions`
- `get_assumptions`
- `get_production_plan`: who makes the product, at what rate, stock and open orders

The app shows which tools it called and what they returned.

**Every figure is checked.** Gemini must quote only numbers that tools returned, and cite the tool. A guardrail then checks every number in the answer against the tool results. If any don't match, Gemini rewrites once; anything still unverified is flagged in the app, never shown silently.

**Example (machine 90):** "Replace comp2 now while machine 47 covers… a net benefit of $18,176 [simulate_interventions] by avoiding $24,249 [simulate_interventions] at a cost of $6,074 [simulate_interventions]…"

**Setup:**
- **Locally:** put an AI Studio key in `.env` as `GOOGLE_API_KEY=...`.
- **Cloud Run:** `deploy.sh` switches Gemini to Vertex AI in the project. It's configuration only, and no key is deployed.
- **Models:** `gemini-3.8-flash`, falling back to `gemini-3.5-flash` when overloaded. Override with `FACTORYMIND_MODEL` and `FACTORYMIND_FALLBACK_MODELS`.

**Tests:** they replace Gemini with a scripted model, so they run offline with no key.

**Design:** see [ADR 0004](docs/adr/0004-gemini-agent.md).

**Cached answers.** Each answer is cached by the bundle's *contents* (a zip and its loose files match), the machine, the full set of assumptions, and a fingerprint of the prompts.
- **The demo bundles never wait on Gemini.** `scripts/cache_recommendations.py` generates verified answers for every machine in them, plus the $300/h downtime scenario for machine 90. They ship with the app as `factorymind/recommendation_cache.json`, so they're instant even right after Cloud Run scales from zero.
- **Other answers** are cached in memory after their first call. Answers that fail the guardrail are never cached.
- **The logs** record `cache hit (no Gemini call)` or `Gemini call to <model> took <n>s` for every request.
- **Keeping it current:** a test fails if any demo machine lacks a current cached answer. Re-run the script after changing the prompts, the analysis or the demo data.

## Ask FactoryMind: follow-up questions

Below the recommendation, a chat answers follow-up questions about the uploaded machines, with the same agent and tools (`Chat` in `factorymind/agent.py`). Try "Which customers are affected if it fails?" or "What if we wait until the weekend?". Questions are about the machine on screen unless you name another.

- **It remembers the conversation.** Follow-ups like "and how much does that save?" work.
- **It checks every number.** Each figure must come from a tool result anywhere in the conversation. It gets one rewrite if not, and anything still unverified is flagged.
- **It says when it can't answer.** For a scenario the tools don't simulate (such as waiting until the weekend), it starts with "I can't answer that from the available data", says what's missing, and gives the closest figures the tools do have.
- **It starts fresh when the data changes.** A new bundle or changed assumptions start a new conversation.

## Costs and guardrails

The project runs on Google Cloud free-trial credits:

| Guardrail | Why |
|---|---|
| Cloud Run: `min-instances=0`, `max-instances=2`, 1 vCPU / 1 GiB | Costs nothing when idle, and is capped under load |
| The failure model runs **inside the app** | No Vertex AI prediction endpoint, which would bill by the hour even when idle |
| No Agent Engine, AutoML, Cloud SQL or GKE | Checked: none exist in the project |
| Gemini Flash, with cached answers for the demo bundles | Each live answer costs a fraction of a cent; the demo costs nothing |
| A budget alert on the billing account | Emails you before credits run out (it alerts, it doesn't cap spending) |

**Before recording or presenting,** run the [smoke test](docs/smoke-test.md) on the deployed URL.

## Deploy to Cloud Run

Requirements:
- the [gcloud CLI](https://cloud.google.com/sdk/docs/install), signed in with `gcloud auth login`
- a project with billing enabled, selected with `gcloud config set project <PROJECT_ID>`

Then run:

```bash
./scripts/deploy.sh
```

The script deploys from source to `asia-southeast1` with `min-instances=0` and `max-instances=2` (1 vCPU, 1 GiB) to stay inside free-trial credits. Override the region with `REGION=...`.
