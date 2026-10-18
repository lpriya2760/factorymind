# FactoryMind spec

This is the product spec the team agreed at the start of the hackathon (10 Oct 2026). It is kept as written, so it records the plan. It is also tracked as issue [#1](https://github.com/lpriya2760/factorymind/issues/1). How each part was built is in the [README](README.md) and the decision records in [`docs/adr/`](docs/adr/).

## Where the shipped app differs from the spec

- **Live data (listed as out of scope below):**
  - The app can stream a demo bundle through Google Cloud Pub/Sub, one hour every 2 seconds, and re-run the whole analysis as each hour arrives.
  - A command-line publisher and monitor do the same outside the app ([ADR 0006](docs/adr/0006-live-monitoring-over-pubsub.md)).
  - Replay of uploaded data (user stories 39–41) is built as well.
- **Alerts (not in the spec):**
  - A machine that becomes high-risk is posted to Slack automatically, once per risk episode.
  - The same alert can be sent by email with a button.
- **Interventions** ([ADR 0003](docs/adr/0003-intervention-simulator.md)):
  - There are four levers plus "do nothing": slow down, replace the at-risk component, replace it while a machine of the same model covers the work ("reroute"), and a full service ("maintain now").
  - Rerouting is modelled as cover during a repair, not as moving the whole schedule.
- **Loss range** ([ADR 0005](docs/adr/0005-monte-carlo-loss-range.md)):
  - The Monte Carlo draws whether the machine fails and in which hour of the 24h window. It doesn't draw repair time or costs.
  - Interventions are still ranked on the expected loss.
- **Demo data dates:** the Azure PdM readings are from 2015. The demo bundles shift them forward by whole days so each bundle ends on 18 Oct 2026. Predictions and figures are unchanged.
- **Gemini models:**
  - The deployed app uses Gemini 3.8 Flash through Vertex AI, falling back to 3.5 Flash.
  - Answers for every demo machine are shipped in a cache, so the demo never waits on Gemini.
- **Repository:** public, as the submission requires.

---

## Problem Statement

A plant operations manager sees machine telemetry and alarms, but not what a failing machine will *cost*. When a component starts to degrade, they can't easily answer: How likely is it to fail, and when? What happens to product quality if we keep running? Which customer orders will slip, and what's the penalty? Is it cheaper to slow down, replace a part now, or move the work to another machine? Today those answers sit in different systems and people's heads, so decisions are made late, by gut feel, after the loss has already spread from the machine to quality, inventory and customers.

## Solution

FactoryMind is a decision-intelligence web app. The operator uploads a machine telemetry slice together with the plant's business context (production schedule, inventory, orders). FactoryMind then:

1. Predicts, per machine, the probability that each component fails within the next 24 hours.
2. Projects product quality (defect rate and scrap cost) if the machine keeps running.
3. Traces the **cascade**: machine failure → lost throughput → inventory buffer drain → late orders → financial loss, with a range of losses from Monte Carlo simulation.
4. Simulates four **interventions** (reduce speed, maintain now, replace component, reroute to another machine) by re-running the real model and cascade under each one.
5. Has a Gemini agent recommend **the smallest action that prevents the largest loss**, explaining the trade-offs and quoting only numbers that came from the tools.

The operator can edit plant cost assumptions and see the recommendation change live, replay the telemetry to watch the risk build up, and ask follow-up questions in an "Ask FactoryMind" chat panel.

## User Stories

### Upload and data

1. As a plant operator, I want to upload a telemetry slice CSV, so that FactoryMind analyses my machines' recent condition.
2. As a plant operator, I want to upload error, maintenance and machine-metadata CSVs for the same machines, so that the prediction accounts for alarms, component age and machine model.
3. As a plant operator, I want to upload production schedule, inventory and orders CSVs, so that the cascade reflects my real business context.
4. As a plant operator, I want clear validation errors when a file has missing columns, wrong types or unknown machine IDs, so that I can fix the file instead of getting wrong results.
5. As a plant operator, I want a downloadable set of sample files, so that I know the expected format.
6. As a demo presenter, I want 2–3 prepared demo bundles, including one where a component failure is clearly building, so that the demo is reliable and tells a story.
7. As a plant operator, I want the app to show which inputs are synthetic, so that I know which numbers rest on assumed data.
8. As a plant operator, I want uploads to stay under the hosting size limit and to be told if mine is too large, so that the upload doesn't fail silently.

### Failure prediction

9. As a plant operator, I want a risk table of all machines ranked by highest component failure probability in the next 24h, so that I know where to look first.
10. As a plant operator, I want each machine's failure probability broken down by component (comp1–comp4), so that I know which part is at risk.
11. As a plant operator, I want to see the main drivers behind a prediction (e.g. rising vibration, days since comp2 was replaced, recent error counts), so that I can trust and sanity-check it.
12. As a judge or engineer, I want to see the model's evaluation metrics (per-component AUC, precision and recall on a held-out time period), so that I can judge how good the prediction is.
13. As a plant operator, I want risk levels colour-coded (low / medium / high), so that I can scan the table quickly.

### Quality projection

14. As a quality manager, I want the projected defect rate if the machine keeps running for the next N hours, so that I know when quality becomes unacceptable.
15. As a quality manager, I want the projected scrap cost over that horizon, so that quality loss is expressed in money.
16. As a judge, I want the quality formula documented in the app and README, so that the assumption is transparent.

### Cascade

17. As a plant manager, I want to see how a machine failure cascades to lost throughput, inventory buffer and order delivery, so that I understand the business impact, not just the machine status.
18. As a plant manager, I want to see which specific orders would be late, by how many hours, and their penalty, so that I can warn or prioritise customers.
19. As a plant manager, I want the time until inventory drops below safety stock, so that I know how much slack I have.
20. As a plant manager, I want the total expected loss and a P10–P90 range, so that I understand both the likely and the worst-case outcome.
21. As a plant manager, I want the cascade shown as a visual chain (machine → quality → inventory → orders → loss), so that I can explain it to others quickly.
22. As an engineer, I want the cascade to give identical results for the same inputs and seed, so that results are reproducible and testable.

### Intervention simulation

23. As a plant manager, I want to simulate reducing machine speed, so that I see the trade-off between lower failure risk and lower throughput.
24. As a plant manager, I want to simulate doing maintenance now versus later, so that I see planned downtime cost against failure risk.
25. As a plant manager, I want to simulate replacing the at-risk component, so that I see the effect of resetting its age.
26. As a plant manager, I want to simulate rerouting production to another machine of the same model, so that I see whether spare capacity absorbs the risk.
27. As a plant manager, I want each intervention to show its cost, expected loss avoided and net benefit, so that I can compare them on the same scale.
28. As a plant manager, I want interventions ranked by net benefit, with ties broken by least disruption, so that the best option is on top.
29. As a plant manager, I want to see the "do nothing" baseline next to the interventions, so that I know what I'm comparing against.

### Plant assumptions

30. As a plant manager, I want to edit cost assumptions (downtime cost per hour, component replacement cost, maintenance labour cost, scrap cost per unit, order penalties), so that the analysis matches my plant.
31. As a demo presenter, I want the ranking and recommendation to update when I change an assumption, so that I can show the system responding live.
32. As a plant manager, I want sensible defaults for every assumption, so that the app is useful immediately.

### Gemini agent and recommendation

33. As a plant manager, I want a plain-language recommendation naming the action, the timing and the money saved, so that I can act on it immediately.
34. As a plant manager, I want the recommendation to explain why it beats the alternatives and what it risks, so that I understand the trade-off.
35. As a plant manager, I want every number in the recommendation to come from FactoryMind's tools and show where it came from, so that I can trust it isn't made up.
36. As a plant manager, I want to ask follow-up questions in an "Ask FactoryMind" chat (e.g. "what if we wait until the weekend?", "which customers are affected?"), so that I can explore without clicking through screens.
37. As a plant manager, I want the agent to run the prediction, cascade and simulator tools to answer my questions, so that answers are computed rather than guessed.
38. As a plant manager, I want the agent to say when it can't answer from the available data, so that I'm not misled.

### Replay

39. As a demo presenter, I want a replay button that steps through the telemetry slice over time, so that the audience watches risk climb.
40. As a demo presenter, I want the agent's recommendation to appear when risk crosses a threshold during replay, so that the AI visibly steps in at the right moment.
41. As a demo presenter, I want to pause, reset and change replay speed, so that I control the pace of the demo.

### Deployment and operations

42. As a judge, I want a public URL that works end to end, so that I can try FactoryMind myself.
43. As the team, I want the app to scale to zero when idle and cap at a small number of instances, so that hosting stays within free-trial credits.
44. As the team, I want Gemini responses for the prepared demo bundles cached, so that rehearsals and judging don't add cost or hit rate limits.
45. As the team, I want to switch between the AI Studio key (development) and Vertex AI (deployed) through configuration only, so that development is free and the deployed app is reliable.
46. As the team, I want the trained model packaged with the app, so that predictions need no paid prediction endpoint.
47. As a judge reading the repo, I want a README with the architecture, the data sources, the quality formula and the cost assumptions, so that I can assess technical merit.

## Implementation Decisions

**Product name:** FactoryMind. Hackathon: Google AI Builder Cup 2026, Manufacturing theme ("Intelligent Operations & Industrial Efficiency"). Submission due Oct 18, 2026.

### Data
- **Machine data:** the public Microsoft Azure Predictive Maintenance dataset: hourly telemetry (volt, rotate, pressure, vibration) for 100 machines, plus errors, maintenance records (component replacements), failures (comp1–comp4) and machine metadata (model, age).
- **Business data is synthetic**, from a seeded generator kept in the repo:
  - `production_schedule`: machine_id, product, units_per_hour
  - `inventory`: product, on_hand, safety_stock
  - `orders`: order_id, customer, product, qty, due_date, penalty_per_day, priority
- **Demo bundles:** 2–3 prepared bundles, each a telemetry slice (about 10 machines × 30 days, well under the 32MB Cloud Run request limit) plus the related errors, maintenance and metadata, plus the synthetic business CSVs. In the demo, the business CSVs are uploaded alongside the telemetry. The UI marks synthetic inputs.

### Modules
1. **Data loading and validation:** parses the uploaded bundle, checks schemas and machine-ID consistency, and returns typed tables or field-level errors.
2. **Feature builder:** turns raw tables into model features per machine and hour: rolling 3h and 24h means and standard deviations of telemetry, rolling error counts, hours since each component was last replaced, and machine model and age. The same code runs at training and inference time.
3. **Failure model:** a BigQuery ML `BOOSTED_TREE_CLASSIFIER`, trained offline on the full dataset with a time-based train/test split. Target: which component fails within the next 24h (none / comp1 / comp2 / comp3 / comp4). The trained model is exported (XGBoost booster) and bundled in the container. Inference runs in-process, with no Vertex endpoint and no `ML.PREDICT` at runtime.
4. **Quality model:** a transparent formula. Defect rate = f(component age relative to typical life, vibration and pressure drift from the machine's own baseline, recent error frequency). It is projected over a horizon of N hours as age and drift grow. Output: defect-rate curve and scrap cost.
5. **Cascade engine:** a fixed dependency graph. If a component fails, the machine is down for a repair time from the assumptions. Lost units = units_per_hour × downtime, which drains inventory; once on_hand falls below demand, orders are delayed and incur penalty_per_day × days late. Quality loss adds scrap cost. A **Monte Carlo** run (fixed seed, about 1,000 samples) draws failure and timing from the predicted probabilities and returns the expected loss and P10/P50/P90.
6. **Intervention simulator:** four levers, each changing the inputs to the model and cascade before re-running them:
   - **reduce_speed(pct):** scales `rotate` down, lowering the predicted failure probability and throughput
   - **maintain_now:** planned downtime and labour cost now; resets the affected component's age
   - **replace_component(comp):** part plus labour cost; resets that component's age
   - **reroute(target_machine):** moves the schedule to a machine of the same model with spare capacity; that machine's own risk applies

   Each lever returns its cost, the expected loss with the lever applied, loss avoided, net benefit (loss avoided − action cost) and a disruption score. Levers are ranked by net benefit, with ties broken by lower disruption. A "do nothing" baseline is always included.
7. **Analysis engine (the main seam):** one entry point, `analyze(bundle, assumptions, seed, options) → AnalysisReport`. It runs validation → features → prediction → quality → cascade → simulation and returns one structured report. Both the UI and the agent tools call this, or the narrower functions behind it. It never calls Gemini.
8. **Agent:** built with Google ADK on **Gemini Flash** (AI Studio key during development, Vertex AI when deployed, switched by environment variable). Its tools are thin wrappers over the analysis engine:
   - `get_risk_overview`
   - `get_machine_risk(machine_id)`
   - `project_quality(machine_id, hours)`
   - `get_cascade(machine_id)`
   - `simulate_interventions(machine_id, levers?)`
   - `get_assumptions`

   System-prompt rules: only state numbers returned by tools, cite the tool behind each number, say when the data can't answer a question, and keep output to about 800 tokens or less. If ADK isn't working by end of Oct 13, fall back to plain Gemini function calling with the same tools.
9. **UI:** a single Streamlit app with these sections:
   - Upload (with a sample-file download)
   - Risk table
   - Machine detail (component probabilities, drivers, quality curve, cascade chain, loss range)
   - Interventions table
   - Editable plant assumptions
   - Recommendation panel
   - "Ask FactoryMind" chat
   - Replay controls (play, pause, reset, speed)
10. **Response cache:** agent responses are keyed by (bundle hash, assumptions hash, prompt), so demo bundles replay from cache.

### AnalysisReport shape (agreed contract between modules)
- `machines[]`: machine_id, model, age, `component_risk` {comp1..comp4: probability}, top_drivers[], risk_level
- `quality[machine_id]`: horizon_hours[], defect_rate[], scrap_cost
- `cascade[machine_id]`: downtime_hours, lost_units, inventory_breach_at, late_orders[] {order_id, customer, hours_late, penalty}, expected_loss, loss_p10/p50/p90
- `interventions[machine_id][]`: lever, params, cost, expected_loss_after, loss_avoided, net_benefit, disruption, rank
- `assumptions_used`, `seed`, `data_provenance` (real or synthetic for each input)

### Deployment and cost
- **Hosting:** one Cloud Run service with `min-instances=0`, `max-instances=2`, 1 vCPU and 1GiB memory.
- **Accounts and alerts:** a free-trial Google Cloud project with a $50 budget alert.
- **Avoid:** Vertex AI prediction endpoints, Vertex Agent Engine, AutoML, Cloud SQL and GKE.
- **Secrets:** the Gemini key and config come from environment variables or Secret Manager, never the repo.
- **Teardown:** tear everything down after judging (Nov 7, or Dec 4 if the team is a finalist).

## Testing Decisions

- **What makes a good test:** it checks external behaviour through the public seam, not internal functions. It uses small fixture bundles and a fixed seed, so the same inputs always give the same report. Tests must not depend on Gemini or the network, and should survive refactors of the engine's internals.
- **Seam 1 (main): the analysis engine.** Call `analyze()` with small fixture bundles and assert on the `AnalysisReport`:
  - Validation errors for malformed bundles (missing columns, unknown machine IDs, oversized input)
  - A fixture with a degrading comp2 ranks that machine and component highest
  - Quality: defect rate doesn't decrease as the horizon grows, and scrap cost goes up as component age goes up
  - Cascade: zero failure probability means zero loss. A known failure leads to known late orders and penalties (worked out by hand for a tiny fixture). P10 ≤ P50 ≤ P90. Identical reports for the same seed.
  - Interventions: replace_component lowers that component's risk. Raising the downtime cost assumption makes earlier maintenance rank higher. Do-nothing is always present. The ranking follows net benefit, with ties broken by disruption.
  - Provenance flags reflect which inputs were synthetic
- **Seam 2: the agent.** Run the agent with a **stub Gemini** that returns scripted tool calls and final text, then assert:
  - The expected tools get called for representative questions
  - Every number in the final recommendation appears in some tool output from that turn (the "numbers only from tools" guardrail, checked automatically)
  - The agent says it can't answer when tools return nothing relevant
- **Model quality (offline, not a unit test):** a training notebook or script reports per-component AUC, precision and recall on a held-out time period. The results go in the README.
- **Not automated:** the Streamlit UI and the Cloud Run deployment. These get a scripted manual smoke test on the deployed URL before the Oct 16 feature freeze: upload a demo bundle → risk table → replay → change an assumption → the recommendation changes → ask the chat a question.
- **Prior art:** none. The repo is new, and this spec sets the pattern.

## Out of Scope

- Real-time sensor ingestion, streaming, or connecting to real plant systems (MES/ERP/SCADA). Replay of uploaded data stands in for live data.
- Retraining the model on upload. Training is offline only.
- Physical hardware.
- Fleets beyond one upload bundle, multiple plants, user accounts, logins, roles or multi-tenancy.
- Gemini producing failure probabilities or loss numbers itself.
- AutoML, Vertex prediction endpoints and Agent Engine (cost).
- Automatically carrying out interventions (FactoryMind recommends; humans act).
- Mobile-specific UI and full accessibility auditing beyond Streamlit defaults.

## Further Notes

- **Judging weights:** Technical Merit & Gen AI Implementation 40%, Problem Alignment & Impact 25%, Innovation 25%, UX 10%. Gemini must *orchestrate* through tools, not just write summaries.
- **Submission:** deployed Cloud Run URL, demo video under 3 minutes (aim for 2:55 or less), **public** GitHub repo (the repo is private now and must be made public before submitting), and a PDF deck. All in English. Only new work is eligible.
- **Team split:**
  - **lpriya (@lpriya2760):** data, features, BigQuery ML model and quality formula
  - **Ramya (@ram-ctrl):** synthetic generator, cascade, Monte Carlo, simulator and ADK agent
  - **Yash:** Streamlit UI, Cloud Run deployment, README, deck and video
- **Schedule:**

  | Date | Milestone |
  |---|---|
  | Oct 11 | Tool spikes (Cloud Run hello, BigQuery ML sample, ADK hello); agree the `AnalysisReport` contract |
  | Oct 12–13 | Model, generator and cascade working offline; UI running on a mock report |
  | Oct 14 | Integrated and deployed |
  | Oct 15 | Simulator, Monte Carlo, replay, assumptions panel |
  | Oct 16 | Feature freeze |
  | Oct 17 | Video, deck, README |
  | Oct 18 | Submit by midday |

- **Order of cuts if behind:** Monte Carlo range → replay → custom business-data upload → chat panel. Never cut: prediction → cascade → simulated levers → Gemini recommendation, deployed.
- **Video outline:**
  - Problem (0:00–0:20)
  - Demo (0:20–2:15), including a live change to an assumption that changes the recommendation
  - Architecture (2:15–2:45)
  - Impact and scale (2:45–3:00)


