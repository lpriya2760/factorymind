# ADR 0001: Quality projection uses a transparent formula, not a learned model

**Status:** accepted (October 2026, issue #5)
**Code:** `factorymind/quality.py`

## Context

FactoryMind has to answer: *"If this machine keeps running for the next N hours, what happens to product quality, and what does that cost?"* The cascade (#6) and the intervention simulator (#7) both build on that answer.

The obvious approach is to learn it from data, but **no public machine dataset records defects**. The Azure Predictive Maintenance dataset we use has sensor readings, errors, maintenance and failures, but nothing about the parts the machines produce. That leaves nothing to train a quality model against or test it on.

So the question was not "which model?" but "what can we defend with no labels?"

## Options considered

| Option | Why not (or why) |
|---|---|
| **Train a defect model** | Impossible: there are no defect labels. |
| **Use the failure probability as a quality proxy** | It mixes up two different things. The failure model looks only 24h ahead, and is near 0 until a failure is imminent. Quality problems from wear or drift build up over days or weeks without any failure, as with machine 72 below. The cascade (#6) already accounts for failure. |
| **Physics-based model of the process** | Needs knowledge of the product and process that we don't have for simulated machines. |
| **Hidden heuristic** | Numbers a judge or operator can't check undermine trust in the whole product. |
| **Transparent formula with every parameter shown** ✅ | Uses only signals we really have, behaves predictably, and can be shown in full in the app and README. It's easy to replace with a fitted model once a plant has defect records. |

## Decision

```
defect(h) = min(MAX_RATE,
                BASE_RATE
                + WEAR_WEIGHT  × min(wear(h), WEAR_CAP)²
                + DRIFT_WEIGHT × min(max(drift(h) − DRIFT_DEADBAND, 0), DRIFT_CAP)
                + ERROR_WEIGHT × errors in the last 24h)

scrap cost = Σ over hours of units_per_hour × defect(h) × scrap_cost_per_unit
```

The formula adds up **three independent ways a machine makes bad parts** on top of a healthy baseline. Each one uses a signal that's in the data.

### 1. Base rate: `BASE_RATE = 0.5%`

Even a healthy machine makes some defective parts. The formula needs a floor so a "good" machine still has a non-zero scrap cost, and so the other terms show up as *additions* to normal. The value 0.5% is a **judgment call**: low enough to read as healthy, and far enough below a degrading machine (around 10%) that the difference is obvious. It isn't derived from real defect statistics.

### 2. Component wear: `WEAR_WEIGHT × min(wear, WEAR_CAP)²`

**Why wear matters.** Worn parts such as bearings, tooling and seals hold tolerances less well, so defects rise as components age.

**Signal used.** Hours since each component was last replaced, from the maintenance records. That's the same `hours_since_compN` feature the failure model uses.

**Why the *oldest* component.** One worn part is enough to spoil output. Averaging across components would hide it.

**Why squared.** In the classic "bathtub curve", failure and wear-out effects are small for most of a part's life, then accelerate near the end. A straight line would give a new part almost the same penalty as a half-worn one. The square keeps young parts near 0 and makes old parts count.

**Why a cap (`WEAR_CAP = 2` lifetimes).** Without one, a machine with no replacement on record (we treat that as one year) or a very old part would dominate the projection.

**How `TYPICAL_LIFE_H` and `WEAR_WEIGHT` were tuned.** This was the one term I had to change after looking at the numbers.

- The **first version** used a 90-day life and a 2% weight. Measured on the demo bundles, ordinary machines with old components came out at **7.7–9.0% defects**, almost the same as machine 90, which really is degrading at 9.3%. The wear term swamped the signal.
- **Why.** The average component is replaced about every **58 days** (the training-data mean of `hours_since_compN`). But a machine has four components, and the *oldest* is usually much older: **36–232 days** across the demo bundles, typically around 90–120. A 90-day "life" meant most machines were permanently past it.
- **Fix:** a **180-day life** (about the age of the oldest component on an old-but-normal machine) and a **1% weight**.
  - A component at one full life adds 1%. At the cap of two lives it adds 4%.
  - A 60-day-old part adds about 0.1%.
  - With the final 4% drift deadband, the healthy fleet's median is about 1%, and degrading machines stand out.

### 3. Sensor drift: `DRIFT_WEIGHT × min(max(drift − DEADBAND, 0), DRIFT_CAP)`

**Why drift matters.** A machine running far from its normal operating point is making parts under conditions it wasn't set up for: wrong speed, pressure or vibration. This is the strongest *current* signal of a quality problem we have.

**How drift is measured.**

`drift = max over sensors of |24h average − this machine's normal level| ÷ normal level`

- **Compared with the machine's own normal, not the fleet's.** Machines differ in their normal levels, and an absolute threshold would flag some permanently. "Normal" is the machine's median 24h average across the uploaded data, typically 30 days. Using the median means a short anomaly doesn't shift the baseline.
- **Relative (%), not standard deviations.** "Rotation is 16% below normal" is something an operator understands. Standard deviations of 24h averages turned out to be tiny, which made z-scores explode for even modest changes.
- **The single worst sensor.** As with wear, one bad parameter is enough to spoil output.
- **All four sensors.** The spec said vibration and pressure, but in this dataset comp2 problems show up in **rotation**. Machine 90's rotation runs at 378 against a normal of about 450, 15.8% low. Leaving rotation out would have meant the main demo machine's quality didn't change at all.

**Deadband: `DRIFT_DEADBAND = 4%`.** Sensors are never perfectly steady, and small gaps are noise.

- **First choice: 2%,** made up front as a judgment call.
- **What the data showed:** measured afterwards on `healthy_fleet`, normal machines drift **1.4–6.9%**, median about 3%. So 2% sat slightly *inside* normal noise, and ordinary machines picked up up to about 2.5 points of defects from this term.
- **Change:** raised to **4%**, just above the typical normal drift. Only the noisiest normal machines (drift up to about 7%) now get anything from this term, at most about 1.5 points.
- **Effect:** the `healthy_fleet` median fell from 1.8% to 0.9%, and the comp2 bundle's fleet median from 2.3% to 1.3%. Machine 90 went from 8.5% to 7.5% now, so it stands out more against the fleet.

**Weight: `DRIFT_WEIGHT = 0.5`** (half a point of defect rate per point of drift beyond the deadband). This was calibrated against the demo bundles so a machine drifting about 16% (like machine 90) gains about 6 points. A drift of around 24% (machine 72's pressure) gains about 10 points. Those are clearly abnormal levels, but not absurd. `DRIFT_CAP = 50%` stops extreme readings from running away.

**Projecting drift forward.** `drift(h) = drift now + growth per hour × h`

- **Growth** is how much the gap widened over the last 24h, extended in a straight line. A straight line is the simplest projection we can defend from one day of trend. Anything more elaborate would claim more than one day of data supports.
- **Growth is never negative.** An improving machine keeps its current drift rather than being projected to recover. That's deliberately pessimistic: we project "if it keeps running unattended", not "if it fixes itself". It also guarantees the defect curve never goes down, which the ticket requires.

### 4. Recent errors: `ERROR_WEIGHT × errors in the last 24h`, with `ERROR_WEIGHT = 0.5%`

Error events, such as the `error2` and `error3` codes machine 90 logs before its comp2 failure, show the machine is misbehaving *now*. Each error in the last 24h adds half a point. Errors are rare (about 0.02 per machine per 24h on average), so this term matters only when something is wrong. It's held constant over the horizon because the data gives no basis for projecting error rates. The weight is a **judgment call**: an error is evidence of trouble, but it shouldn't outweigh a measured drift.

### 5. Ceiling: `MAX_RATE = 50%`

Extending trends in a straight line over a week can give silly numbers. Capping at 50% keeps the 168h projection meaningful ("half the output is scrap, stop the line") rather than reaching 100%+.

### Scrap cost

`Σ over hours of units_per_hour × defect(h) × scrap_cost_per_unit`

This is a direct cost calculation. Units per hour comes from the production schedule. The scrap cost per unit ($15 by default) is a plant assumption that #7 makes editable, so a judge can see the cost react live.

## How the result looks

On the demo bundles, at 48h:

| Machine | Situation | Now | 48h | Scrap (48h) |
|---|---|---|---|---|
| 90 (`comp2_failure_building`) | comp2 failing; rotation −15.8% and widening; 2 errors | 7.5% | 12.9% | $2,746 |
| 72 (same bundle) | pressure +23.7%, *no* failure for 108 days | 10.6% | 12.4% | $3,349 |
| fleet median, same bundle | normal | 1.3% | 2.5% | n/a |
| `healthy_fleet` worst machine | normal | 2.0% | 6.1% | $1,134 |

Machine 72 is the interesting case. The failure model correctly says it's at low risk, but its quality is degrading. That's exactly why quality is a separate projection rather than being derived from failure risk.

**`healthy_fleet` was regenerated as part of this work.** The original "healthy" bundle only excluded failures in the next 24h. It included machine 70 with vibration 27% above normal, which projected about 20% defects. It was "healthy" by the failure model's definition, but not a calm baseline. The bundle now excludes machines that fail within 14 days.

## Consequences

- **Good:** every number in the app can be traced to an input and a parameter, and the app shows the formula. The behaviour is predictable: the curve never decreases, stays within bounds, and older parts mean more scrap. The tests check all of this.
- **Limitation:** the parameters are **calibrated by judgment against simulated data, not fitted to real defects**. The *ranking* of machines (which ones are degrading) is well supported by real sensor and maintenance signals. The *absolute* percentages are illustrative.
- **Limitation:** the terms are added together. In reality, wear and drift probably make each other worse.
- **Limitation:** the straight-line drift projection is crude beyond a day or two. That's why the horizon is capped at 168h and the defect rate at 50%.

## How to replace it with real data

With defect or scrap records from a real plant:
1. Keep the same inputs (component age, drift, drift growth, errors). They're already computed and stored in `QualityProjection`.
2. Fit the weights, for example with a regression from those inputs to the observed defect rate, or train a model (e.g. BigQuery ML) to replace the formula.
3. `project_quality()` returns the same `QualityProjection`, so the cascade, the simulator, the UI and the agent are unaffected.
