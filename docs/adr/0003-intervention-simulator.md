# ADR 0003: Interventions re-run the cascade with changed inputs

**Status:** accepted (October 2026, issue #8)
**Code:** `factorymind/interventions.py`

## Context

FactoryMind's promise is "the smallest action that prevents the largest loss". The Gemini agent (#9) will *explain* and *recommend*, but the numbers have to come from a simulation, not from the language model. So each action must be evaluated with the same machinery as the do-nothing case: the failure model (#4), the quality formula (#5) and the cascade (#6).

## Decision

Every action **changes the cascade's inputs and re-runs the same two-scenario simulation** (`cascade.run_scenarios`):
- the machine's output, hour by hour
- its failure chance
- its defect curve
- for rerouting, another machine's output

This gives `expected_loss_after`, and from it:

```
loss avoided = expected loss (do nothing) − expected loss after
net benefit  = loss avoided − cost of the action
```

Actions are ranked by net benefit, with ties going to the less disruptive one (none < slow down < replace a part < reroute < full service).

### One accounting rule for every action

Each hour of a machine's output that isn't made anywhere costs `downtime_cost_per_hour`. That covers:
- unplanned repair (8h)
- a planned stop (3h swap, 6h service)
- a slowdown (25% × 24h = 6 machine-hours)
- the share of output another machine can't cover during a reroute

Without a single rule, actions would be compared on inconsistent bases. For example, if slowing down were charged nothing for lost output, it would always look free.

### The actions

| Action | What changes | Failure chance after | Cost of the action |
|---|---|---|---|
| **Do nothing** | n/a | the model's prediction | $0 |
| **Replace the at-risk part** | planned stop of `planned_replacement_hours` (3h) now. The part's age resets, the sensor that part affects returns to the machine's normal level, recent errors clear. | **re-predicted by the model** on those inputs | part + labour + 3h of lost output |
| **Reroute while replacing** | as above, but the healthiest **same-model** machine covers the output during the stop, up to `spare_capacity_share` (50%) of its rate | re-predicted by the model | part + labour + `reroute_setup_cost` + the uncovered output |
| **Full service now** | planned stop of `full_service_hours` (6h); all four parts replaced; all sensors back to normal | re-predicted by the model | 4 parts + labour + 6h of lost output |
| **Slow down 25% for 24h** | 25% less output for the prediction window | **assumed**: 1 − (1 − p)^(0.75²) | 6 machine-hours of lost output |

### Why "back to normal" means resetting the sensor features

In this dataset each component's failures show up in one sensor: **comp1 → voltage, comp2 → rotation, comp3 → pressure, comp4 → vibration**. Machine 90's comp2 failure, for example, shows as rotation 16% below normal. If a replacement only reset the part's *age*, the model would still see the anomaly in the sensor features and keep predicting failure, even though the faulty part is gone. So a replacement also returns that part's sensor features (3h and 24h averages and variability) to the machine's own median levels and clears recent error counts. The **real model** then re-predicts the failure chance from those inputs. Machine 90 falls from 99.8% to 0.2%.

### Why slowing down uses an assumption instead of the model

The model learned that **low rotation is a symptom** of comp2 failing. Feed it a deliberately slower speed and it predicts a *higher* failure risk, which is the opposite of the physics. A correlational model can't answer "what if we slow down?", so this action uses an explicit, labelled assumption instead:

- Failures are treated as a constant hazard over the 24h window, and stress is assumed to scale with speed².
- So `p_after = 1 − (1 − p)^(0.75²)`. For a near-certain failure (p = 0.999) that's still 98%, because a part about to break will break. For p = 0.3 it's 18%.
- The app and the report mark this as **"assumed"** (`effect_basis = "assumption"`).

### Choosing the reroute target

The target is the healthiest same-model machine making the same product: lowest failure chance, then lowest current defect rate. In the demo data, machines of the same model make the same product, so the target can really take the work.

## Result on the demo data

**Machine 90** (comp2, >99% failure chance; doing nothing: $24,412 expected loss):

| # | Action | Cost | Failure chance after | Loss avoided | Net benefit |
|---|---|---|---|---|---|
| 1 | Replace comp2, machine 47 covers | $6,074 | <1% (model) | $24,249 | **$18,176** |
| 2 | Replace comp2 now | $8,970 | <1% (model) | $24,260 | $15,290 |
| 3 | Full service now | $20,340 | <1% (model) | $24,369 | $4,029 |
| 4 | Do nothing | $0 | >99% | $0 | $0 |
| 5 | Slow down 25% for 24h | $15,000 | 98% (assumed) | −$1,951 | −$16,951 |

- **Healthy machines:** "do nothing" ranks first.
- **Machine 72:** its failure risk is low, but its pressure runs 24% above normal. Replacing comp3 (the pressure-linked part) ranks first, because it removes about $11.6k of projected excess scrap. That's a recommendation failure risk alone wouldn't give.

## A refinement to the ticket's acceptance criteria

The ticket asked that "raising the downtime cost moves `maintain_now` up the ranking". That holds for a machine **likely to fail**: dearer downtime makes the avoided 8h failure worth more than the 6h planned stop costs. For a healthy machine it does the opposite, because dearer downtime only makes the planned stop more expensive. The test checks the case where it holds: for machine 90, full service ranks *below* doing nothing at $0/h and *above* it at $10,000/h.

## Consequences

- **Good:**
  - Every recommendation traces to the same simulation as the baseline.
  - The failure chance after each action comes from the real model wherever the model can answer, and is labelled "assumed" where it can't.
  - Changing a plant assumption (#7) changes the ranking, so a judge can see the trade-offs.
- **Limitation:** a single fixed action per lever: 25% for 24h, the at-risk part, one reroute target. A finer search (other speeds, other parts, combinations) would fit easily in the same structure.
- **Limitation:** the component–sensor link is specific to this dataset. A real plant would supply its own mapping, or learn it.
- **Limitation:** the reroute target's own failure risk is ignored during the short 3h cover window.
