# ADR 0005: The loss range comes from Monte Carlo draws of whether and when a machine fails

**Status:** accepted (October 2026, issue #11)
**Code:** `loss_range()` in `factorymind/cascade.py`

## Context

The cascade (ADR 0002) gives one expected loss per machine, from one assumed failure time (hour 12). An operator also needs to know how far the actual week could land from that figure: "$24k expected" reads very differently with a range of $22k–$25k than with $1k–$45k.

## Decision

Run **1,000 seeded Monte Carlo draws** per machine and report the **P10, P50 and P90** of the week's loss alongside the expected loss.

1. **What is drawn:** whether the machine fails, with the model's predicted probability, and if it does, in which hour of the 24h prediction window, each hour equally likely. Everything else (repair time, costs, orders) is as in the expected-value cascade, so the range shows the uncertainty the model actually reports and adds no new assumptions.
2. **Percentiles are actual outcomes.** They use the inverted-CDF method, so each one is a loss that occurs in some draw. Linear interpolation would invent figures between "fails" and "keeps running" that no week ever produces.
3. **Reproducible:** the generator is seeded with `(analyze seed, machine_id)`. The same data and seed always give the same report.
4. **The expected loss is unchanged.** Interventions are still ranked on it (ADR 0003), and Gemini's tools and cached answers don't change.

## Keeping it fast

A failure's loss depends only on its hour, so each drawn hour is simulated **once** (at most 24 runs per machine), and the 1,000 draws pick from those results.

When fewer than 10% of draws fail, P10, P50 and P90 are all the keeps-running loss, whichever way the failure losses fall, so no failure is simulated at all. This is exact, not an approximation, and it covers most machines.

On the demo bundles the whole analysis went from about 0.5s to 0.6–0.75s (2 vCPU); without these two shortcuts it took about 1.7s.

## Result on the demo data

| Machine | Failure chance | Expected loss | P10 | P50 | P90 |
|---|---|---|---|---|---|
| 90 (`comp2_failure_building`) | >99% | $24,412 | $21,999 | $24,374 | $24,862 |
| 92 (`two_machines_at_risk`) | >99% | $27,936 | $27,322 | $27,874 | $28,575 |
| 44 (`comp2_failure_building`) | 1% | $1,405 | $1,203 | $1,203 | $1,203 |

For machine 90, an early failure costs less ($22k): the repair stops its scrap sooner. A late one costs more ($24.9k) because degrading quality runs longer first.

For machine 44, the expected loss sits **above** P90. In at least 9 of 10 weeks it keeps running ($1,203), while the 1% chance of a $21,941 failure lifts the expected value. The UI says this in words rather than showing a misleading "$1,203–$1,203" alone.

## Consequences

- **Good:** the range comes straight from the model's own probability, so it adds no new assumptions. It is reproducible and adds about 0.1–0.2s.
- **Limitation:** repair time, costs and other machines failing are not varied. Drawing those would need distributions the data doesn't give us.
- **Not yet:** ranges for each intervention. After an intervention the failure chance is usually near zero, so its range would collapse to one figure and add little. Gemini's tools don't return the range yet. Adding it means regenerating the cached demo answers.
