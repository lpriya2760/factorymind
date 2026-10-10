# ADR 0002: The cascade is an hour-by-hour simulation of two scenarios

**Status:** accepted (October 2026, issue #6)
**Code:** `factorymind/cascade.py`

## Context

The failure model (#4) says *how likely* a machine is to fail, and the quality projection (#5) says *how its output degrades*. Neither says what that means for the business. The cascade has to turn them into the chain an operator cares about:

> machine fails → output lost → inventory falls below safety stock → customer orders late → money lost

The intervention simulator (#8) will re-run this chain under different actions, and the Gemini agent (#9) will quote its numbers. So it must be **deterministic, explainable, and cheap enough to re-run many times**.

## Decision

Simulate the machine's product **hour by hour for 168h** (one week from the end of the uploaded data), in two scenarios, then weight them by the failure probability.

### 1. Two scenarios, combined by expected value

- **If it fails:** the machine stops at hour 12 for `repair_hours` (8h by default). After the repair it runs at the normal 0.5% defect rate.
- **If it keeps running:** it produces all week at its projected defect rate.

```
expected loss = p × loss if it fails + (1 − p) × loss if it keeps running
```

`p` is the probability that *any* component fails in the next 24h (1 − P(none) from the model).

**Why not only the failure scenario?** A machine can lose money without failing. Machine 72 in `comp2_failure_building` has a low failure risk, but its 24% pressure drift makes it the second-costliest machine through scrap alone. Showing both scenarios also lets the operator see what a failure *adds*.

**Why expected value rather than worst case?** It ranks machines sensibly: a 0.1%-likely failure shouldn't outrank a 99.9%-likely one. It's also what the intervention ranking (#8) needs to compare "loss avoided" across actions. The *range* of outcomes comes from the Monte Carlo simulation in #11.

### 2. Failure at hour 12

The model predicts a failure *somewhere* in the next 24h without saying when. Assuming it's equally likely at any hour, the expected time is hour 12. A single deterministic time keeps the chain explainable ("down from about hour 12"). #11 will sample the timing instead.

### 3. Repair restores normal quality

A failed component gets replaced, so the wear and drift behind the projected defect rate go away. This matters. For machine 90, the repair *recovers* output that its degrading quality would otherwise lose, and its expected scrap falls from $10,890 to $510. It also explains a result that looks odd at first: machine 47, which makes the same product and is healthy, would cause **more** late orders if it failed (4) than machine 90 would (1). Machine 47's repair doesn't fix anything, while machine 90's repair fixes its quality problem.

### 4. Inventory and orders

- **Production:** every machine making the product adds its **good** output each hour: units per hour × (1 − its defect rate). This is how quality feeds into inventory.
- **Orders:** they ship **first in, first out by due date** (then priority) as soon as there's enough stock. FIFO is simple, common and easy to explain. Optimising which customer to delay is the kind of decision #8 and #9 can recommend, not something the baseline should hide.
- **Safety stock can be used.** That's what it's for.
- **Safety stock breach** is measured on the **inventory position**: stock minus orders that are due but not yet shipped. An early version measured stock *after* shipping. A delayed order then kept stock artificially high, so a failure seemed to breach safety stock *later* (33h) than normal running (30h). The inventory position counts a delayed order against stock straight away.
- **Penalties:** penalty_per_day × days late (part days count as a whole day, the common contract convention) × penalty_multiplier. Orders still unshipped at 168h count as late up to that point.

### 5. What counts as loss

| | If it fails | If it keeps running |
|---|---|---|
| Downtime: `downtime_cost_per_hour × repair_hours` | ✓ | |
| Repair: part + labour × repair_hours | ✓ | |
| Late-order penalties for the machine's product | ✓ | ✓ |
| **Excess** scrap: this machine's scrap *above* the normal 0.5% rate | ✓ (until it fails) | ✓ (all week) |

Only **excess** scrap counts, because normal scrap isn't caused by anything. Without this, every healthy machine would show a large "loss".

## The demo data had to change

The cascade exposed two problems in earlier work. Both were fixed with your approval as part of #6.

1. **Orders were too loosely sized.** The generator's docstring promised that "losing a machine for a shift makes some orders late", but machine 90's failure caused no late orders at all. The data now uses:
   - **Safety stock:** 2–4 hours of production.
   - **Order sizes:** by each due date, demand uses the **stock above safety level plus 90–95%** of what the machines can make.

   Normal running then ships on time and keeps safety stock intact, while losing a machine for a shift eats into it and can make orders late. Only `inventory.csv` and `orders.csv` changed in the demo bundles. The seed, due dates, penalties, machine data, risk and quality are all the same.
2. **The quality projection extended noise over a week.** It now projects the drift trend for at most 48h (see ADR 0001).

## Result on the demo data

**`comp2_failure_building`, machine 90** (failure chance >99%):

| | If it fails | If it keeps running |
|---|---|---|
| Units not made | 297 | n/a |
| P-200 below safety stock | at 30h | at 54h (degrading quality) |
| Late orders | 1 (O-0001, Everline Packaging) | 0 |
| Penalties | $1,990 | $0 |
| Excess scrap | $510 | $10,890 |
| Downtime + repair | $20,000 + $1,920 | $0 |
| **Total** | **$24,420** | **$10,890** |

**Expected loss: $24,412**, the largest in the bundle. In `healthy_fleet`, no machine's product falls below safety stock or ships late in normal running.

## Consequences

- **Good:** every number traces to an hour-by-hour simulation with stated rules, and two hand-calculated examples in the tests pin the arithmetic. It runs in milliseconds, so #8 can re-run it for each intervention.
- **Limitation:** one failure time (hour 12) and one repair duration. #11 adds the spread.
- **Limitation:** the machine's other failures, other machines failing at the same time, and rerouting aren't modelled here. Rerouting is an intervention in #8.
- **Limitation:** downtime cost per hour is a flat plant assumption ($2,500) and dominates the failure loss. #7 makes it editable, so a judge can see how the conclusion depends on it.
