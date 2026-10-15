# Smoke test: the deployed app

Run this on **https://factorymind-929560698773.asia-southeast1.run.app** after every deploy, and before recording the demo. It takes about 5 minutes. The expected values are for the demo bundles with default plant assumptions.

| # | Do | Expect |
|---|---|---|
| 1 | Open the URL (the first load after idle can take ~10s) | Title "FactoryMind"; the sidebar shows **Plant assumptions** |
| 2 | Upload `demo_bundles/comp2_failure_building.zip` | Banner "Synthetic inputs: production_schedule, inventory, orders"; **10 machines** |
| 3 | Risk table | Red banner **"Machine 90: comp2 is >99% likely to fail in the next 24h"**; machine 90 first, 🔴 High, comp2 99.8%; everything else 🟢 Low |
| 4 | Machine detail (machine 90 is selected) → **Quality** | 7.5% now → **12.9% in 48h**, scrap **$2,746**. Drag *Hours ahead* to 168: 13.0% |
| 5 | **What a failure would cause** | >99% · 297 units not made · P-200 below safety stock at 30h · **1 late order (O-0001), $1,990** · expected loss **$24,412**; *synthetic data* badges on inventory and orders |
| 6 | **What to do about it** | Best: **Replace comp2, machine 47 covers**, net benefit **$18,176**; slowing down is last, marked "(assumed)" |
| 7 | **Ask Gemini what to do** | Answer in about a second, starting with the replace-comp2 recommendation, figures cited like `[simulate_interventions]`; caption **"✅ Every figure was checked … ⚡ cached answer, no new Gemini call"**; *Tools Gemini used* lists the calls |
| 8 | Sidebar: set **Downtime cost** to **300** | Expected loss **$6,822**; best action changes to **Replace comp2 now** (net $4,327); the sidebar says "Changed from default: downtime cost 300 (default 2500)" |
| 9 | **Ask Gemini what to do** again | A cached answer for the $300 scenario (⚡) recommending *replace comp2 now* |
| 10 | **Reset to defaults** | Back to $24,412 and *Replace comp2, machine 47 covers* |
| 11 | Upload `healthy_fleet.zip` instead | No red banner; all 🟢 Low; best action **Do nothing** with a reason |
| 12 | Upload a broken file (e.g. rename `orders.csv` to `orders.txt`) | A table of problems instead of a crash |

## Checking that cached answers don't call Gemini

In the Cloud Run logs, each **Ask Gemini** shows one of these lines:

```bash
gcloud logging read 'resource.labels.service_name="factorymind" AND textPayload:"recommendation for machine"' --limit 20 --format "value(timestamp,textPayload)"
```

- `cache hit (no Gemini call)`: served from the shipped or in-memory cache
- `Gemini call to <model> took <n>s`: a live call (expected for machines or assumptions that aren't cached)

## If something's off

| Symptom | Likely cause |
|---|---|
| The Gemini answer takes 10–30s and has no ⚡ | The prompts, analysis or demo data changed since the cache was built. Re-run `scripts/cache_recommendations.py`; a test fails until you do. |
| "Gemini couldn't answer right now" | Both models are overloaded or the Vertex AI permission is missing. Check the logs. |
| Numbers differ from this page | The demo data or a formula changed. Update this page along with the README and ADRs. |
