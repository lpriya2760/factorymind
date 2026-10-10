# ADR 0004: A Gemini agent that may only quote tool results

**Status:** accepted (October 2026, issue #9)
**Code:** `factorymind/agent.py`

## Context

The analysis engine already computes the answer: the risk, the cascade and the ranked actions. Gemini's job is to **orchestrate and explain**. It decides what to look at, calls the engine through tools, and turns the results into a recommendation a busy manager can act on. The risk is that a language model invents or "adjusts" numbers. A recommendation with one made-up dollar figure is worse than none.

## Decision

### An ADK agent with six tools over the analysis

Built with **Google ADK**. The tools are thin wrappers over the same `AnalysisReport` the UI shows:

| Tool | Returns |
|---|---|
| `get_risk_overview()` | every machine: risk level, riskiest part, failure chance, expected loss, best action |
| `get_machine_risk(machine_id)` | each component's failure chance and the reasons |
| `project_quality(machine_id, hours)` | defect rate now and at the horizon, scrap cost, and the inputs behind them |
| `get_cascade(machine_id)` | both scenarios: late orders, safety-stock breach, losses, expected loss |
| `simulate_interventions(machine_id)` | the ranked actions with cost, failure chance after, loss avoided and net benefit |
| `get_assumptions()` | the plant cost assumptions |

- **Numbers in quotable units.** Tools return figures in the form they should be quoted: whole dollars, percentages to one decimal, hours. So Gemini never has to convert anything.
- **Unknown machines.** An unknown machine returns an error plus the list of machines, rather than a guess.

### The number guardrail

The system prompt sets three rules:
1. Every number must come from a tool result.
2. Cite the tool after each figure.
3. Never calculate new numbers.

Prompts alone aren't a guarantee, so after Gemini answers, **`unverified_numbers` checks every number in the text** against the tool results from that conversation and the prompt:
- **Rounding is allowed:** 18176.0 can be quoted as $18,176, and 99.83 as 99.8%.
- **Identifiers aren't numbers:** comp2, model3, O-0001 and error2 are skipped.
- **If anything is unverified,** Gemini is told which numbers and asked to rewrite **once**.
- **If anything is still unverified after the rewrite,** the app shows a warning listing those figures. It never shows them silently.

In practice this has already fired. On one live run the first answer included a figure the tools hadn't returned, and the rewrite fixed it ("✅ checked … after one rewrite").

### Gemini Flash, with a fallback model

- **Models:** the default is `gemini-3.8-flash`. If a model is overloaded (503) or rate-limited (429), the agent moves to the next in `FACTORYMIND_FALLBACK_MODELS` (default `gemini-3.5-flash`). During development, 3.8-flash was overloaded several times while 3.5-flash answered, so a live demo can't depend on one model being up.
- **Thinking budget:** newer Flash models "think" before answering, and that thinking counts against `max_output_tokens`. A cap of 800 cut answers off after about 16 words. The cap is now 4,096 with `thinking_level = LOW`, and the prompt asks for under about 200 words.
- **Temperature 0.2:** consistent, factual answers.

### AI Studio locally, Vertex AI on Cloud Run, by configuration only

The code never chooses the provider. The `google-genai` client reads the environment:
- **Locally:** `GOOGLE_API_KEY` from the git-ignored `.env`, with `GOOGLE_GENAI_USE_VERTEXAI=FALSE`.
- **Cloud Run:** `scripts/deploy.sh` sets `GOOGLE_GENAI_USE_VERTEXAI=TRUE`, `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION=global`. No key is deployed, and Gemini is billed to the project's credits.

Both were checked with the full agent: the same tools, the same recommendation and no unverified numbers.

## Testing without Gemini

`tests/test_agent.py` replaces Gemini with a **scripted model** (a `BaseLlm` subclass that plays back tool calls and answers). It runs with sockets blocked and no key, and checks:
- which tools are called, and with what arguments
- that a good answer passes the guardrail
- that an invented "$99,999" triggers one rewrite, and is reported if it survives
- that an overloaded model falls back to the next
- that tool results are plain JSON
- that unknown machines get an error, not a guess

## Consequences

- **Good:**
  - Gemini decides *what* to look at and *how* to explain it, but every figure it states traces to the engine and is machine-checked.
  - The tool calls are shown in the app, so a judge can see the orchestration.
- **Limitation:** the guardrail checks *numbers*, not claims. "Beats the alternatives" is verified only to the extent the cited figures support it.
- **Limitation:** each answer takes 10–30s. Caching the answers for the demo bundles is #13.
- **Fallback:** the plan allowed falling back to plain Gemini function calling if ADK wasn't working by Oct 13. It wasn't needed.
