"""The Gemini agent: recommends what to do about a machine, using the analysis engine as tools.

Built with Google ADK. The model is Gemini Flash (FACTORYMIND_MODEL to override). Locally it uses
the AI Studio key in GOOGLE_API_KEY; on Cloud Run, GOOGLE_GENAI_USE_VERTEXAI=TRUE with
GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION switches it to Vertex AI. No code changes.

Every number in the agent's answer must come from a tool result. `unverified_numbers` checks
this; `recommend` asks Gemini to rewrite once if it fails, and reports anything still unverified.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import time

import pandas as pd
from dataclasses import asdict, dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any, Callable, Mapping

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import errors, types

from factorymind.assumptions import resolve_assumptions
from factorymind.bundle import bundle_fingerprint, load_bundle
from factorymind.engine import DEFAULT_QUALITY_HORIZON_H, analyze
from factorymind.report import AnalysisReport

DEFAULT_MODEL = "gemini-3.8-flash"
# Tried in order when the model is overloaded (503) or rate-limited (429).
DEFAULT_FALLBACK_MODELS = "gemini-3.5-flash"
APP_NAME = "factorymind"

INSTRUCTION = """\
You are FactoryMind's maintenance advisor for a plant operations manager.
Use the tools to look at the data before answering. Never guess.

Rules:
1. Every number you write must appear in a tool result from this conversation. You may round it
   and format it: money as $18,176 (from 18176), shares as 99.8% (from 99.8). Never calculate new
   numbers: no sums, differences, ratios or percentages of your own.
2. Cite the tool in square brackets after each figure, e.g. "net benefit $18,176
   [simulate_interventions]".
3. If the tools can't answer part of the question, say so plainly.
4. Plain language for a busy manager, under about 200 words. No markdown headings.
"""

RECOMMEND_PROMPT = (
    "Recommend what to do about machine {machine_id} right now. Name the action, when to do it "
    "and the money it saves; explain briefly why it beats the alternatives and what risk "
    "remains."
)

CANT_ANSWER = "I can't answer that from the available data"

CHAT_INSTRUCTION = f"""\
You are FactoryMind's maintenance advisor, answering a plant operations manager's follow-up
questions about the uploaded machines. Use the tools to look at the data before answering. Never
guess.

Rules:
1. Every number you write must appear in a tool result from this conversation. You may round it
   and format it: money as $18,176 (from 18176), shares as 99.8% (from 99.8). Never calculate new
   numbers: no sums, differences, ratios or percentages of your own.
2. Cite the tool in square brackets after each figure, e.g. "net benefit $18,176
   [simulate_interventions]".
3. If the tools can't answer the question (for example a scenario they don't simulate), start
   with "{CANT_ANSWER}", say what's missing, then give the closest thing the tools do show.
4. Short, plain answers: under about 120 words unless asked for detail. No markdown headings.
"""

RETRY_PROMPT = (
    "These numbers in your answer don't appear in any tool result: {numbers}. Rewrite the answer "
    "using only numbers that appear in tool results, cited and formatted as before (money as "
    "$8,970, shares as 12.5%)."
)


# --- tools ------------------------------------------------------------------------------------

class AnalysisSession:
    """One uploaded bundle with its plant assumptions; analyses are cached by horizon."""

    def __init__(self, files: Mapping[str, bytes], assumptions: Mapping[str, float] | None = None):
        self.files = dict(files)
        self.assumptions = dict(assumptions or {})
        self._reports: dict[int, AnalysisReport] = {}

    def tables(self) -> dict:
        if not hasattr(self, "_tables"):
            self._tables = load_bundle(self.files).tables
        return self._tables

    def report(self, horizon: int = DEFAULT_QUALITY_HORIZON_H) -> AnalysisReport:
        if horizon not in self._reports:
            self._reports[horizon] = analyze(self.files, self.assumptions,
                                             quality_horizon_hours=horizon)
        return self._reports[horizon]


def _pct(p: float) -> float:
    return round(100 * p, 1)


def _usd(x: float) -> int:
    return int(round(x))


def make_tools(session: AnalysisSession) -> list[Callable[..., dict]]:
    """The agent's tools. Results are plain dicts with numbers in the units to quote them in."""

    def machine_or_error(report: AnalysisReport, machine_id: int):
        for m in report.machines:
            if m.machine_id == machine_id:
                return m, None
        known = sorted(m.machine_id for m in report.machines)
        return None, {"error": f"No machine {machine_id} in the uploaded data.",
                      "machines": known}

    def get_risk_overview() -> dict:
        """Every machine's failure risk in the next 24h, expected loss and best action,
        riskiest first."""
        r = session.report()
        return {"machines": [
            {"machine_id": m.machine_id, "model": m.model, "risk_level": m.risk_level,
             "riskiest_component": m.top_component,
             "failure_chance_pct": _pct(sum(m.component_risk.values())),
             "expected_loss_usd": _usd(r.cascade[m.machine_id].expected_loss),
             "best_action": r.interventions[m.machine_id][0].label}
            for m in r.machines]}

    def get_machine_risk(machine_id: int) -> dict:
        """One machine's chance of each component failing in the next 24h, and why."""
        m, error = machine_or_error(session.report(), machine_id)
        if error:
            return error
        return {"machine_id": m.machine_id, "model": m.model, "age_years": m.age,
                "risk_level": m.risk_level, "riskiest_component": m.top_component,
                "component_failure_chance_pct": {c: _pct(p) for c, p in m.component_risk.items()},
                "drivers": [d.description for d in m.top_drivers]}

    def project_quality(machine_id: int, hours: int = DEFAULT_QUALITY_HORIZON_H) -> dict:
        """One machine's projected defect rate and scrap cost if it keeps running for `hours`
        (1 to 168)."""
        hours = max(1, min(168, int(hours)))
        r = session.report(hours)
        m, error = machine_or_error(r, machine_id)
        if error:
            return error
        q = r.quality[machine_id]
        return {"machine_id": machine_id, "horizon_hours": hours,
                "defect_rate_now_pct": _pct(q.defect_rate[0]),
                "defect_rate_at_horizon_pct": _pct(q.defect_rate[-1]),
                "scrap_cost_usd": _usd(q.scrap_cost),
                "oldest_component_age_days": int(round(q.component_age_hours / 24)),
                "drift_sensor": q.drift_sensor, "drift_pct": _pct(q.drift),
                "drift_growth_pct_per_day": _pct(q.drift_growth_per_hour * 24),
                "errors_last_24h": q.errors_24h, "units_per_hour": round(q.units_per_hour, 1)}

    def get_cascade(machine_id: int) -> dict:
        """What one machine failing would cause over the next week (inventory, late orders,
        losses), compared with it keeping running."""
        r = session.report()
        m, error = machine_or_error(r, machine_id)
        if error:
            return error
        c = r.cascade[machine_id]

        def scenario(s):
            return {"late_order_count": len(s.late_orders),
                    "late_orders": [{"order_id": o.order_id, "customer": o.customer,
                                     "hours_late": round(o.hours_late, 1),
                                     "penalty_usd": _usd(o.penalty), "shipped": o.shipped}
                                    for o in s.late_orders],
                    "safety_stock_breached_at_hour": s.inventory_breach_hours,
                    "penalties_usd": _usd(s.penalties),
                    "excess_scrap_usd": _usd(s.excess_scrap_cost),
                    "downtime_cost_usd": _usd(s.downtime_cost),
                    "repair_cost_usd": _usd(s.repair_cost),
                    "total_loss_usd": _usd(s.total_loss)}

        return {"machine_id": machine_id, "product": c.product,
                "failure_chance_pct": _pct(c.failure_probability),
                "failure_at_hour": c.failure_at_hours, "downtime_hours": c.downtime_hours,
                "units_not_made": round(c.lost_units),
                "if_it_fails": scenario(c.if_failure),
                "if_it_keeps_running": scenario(c.if_no_failure),
                "expected_loss_usd": _usd(c.expected_loss)}

    def simulate_interventions(machine_id: int) -> dict:
        """Every possible action for one machine, simulated and ranked by net benefit
        (loss avoided minus the action's cost)."""
        r = session.report()
        m, error = machine_or_error(r, machine_id)
        if error:
            return error
        return {"machine_id": machine_id, "options": [
            {"rank": o.rank, "action": o.label, "cost_usd": _usd(o.cost),
             "failure_chance_after_pct": _pct(o.failure_probability_after),
             "effect_basis": o.effect_basis,
             "expected_loss_after_usd": _usd(o.expected_loss_after),
             "loss_avoided_usd": _usd(o.loss_avoided), "net_benefit_usd": _usd(o.net_benefit),
             "details": o.params, "note": o.note}
            for o in r.interventions[machine_id]]}

    def get_assumptions() -> dict:
        """The plant cost assumptions behind every money figure."""
        return dict(session.report().assumptions_used)

    def get_production_plan(machine_id: int) -> dict:
        """The production plan around one machine: every machine making the same product and
        its scheduled rate, the product's stock and safety stock, and its open orders. Machines
        can run above their scheduled rate (by the spare capacity share) to cover another."""
        r = session.report()
        m, error = machine_or_error(r, machine_id)
        if error:
            return error
        t = session.tables()
        schedule, product = t["production_schedule"], r.cascade[machine_id].product
        if product is None:
            return {"machine_id": machine_id, "product": None,
                    "note": "This machine has nothing scheduled in the uploaded plan."}
        as_of = t["telemetry"]["datetime"].max()
        levels = {x.machine_id: x.risk_level for x in r.machines}
        stock = t["inventory"][t["inventory"]["product"] == product]
        orders = t["orders"][t["orders"]["product"] == product].sort_values("due_date")
        return {
            "machine_id": machine_id, "product": product,
            "machines_making_it": [
                {"machine_id": int(row.machine_id), "scheduled_units_per_hour":
                 round(float(row.units_per_hour), 1), "risk_level": levels.get(int(row.machine_id))}
                for row in schedule[schedule["product"] == product].itertuples()],
            "spare_capacity_share": r.assumptions_used["spare_capacity_share"],
            "on_hand_units": int(stock["on_hand"].sum()),
            "safety_stock_units": int(stock["safety_stock"].sum()),
            "open_orders": [
                {"order_id": o.order_id, "customer": o.customer, "units": int(o.qty),
                 "due_in_hours": round((o.due_date - as_of) / pd.Timedelta(hours=1), 1),
                 "penalty_per_day_usd": _usd(o.penalty_per_day), "priority": o.priority}
                for o in orders.itertuples()],
        }

    return [get_risk_overview, get_machine_risk, project_quality, get_cascade,
            simulate_interventions, get_assumptions, get_production_plan]


# --- the number guardrail ---------------------------------------------------------------------

# A number not glued to a letter, digit, hyphen or dot on the left (so comp2, O-0001 and
# model3 are identifiers, not numbers).
_NUMBER = re.compile(r"(?<![A-Za-z0-9\-.])\$?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


# In tool results, a minus sign is part of the number (net benefit -8502 may be quoted as
# "−$8,502"); its size is what counts.
_SOURCE_NUMBER = re.compile(r"(?<![A-Za-z0-9.])-?(\d+(?:\.\d+)?)")


def _numbers(text: str) -> list[tuple[str, float]]:
    return [(m.group(0), float(m.group(1).replace(",", ""))) for m in _NUMBER.finditer(text)]


def unverified_numbers(answer: str, sources: list[str]) -> list[str]:
    """Numbers in `answer` that don't appear (up to rounding) in any of `sources`."""
    known = {float(m.group(1)) for s in sources for m in _SOURCE_NUMBER.finditer(s)}
    roundings = {round(k, d) for k in known for d in (0, 1, 2)}
    return [raw for raw, v in _numbers(answer) if v not in known and v not in roundings]


# --- running the agent ------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class Recommendation:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)
    retried: bool = False
    model: str = ""
    cached: bool = False  # served from the cache, without calling Gemini


def model_name() -> str:
    return os.environ.get("FACTORYMIND_MODEL", DEFAULT_MODEL)


def model_names() -> list[str]:
    """The model to use, then the fallbacks; models that were recently overloaded go last."""
    fallbacks = os.environ.get("FACTORYMIND_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS)
    names = [model_name()] + [m.strip() for m in fallbacks.split(",") if m.strip()]
    now = time.monotonic()
    return sorted(names, key=lambda n: _unavailable_until.get(n, 0) > now)


# A model that just answered 503 is tried last for a while, so each question doesn't spend
# seconds rediscovering that it's overloaded.
OVERLOAD_COOLDOWN_S = 300
_unavailable_until: dict[str, float] = {}


def _mark_overloaded(name) -> None:
    if isinstance(name, str):
        _unavailable_until[name] = time.monotonic() + OVERLOAD_COOLDOWN_S


def _overloaded(error: Exception) -> bool:
    return isinstance(error, errors.ServerError) or getattr(error, "code", None) == 429


def build_agent(session: AnalysisSession, model: str | BaseLlm | None = None,
                instruction: str = INSTRUCTION) -> Agent:
    model = model or model_name()
    if isinstance(model, str):
        # The client otherwise retries an overloaded model (503) with exponential backoff for
        # half a minute or more per call. Fail fast instead, so model_names()'s fallback takes over.
        model = Gemini(model=model, retry_options=types.HttpRetryOptions(
            attempts=2, initial_delay=1, max_delay=2))
    return Agent(
        name="factorymind_advisor",
        model=model,
        instruction=instruction,
        tools=make_tools(session),
        # The cap includes the model's thinking; the prompt keeps the answer itself short.
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2, max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)),
    )


def recommend(files: Mapping[str, bytes], machine_id: int,
              assumptions: Mapping[str, float] | None = None,
              model: str | BaseLlm | None = None, use_cache: bool = True) -> Recommendation:
    """Ask the agent what to do about one machine.

    With no `model` given, answers come from the cache when possible (see cache_key), and
    otherwise from model_names() in order, moving on when one is overloaded or rate-limited.
    """
    session = AnalysisSession(files, assumptions)
    if model is not None:
        return asyncio.run(_recommend(session, machine_id, model))
    key = cache_key(files, machine_id, assumptions) if use_cache else None
    if key:
        hit = _cached(key)
        if hit:
            log.info("recommendation for machine %s: cache hit (no Gemini call)", machine_id)
            return hit
    names = model_names()
    for i, name in enumerate(names):
        started = time.monotonic()
        try:
            rec = asyncio.run(_recommend(session, machine_id, name))
        except Exception as error:  # noqa: BLE001 - re-raised unless it's an overload
            if not _overloaded(error) or i == len(names) - 1:
                raise
            _mark_overloaded(name)
            log.warning("model %s unavailable (%s); trying the next", name, error)
            continue
        log.info("recommendation for machine %s: Gemini call to %s took %.1fs", machine_id,
                 rec.model, time.monotonic() - started)
        if key and not rec.unverified_numbers:
            _memory[key] = rec
        return rec
    raise AssertionError("unreachable")


# --- the response cache -------------------------------------------------------------------------

log = logging.getLogger("factorymind.agent")
# Answers for the demo bundles, generated by scripts/cache_recommendations.py and shipped with
# the app, so the demo never waits on Gemini (even right after Cloud Run scales from zero).
SHIPPED_CACHE = Path(__file__).parent / "recommendation_cache.json"
# Bump when a tool's results change shape or wording.
TOOL_RESULTS_VERSION = 2
# Changing the prompts or the tools changes this, so old answers are no longer used.
PROMPT_VERSION = hashlib.sha256("".join(
    [INSTRUCTION, RECOMMEND_PROMPT, RETRY_PROMPT, str(TOOL_RESULTS_VERSION)]
    + [f"{t.__name__}{inspect.signature(t)}{t.__doc__}" for t in make_tools(None)]
).encode()).hexdigest()[:12]
_memory: dict[str, Recommendation] = {}


def cache_key(files: Mapping[str, bytes], machine_id: int,
              assumptions: Mapping[str, float] | None = None) -> str:
    """The bundle's contents (zip or loose files alike), the machine, the full set of plant
    assumptions and the prompt version."""
    payload = json.dumps({"bundle": bundle_fingerprint(files), "machine_id": machine_id,
                          "assumptions": resolve_assumptions(assumptions),
                          "prompts": PROMPT_VERSION}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cached(key: str) -> Recommendation | None:
    if key in _memory:
        return replace(_memory[key], cached=True)
    entry = _shipped().get(key)
    if entry:
        rec = recommendation_from_dict(entry)
        _memory[key] = rec
        return replace(rec, cached=True)
    return None


@cache
def _shipped() -> dict[str, dict]:
    if not SHIPPED_CACHE.exists():
        return {}
    return json.loads(SHIPPED_CACHE.read_text())["entries"]


def recommendation_to_dict(rec: Recommendation) -> dict:
    return asdict(replace(rec, cached=False))


def recommendation_from_dict(data: dict) -> Recommendation:
    return Recommendation(**{**data, "tool_calls": [ToolCall(**c) for c in data["tool_calls"]]})


async def _recommend(session, machine_id, model) -> Recommendation:
    agent = build_agent(session, model)
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    chat = await runner.session_service.create_session(app_name=APP_NAME, user_id="operator")
    prompt = RECOMMEND_PROMPT.format(machine_id=machine_id)
    calls: list[ToolCall] = []

    text = await _turn(runner, chat.id, prompt, calls)
    sources = [prompt] + [json.dumps(c.result) for c in calls]
    missing = unverified_numbers(text, sources)
    retried = False
    if missing:
        retried = True
        text = await _turn(runner, chat.id, RETRY_PROMPT.format(numbers=", ".join(missing)), calls)
        sources = [prompt] + [json.dumps(c.result) for c in calls]
        missing = unverified_numbers(text, sources)
    name = model if isinstance(model, str) else getattr(model, "model", None) or model_name()
    return Recommendation(text=text, tool_calls=calls, unverified_numbers=missing,
                          retried=retried, model=name)


async def _turn(runner: InMemoryRunner, session_id: str, message: str,
                calls: list[ToolCall]) -> str:
    """Send one user message; record tool calls; return the final text."""
    pending: dict[str, ToolCall] = {}
    text = ""
    async for event in runner.run_async(
            user_id="operator", session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)])):
        for part in (event.content.parts if event.content else None) or []:
            if part.function_call:
                call = ToolCall(part.function_call.name, dict(part.function_call.args or {}), {})
                pending[part.function_call.id or part.function_call.name] = call
                calls.append(call)
            if part.function_response:
                key = part.function_response.id or part.function_response.name
                call = pending.pop(key, None)
                if call is not None:
                    call.result = dict(part.function_response.response or {})
        if event.is_final_response() and event.content and event.content.parts:
            text = "".join(p.text for p in event.content.parts if p.text) or text
    return text.strip()


# --- "Ask FactoryMind" chat -------------------------------------------------------------------

@dataclass
class ChatReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)  # made for this answer
    unverified_numbers: list[str] = field(default_factory=list)
    retried: bool = False
    model: str = ""


class Chat:
    """A conversation with the agent about one bundle. Keeps its history across questions, and
    checks every answer's numbers against all tool results so far."""

    def __init__(self, files: Mapping[str, bytes],
                 assumptions: Mapping[str, float] | None = None,
                 model: str | BaseLlm | None = None):
        self.analysis = AnalysisSession(files, assumptions)
        self.model = model
        self.focus_machine: int | None = None  # the machine the operator is looking at
        self.history: list[tuple[str, str]] = []  # (question, answer)
        self._sessions = InMemorySessionService()
        self._session_id: str | None = None
        self._calls: list[ToolCall] = []
        self._messages: list[str] = []

    def ask(self, question: str) -> ChatReply:
        reply = asyncio.run(self._ask(question))
        self.history.append((question, reply.text))
        return reply

    async def _ask(self, question: str) -> ChatReply:
        if self._session_id is None:
            created = await self._sessions.create_session(app_name=APP_NAME, user_id="operator")
            self._session_id = created.id
        message = (question if self.focus_machine is None else
                   f"(The operator is looking at machine {self.focus_machine}.) {question}")
        self._messages.append(message)
        models = [self.model] if self.model is not None else model_names()
        for i, model in enumerate(models):
            calls: list[ToolCall] = []
            try:
                text = await _turn(self._runner(model), self._session_id, message, calls)
            except Exception as error:  # noqa: BLE001 - re-raised unless it's an overload
                if self.model is not None or not _overloaded(error) or i == len(models) - 1:
                    raise
                _mark_overloaded(model)
                log.warning("model %s unavailable (%s); trying the next", model, error)
                continue
            missing = unverified_numbers(text, self._sources(calls))
            retried = False
            if missing:
                retried = True
                text = await _turn(self._runner(model), self._session_id,
                                   RETRY_PROMPT.format(numbers=", ".join(missing)), calls)
                missing = unverified_numbers(text, self._sources(calls))
            self._calls += calls
            name = model if isinstance(model, str) else getattr(model, "model", "")
            log.info("chat answer: %d tool calls, model %s", len(calls), name)
            return ChatReply(text=text, tool_calls=calls, unverified_numbers=missing,
                             retried=retried, model=name)
        raise AssertionError("unreachable")

    def _runner(self, model) -> Runner:
        # A fresh runner per turn (each turn runs in its own event loop); the history lives in
        # the shared session service.
        return Runner(app_name=APP_NAME, session_service=self._sessions,
                      agent=build_agent(self.analysis, model, CHAT_INSTRUCTION))

    def _sources(self, calls: list[ToolCall]) -> list[str]:
        return self._messages + [json.dumps(c.result) for c in self._calls + calls]
