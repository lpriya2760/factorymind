"""The Gemini agent, tested offline: a scripted model stands in for Gemini (no network, no key)."""

import json
import socket
from pathlib import Path
from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import errors, types

from factorymind import agent
from factorymind.agent import AnalysisSession, make_tools, recommend, unverified_numbers

DEMO = Path(__file__).parent.parent / "demo_bundles"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No network and no key: any attempt to reach Gemini fails the test."""
    def refuse(*args, **kwargs):
        raise AssertionError("network call in an offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture(scope="module")
def files():
    return {p.name: p.read_bytes() for p in (DEMO / "comp2_failure_building").iterdir()
            if p.is_file()}


class ScriptedLlm(BaseLlm):
    """Plays back scripted model turns: ("call", tool, args) or ("text", str | fn(results))."""
    steps: list[Any]

    async def generate_content_async(self, llm_request, stream=False):
        kind, *rest = self.steps.pop(0)
        if kind == "call":
            part = types.Part(function_call=types.FunctionCall(name=rest[0], args=rest[1]))
        else:
            text = rest[0](_tool_results(llm_request)) if callable(rest[0]) else rest[0]
            part = types.Part(text=text)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def _tool_results(llm_request) -> dict[str, dict]:
    """The latest result of each tool in the conversation so far."""
    out = {}
    for content in llm_request.contents:
        for part in content.parts or []:
            if part.function_response:
                out[part.function_response.name] = part.function_response.response
    return out


def good_answer(results):
    best = results["simulate_interventions"]["options"][0]
    risk = results["get_machine_risk"]["component_failure_chance_pct"]["comp2"]
    return (f"{best['action']} [simulate_interventions]: it costs ${best['cost_usd']:,} and has a "
            f"net benefit of ${best['net_benefit_usd']:,} [simulate_interventions]. comp2 is "
            f"{risk}% likely to fail [get_machine_risk].")


def script(*final_texts):
    return ScriptedLlm(model="scripted", steps=[
        ("call", "simulate_interventions", {"machine_id": 90}),
        ("call", "get_machine_risk", {"machine_id": 90}),
        *[("text", t) for t in final_texts],
    ])


# --- the agent loop ---------------------------------------------------------------------------

def test_agent_calls_the_tools_and_quotes_their_numbers(files):
    rec = recommend(files, 90, model=script(good_answer))
    assert [c.name for c in rec.tool_calls] == ["simulate_interventions", "get_machine_risk"]
    assert rec.tool_calls[0].args == {"machine_id": 90}
    assert rec.tool_calls[0].result["options"][0]["action"] == "Replace comp2, machine 47 covers"
    assert "Replace comp2, machine 47 covers" in rec.text
    assert rec.unverified_numbers == [] and not rec.retried


def test_invented_number_triggers_one_rewrite(files):
    rec = recommend(files, 90, model=script("Replace comp2 now; it saves $99,999.", good_answer))
    assert rec.retried
    assert rec.unverified_numbers == []
    assert "99,999" not in rec.text


def test_numbers_still_unverified_after_the_rewrite_are_reported(files):
    rec = recommend(files, 90, model=script("It saves $99,999.", "Still saves $99,999."))
    assert rec.retried and rec.unverified_numbers == ["$99,999"]


def test_overloaded_model_falls_back_to_the_next(monkeypatch, files):
    monkeypatch.setenv("FACTORYMIND_MODEL", "first-model")
    monkeypatch.setenv("FACTORYMIND_FALLBACK_MODELS", "second-model")
    monkeypatch.setattr(agent, "_unavailable_until", {})
    tried = []

    async def fake(session, machine_id, model):
        tried.append(model)
        if model == "first-model":
            raise errors.ServerError(503, {"error": {"message": "overloaded"}})
        return agent.Recommendation(text="ok", model=model)

    monkeypatch.setattr(agent, "_recommend", fake)
    assert recommend(files, 90, use_cache=False).model == "second-model"
    assert tried == ["first-model", "second-model"]


def test_an_overloaded_model_is_tried_last_for_a_while(monkeypatch):
    monkeypatch.setenv("FACTORYMIND_MODEL", "first-model")
    monkeypatch.setenv("FACTORYMIND_FALLBACK_MODELS", "second-model")
    monkeypatch.setattr(agent, "_unavailable_until", {})
    agent._mark_overloaded("first-model")
    assert agent.model_names() == ["second-model", "first-model"]
    monkeypatch.setattr(agent, "OVERLOAD_COOLDOWN_S", -1)
    agent._mark_overloaded("first-model")
    assert agent.model_names() == ["first-model", "second-model"]


def test_models_are_configured_by_environment(monkeypatch):
    monkeypatch.setenv("FACTORYMIND_MODEL", "gemini-x")
    monkeypatch.setenv("FACTORYMIND_FALLBACK_MODELS", "gemini-y, gemini-z")
    monkeypatch.setattr(agent, "_unavailable_until", {})
    assert agent.model_names() == ["gemini-x", "gemini-y", "gemini-z"]


# --- the guardrail ----------------------------------------------------------------------------

SOURCE = json.dumps({"net_benefit_usd": 18176, "chance_pct": 99.83, "note": "3h stop"})


@pytest.mark.parametrize("answer", [
    "Net benefit $18,176.",
    "comp2 is 99.8% likely to fail.",  # rounded from 99.83
    "A 3h stop.",
    "comp2, model3, order O-0001 and error2 are names, not numbers.",
])
def test_guardrail_accepts_numbers_from_tools_and_ignores_identifiers(answer):
    assert unverified_numbers(answer, [SOURCE]) == []


def test_guardrail_accepts_negative_tool_values_quoted_with_a_minus_or_in_words():
    source = json.dumps({"net_benefit_usd": -8502})
    for answer in ["Net benefit −$8,502.", "Net benefit -$8,502.", "It loses $8,502."]:
        assert unverified_numbers(answer, [source]) == []


def test_guardrail_flags_numbers_not_in_any_tool_result():
    assert unverified_numbers("Saves $20,000, about 12%.", [SOURCE]) == ["$20,000", "12"]


# --- the tools --------------------------------------------------------------------------------

def tools(files):
    return {t.__name__: t for t in make_tools(AnalysisSession(files))}


def test_tool_results_are_plain_json(files):
    t = tools(files)
    for result in [t["get_risk_overview"](), t["get_machine_risk"](90), t["project_quality"](90),
                   t["get_cascade"](90), t["simulate_interventions"](90), t["get_assumptions"]()]:
        json.dumps(result)


def test_overview_puts_the_riskiest_machine_first(files):
    first = tools(files)["get_risk_overview"]()["machines"][0]
    assert first["machine_id"] == 90 and first["best_action"] == "Replace comp2, machine 47 covers"


def test_unknown_machine_returns_an_error_not_a_guess(files):
    result = tools(files)["get_cascade"](999)
    assert "No machine 999" in result["error"] and 90 in result["machines"]


def test_project_quality_uses_the_requested_horizon(files):
    t = tools(files)
    assert t["project_quality"](90, 24)["horizon_hours"] == 24
    assert t["project_quality"](90, 1000)["horizon_hours"] == 168
