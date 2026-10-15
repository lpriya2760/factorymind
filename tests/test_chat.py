"""The "Ask FactoryMind" chat, tested offline with a scripted model standing in for Gemini."""

import socket
from pathlib import Path
from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from factorymind.agent import CANT_ANSWER, Chat

DEMO = Path(__file__).parent.parent / "demo_bundles"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
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
    """Plays back turns: ("call", tool, args) or ("text", str | fn(request) -> str).
    Records every request so tests can inspect the history and instructions sent."""
    steps: list[Any]
    requests: list[Any] = []

    async def generate_content_async(self, llm_request, stream=False):
        self.requests.append(llm_request)
        kind, *rest = self.steps.pop(0)
        if kind == "call":
            part = types.Part(function_call=types.FunctionCall(name=rest[0], args=rest[1]))
        else:
            part = types.Part(text=rest[0](llm_request) if callable(rest[0]) else rest[0])
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def latest(request, tool):
    for content in reversed(request.contents):
        for part in content.parts or []:
            if part.function_response and part.function_response.name == tool:
                return part.function_response.response
    raise AssertionError(f"no {tool} result in the request")


def customers(request):
    late = latest(request, "get_cascade")["if_it_fails"]["late_orders"]
    return ", ".join(f"{o['customer']} ({o['order_id']}, ${o['penalty_usd']:,} penalty "
                     "[get_cascade])" for o in late)


def all_text(request) -> str:
    return " ".join(p.text for c in request.contents for p in (c.parts or []) if p.text)


def test_answers_use_tools_and_keep_the_history(files):
    llm = ScriptedLlm(model="scripted", steps=[
        ("call", "get_cascade", {"machine_id": 90}),
        ("text", customers),
        ("text", lambda r: "The penalty is $1,990 [get_cascade], as above."),
    ])
    chat = Chat(files, model=llm)
    chat.focus_machine = 90
    first = chat.ask("Which customers are affected if it fails?")
    assert [c.name for c in first.tool_calls] == ["get_cascade"]
    assert first.tool_calls[0].args == {"machine_id": 90}
    assert "Everline Packaging" in first.text and "[get_cascade]" in first.text

    second = chat.ask("Remind me of the penalty?")
    # The second request carries the whole conversation, including the first answer.
    history = all_text(llm.requests[-1])
    assert "Which customers are affected" in history and "Everline Packaging" in history
    assert "machine 90" in history  # the focus machine is passed along
    # $1,990 came from the first turn's tool result, so it's verified without a new call.
    assert second.tool_calls == [] and second.unverified_numbers == []
    assert chat.history == [("Which customers are affected if it fails?", first.text),
                            ("Remind me of the penalty?", second.text)]


def test_questions_the_data_cant_answer_get_an_explicit_reply(files):
    def reply(request):
        # The chat instruction tells the model how to say it can't answer.
        assert CANT_ANSWER in str(request.config.system_instruction)
        return f"{CANT_ANSWER}: the tools don't simulate waiting until the weekend."

    reply_text = Chat(files, model=ScriptedLlm(model="scripted", steps=[("text", reply)])).ask(
        "What if we wait until the weekend?").text
    assert reply_text.startswith(CANT_ANSWER)


def test_invented_numbers_in_chat_trigger_a_rewrite(files):
    chat = Chat(files, model=ScriptedLlm(model="scripted", steps=[
        ("call", "get_cascade", {"machine_id": 90}),
        ("text", "It would cost about $50,000."),
        ("text", customers),
    ]))
    reply = chat.ask("Which customers are affected?")
    assert reply.retried and reply.unverified_numbers == []
    assert "50,000" not in reply.text


def test_numbers_still_unverified_are_reported(files):
    chat = Chat(files, model=ScriptedLlm(model="scripted", steps=[
        ("text", "It would cost about $50,000."), ("text", "Still about $50,000."),
    ]))
    assert chat.ask("How bad is it?").unverified_numbers == ["$50,000"]
