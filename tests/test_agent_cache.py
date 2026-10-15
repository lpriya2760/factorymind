"""The recommendation cache: repeated questions don't call Gemini again."""

import json
from pathlib import Path

import pytest

from factorymind import agent
from factorymind.agent import Recommendation, ToolCall, cache_key, recommend

DEMO = Path(__file__).parent.parent / "demo_bundles"


@pytest.fixture
def files():
    return {p.name: p.read_bytes() for p in (DEMO / "comp2_failure_building").iterdir()
            if p.is_file()}


@pytest.fixture
def gemini(monkeypatch, tmp_path):
    """A fake Gemini that counts calls, with empty caches."""
    calls = []

    async def fake(session, machine_id, model):
        calls.append((machine_id, model))
        return Recommendation(text=f"answer {len(calls)}", model=model,
                              tool_calls=[ToolCall("simulate_interventions",
                                                   {"machine_id": machine_id}, {"ok": 1})])

    monkeypatch.setattr(agent, "_recommend", fake)
    monkeypatch.setattr(agent, "_memory", {})
    monkeypatch.setattr(agent, "SHIPPED_CACHE", tmp_path / "none.json")
    agent._shipped.cache_clear()
    yield calls
    agent._shipped.cache_clear()


def test_asking_again_uses_the_cache(gemini, files):
    first = recommend(files, 90)
    second = recommend(files, 90)
    assert len(gemini) == 1
    assert not first.cached and second.cached and second.text == first.text


def test_a_zip_and_its_loose_files_share_answers(gemini, files):
    recommend(files, 90)
    zipped = {"comp2.zip": (DEMO / "comp2_failure_building.zip").read_bytes()}
    assert recommend(zipped, 90).cached and len(gemini) == 1


def test_different_machine_or_assumptions_ask_again(gemini, files):
    recommend(files, 90)
    recommend(files, 72)
    recommend(files, 90, {"downtime_cost_per_hour": 300})
    assert len(gemini) == 3


def test_default_assumptions_given_explicitly_share_the_cache(gemini, files):
    recommend(files, 90)
    assert recommend(files, 90, {"downtime_cost_per_hour": 2500}).cached


def test_answers_with_unverified_numbers_are_not_cached(gemini, monkeypatch, files):
    async def doubtful(session, machine_id, model):
        gemini.append(machine_id)
        return Recommendation(text="It saves $99,999.", unverified_numbers=["$99,999"])

    monkeypatch.setattr(agent, "_recommend", doubtful)
    recommend(files, 90)
    recommend(files, 90)
    assert len(gemini) == 2


def test_shipped_answers_are_used_without_calling_gemini(gemini, monkeypatch, tmp_path, files):
    shipped = Recommendation(text="shipped answer", model="gemini-x",
                             tool_calls=[ToolCall("get_cascade", {"machine_id": 90}, {"a": 1})])
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"entries": {
        cache_key(files, 90): agent.recommendation_to_dict(shipped)}}))
    monkeypatch.setattr(agent, "SHIPPED_CACHE", path)
    agent._shipped.cache_clear()
    rec = recommend(files, 90)
    assert rec.cached and rec.text == "shipped answer" and gemini == []
    assert rec.tool_calls[0].name == "get_cascade"


def test_the_shipped_cache_covers_every_demo_machine():
    """scripts/cache_recommendations.py must be re-run after prompts or demo data change."""
    entries = json.loads(agent.SHIPPED_CACHE.read_text())["entries"]
    for bundle in sorted(p for p in DEMO.iterdir() if p.is_dir()):
        files = {p.name: p.read_bytes() for p in bundle.iterdir() if p.is_file()}
        machines = [int(line.split(",")[0]) for line in
                    (bundle / "machines.csv").read_text().splitlines()[1:]]
        missing = [m for m in machines if cache_key(files, m) not in entries]
        assert not missing, f"{bundle.name}: no cached answer for machines {missing}"
