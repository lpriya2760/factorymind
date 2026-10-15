"""The intervention simulator, through analyze() on the demo bundles."""

from pathlib import Path

import pytest

from factorymind.engine import analyze

DEMO = Path(__file__).parent.parent / "demo_bundles"
LEVERS = {"do_nothing", "reduce_speed", "replace_component", "reroute", "maintain_now"}


def bundle(name):
    return {p.name: p.read_bytes() for p in (DEMO / name).iterdir() if p.is_file()}


@pytest.fixture(scope="module")
def comp2_report():
    return analyze(bundle("comp2_failure_building"))


def by_lever(options):
    return {o.lever: o for o in options}


def test_every_machine_gets_ranked_options_including_do_nothing(comp2_report):
    for machine_id, options in comp2_report.interventions.items():
        levers = {o.lever for o in options}
        assert "do_nothing" in levers and levers <= LEVERS
        assert [o.rank for o in options] == list(range(1, len(options) + 1))


def test_ranking_follows_net_benefit_then_disruption(comp2_report):
    for options in comp2_report.interventions.values():
        for a, b in zip(options, options[1:]):
            assert (a.net_benefit > b.net_benefit
                    or (a.net_benefit == b.net_benefit and a.disruption <= b.disruption))


def test_do_nothing_matches_the_cascade(comp2_report):
    for machine_id, options in comp2_report.interventions.items():
        nothing = by_lever(options)["do_nothing"]
        assert nothing.cost == nothing.loss_avoided == nothing.net_benefit == 0
        assert nothing.expected_loss_after == comp2_report.cascade[machine_id].expected_loss


def test_net_benefit_is_loss_avoided_minus_cost(comp2_report):
    for options in comp2_report.interventions.values():
        for o in options:
            assert o.net_benefit == pytest.approx(o.loss_avoided - o.cost, abs=0.02)


def test_replacing_the_component_lowers_its_predicted_risk(comp2_report):
    machine = next(m for m in comp2_report.machines if m.machine_id == 90)
    replace = by_lever(comp2_report.interventions[90])["replace_component"]
    assert replace.params == {"component": "comp2"}
    assert replace.effect_basis == "model"
    assert replace.component_risk_after["comp2"] < 0.05 < machine.component_risk["comp2"]


def test_machine_90_should_act_rather_than_wait(comp2_report):
    best = comp2_report.interventions[90][0]
    assert best.lever in {"replace_component", "reroute"}
    assert best.net_benefit > 10_000


def test_healthy_machines_are_best_left_alone():
    for options in analyze(bundle("healthy_fleet")).interventions.values():
        assert options[0].lever == "do_nothing"


def test_reroute_only_targets_machines_of_the_same_model(comp2_report):
    models = {m.machine_id: m.model for m in comp2_report.machines}
    for machine_id, options in comp2_report.interventions.items():
        reroute = by_lever(options).get("reroute")
        if reroute:
            assert models[reroute.params["target_machine"]] == models[machine_id]
            assert reroute.params["target_machine"] != machine_id


def test_reroute_is_only_offered_with_a_same_model_machine():
    report = analyze(bundle("comp2_failure_building"))
    models = {m.machine_id: m.model for m in report.machines}
    for machine_id, options in report.interventions.items():
        has_peer = any(models[k] == models[machine_id] and k != machine_id
                       and report.cascade[k].product == report.cascade[machine_id].product
                       for k in models)
        assert ("reroute" in {o.lever for o in options}) == has_peer


def test_slowing_down_is_an_assumed_effect(comp2_report):
    slow = by_lever(comp2_report.interventions[90])["reduce_speed"]
    assert slow.effect_basis == "assumption"
    assert slow.failure_probability_after < comp2_report.cascade[90].failure_probability


def rank_of(report, lever, machine_id=90):
    return by_lever(report.interventions[machine_id])[lever].rank


def test_raising_the_downtime_cost_moves_full_service_up_for_a_failing_machine():
    files = bundle("comp2_failure_building")
    cheap = analyze(files, assumptions={"downtime_cost_per_hour": 0})
    dear = analyze(files, assumptions={"downtime_cost_per_hour": 10_000})
    assert rank_of(cheap, "maintain_now") > rank_of(cheap, "do_nothing")
    assert rank_of(dear, "maintain_now") < rank_of(dear, "do_nothing")


def test_reroute_says_the_target_keeps_its_own_schedule():
    report = analyze(bundle("two_machines_at_risk"))
    reroute = by_lever(report.interventions[95])["reroute"]
    d = reroute.params
    assert d["target_machine"] == 94 and d["target_keeps_its_own_schedule"] is True
    assert d["target_units_per_hour_while_covering"] == pytest.approx(
        d["target_own_units_per_hour"] + d["units_per_hour_moved"], abs=0.11)
    assert d["units_per_hour_moved"] <= d["spare_capacity_share"] * d["target_own_units_per_hour"] + 0.1
    assert "keeps its own" in reroute.note and "on top" in reroute.note
