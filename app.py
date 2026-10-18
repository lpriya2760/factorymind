"""FactoryMind Streamlit UI. All analysis goes through factorymind.engine.analyze()."""

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from factorymind.bundle import MAX_BUNDLE_BYTES, TABLES, BundleValidationError
from factorymind import quality
from factorymind.agent import Chat, recommend
from factorymind.alerts import AlertConfig, build_alert, format_email, send_alert
from factorymind.alerts import subject as alert_subject
from factorymind.assumptions import ASSUMPTION_INFO, DEFAULT_ASSUMPTIONS
from factorymind.cascade import CASCADE_HORIZON_H, FAILURE_AT_H, MC_SAMPLES, PREDICTION_WINDOW_H
from factorymind.engine import DEFAULT_QUALITY_HORIZON_H, analyze
from factorymind.predict import MODEL_DIR
from factorymind import ui

# Local development reads GOOGLE_API_KEY from .env; Cloud Run uses Vertex AI settings instead.
load_dotenv(Path(__file__).with_name(".env"))

# FactoryMind's own log lines (e.g. cache hit or Gemini call) go to stderr, which Cloud Run keeps.
_log = logging.getLogger("factorymind")
_log.setLevel(logging.INFO)
if not _log.handlers:  # Streamlit re-runs this script on every interaction
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    _log.addHandler(_handler)

st.set_page_config(page_title="FactoryMind", page_icon="🏭", layout="wide")
ui.inject_css()
ui.hero()
ui.section("upload", "Analyse your plant",
           "Upload your own data, or try a demo bundle with real sensor data.")

uploads = st.file_uploader(
    f"Upload a bundle: the {len(TABLES)} CSV files (or one .zip), plus an optional manifest.json. "
    f"Max {MAX_BUNDLE_BYTES // 2**20} MB.",
    type=["csv", "zip", "json"],
    accept_multiple_files=True,
)

with st.expander("Expected files"):
    st.dataframe(
        pd.DataFrame([
            {"file": f"{name}.csv", "columns": ", ".join(spec.columns),
             "assumed provenance": spec.provenance}
            for name, spec in TABLES.items()
        ]),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        'Kaggle names such as `PdM_telemetry.csv` also work. To change provenance, add a '
        '`manifest.json` like `{"provenance": {"orders": "real"}}`.'
    )



def assumption_panel() -> dict[str, float]:
    """Sidebar inputs for the plant assumptions; returns their current values."""
    with st.sidebar:
        st.header("Plant assumptions")
        st.caption("Every money figure and the ranking of actions use these. Change one and the "
                   "whole analysis re-runs.")
        if st.button("Reset to defaults"):
            for key, value in DEFAULT_ASSUMPTIONS.items():
                st.session_state[f"assume_{key}"] = value
        values = {}
        for key, info in ASSUMPTION_INFO.items():
            default = DEFAULT_ASSUMPTIONS[key]
            st.session_state.setdefault(f"assume_{key}", default)
            values[key] = st.number_input(
                f"{info.label} ({info.unit})", min_value=0.0, max_value=info.max_value,
                step=info.step, format="%.0f" if info.step >= 1 else "%.2f",
                key=f"assume_{key}", help=f"{info.help} Default: {default:g}.")
        changed = {k: v for k, v in values.items() if v != DEFAULT_ASSUMPTIONS[k]}
        if changed:
            st.info("Changed from default: " + "; ".join(
                f"{ASSUMPTION_INFO[k].label.lower()} {v:g} (default {DEFAULT_ASSUMPTIONS[k]:g})"
                for k, v in changed.items()))
    return values


assumptions = assumption_panel()

# Demo bundles shipped with the app: real Azure PdM telemetry plus synthetic business data.
DEMO_DIR = Path(__file__).with_name("demo_bundles")
DEMOS = {
    "comp2_failure_building": "🔴 A comp2 failure building",
    "two_machines_at_risk": "🟠 Two machines at risk",
    "healthy_fleet": "🟢 A healthy fleet",
}
demos = {name: DEMO_DIR / f"{name}.zip" for name in DEMOS if (DEMO_DIR / f"{name}.zip").exists()}
if demos:
    st.markdown("**No data to hand? Try a demo bundle**: 10 machines × 30 days of real sensor "
                "data from the Azure Predictive Maintenance dataset, with synthetic orders.")
    for col, (name, path) in zip(st.columns(len(demos)), demos.items()):
        if col.button(DEMOS[name], key=f"demo-{name}", width="stretch"):
            st.session_state["demo"] = name
        col.download_button("⬇ Download .zip", path.read_bytes(), file_name=path.name,
                            key=f"download-{name}", width="stretch")

if uploads:
    files = {u.name: u.getvalue() for u in uploads}
elif st.session_state.get("demo") in demos:
    demo = st.session_state["demo"]
    files = {demos[demo].name: demos[demo].read_bytes()}
    st.success(f"Showing the demo bundle **{DEMOS[demo]}**. Upload your own files above to "
               "analyse them instead.")
else:
    st.info("Upload a bundle, or pick a demo bundle above, to start.")
    ui.how_it_works()
    ui.features()
    ui.footer()
    st.stop()

@st.cache_data(show_spinner="Analysing…", max_entries=8)
def run_analysis(files: tuple[tuple[str, bytes], ...], quality_horizon_hours: int,
                 assumptions: tuple[tuple[str, float], ...]):
    return analyze(dict(files), dict(assumptions), quality_horizon_hours=quality_horizon_hours)


# The horizon slider is drawn further down; read its value first so the analysis uses it.
st.session_state.setdefault("horizon", DEFAULT_QUALITY_HORIZON_H)
try:
    report = run_analysis(tuple(files.items()),
                          st.session_state["horizon"], tuple(sorted(assumptions.items())))
except BundleValidationError as exc:
    st.error(f"The bundle has {len(exc.errors)} problem(s). Fix them and upload again.")
    st.dataframe(pd.DataFrame([asdict(e) for e in exc.errors]), hide_index=True, width="stretch")
    st.stop()

synthetic = [name for name, p in report.data_provenance.items() if p == "synthetic"]
if synthetic:
    st.warning("Synthetic inputs: " + ", ".join(f"`{name}`" for name in synthetic))

LEVEL_BADGES = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}


def synthetic_badge(*tables: str) -> str:
    """A badge for figures built on synthetic inputs."""
    return (" :orange-badge[synthetic data]"
            if any(report.data_provenance.get(t) == "synthetic" for t in tables) else "")
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]

high_count = sum(m.risk_level == "high" for m in report.machines)
total_loss = sum(c.expected_loss for c in report.cascade.values())
best_saving = max((o[0].net_benefit for o in report.interventions.values()), default=0.0)
ui.kpis([
    ("navy", "Machines analysed", f"{len(report.machines)}", "failure risk for the next 24h"),
    ("red" if high_count else "teal", "High risk", f"{high_count}",
     "need attention now" if high_count else "nothing urgent"),
    ("amber", "Expected loss this week", f"${total_loss:,.0f}", "across every machine, if nothing is done"),
    ("teal", "Best single action saves", f"${max(best_saving, 0):,.0f}",
     "net of its own cost"),
])

ui.section("pulse", f"Failure risk in the next 24h ({len(report.machines)} machines)",
           "Riskiest first. Each component's chance of failing, and why.")
high = [m for m in report.machines if m.risk_level == "high"]
if high:
    st.error("  \n".join(
        f"**Machine {m.machine_id}**: {m.top_component} is "
        f"{'>99%' if m.max_risk > 0.99 else f'{m.max_risk:.0%}'} likely to fail in the next 24h"
        for m in high
    ))

ui.risk_table(report.machines, COMPONENTS)

with st.expander("About the failure model"):
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    st.markdown(
        "A BigQuery ML boosted-tree classifier, trained on "
        f"{metrics['train_period']} of the Azure Predictive Maintenance data and tested on "
        f"{metrics['test_period']}, which it never saw. Its features are rolling 3h and 24h "
        "sensor averages and variability, error counts, time since each part was replaced, and "
        "machine model and age. It runs inside this app, with no external calls."
    )
    st.dataframe(
        pd.DataFrame(metrics["components"]).T.rename_axis("component").reset_index(),
        hide_index=True,
    )
    st.caption(
        "The public dataset is simulated, and every failure is preceded by a clear sensor "
        "anomaly, so these held-out scores are higher than you'd expect in a real plant."
    )

bundle_key = hashlib.sha256(b"".join(files.values())).hexdigest()[:16]

ui.section("bell", "Alerts", "Tell the team on Slack or email which machines need action.",
           ui.AMBER)
try:
    alert_config = AlertConfig.from_env()
except ValueError as exc:
    st.error(f"Alert settings are invalid: {exc}")
    alert_config = None
if alert_config is not None:
    alert = build_alert(report, alert_config.min_level)
    if alert is None:
        st.caption(f"No machine is at {alert_config.min_level} risk, so there's nothing to alert.")
    else:
        with st.expander(f"Alert preview: {alert_subject(alert)}"):
            st.text(format_email(alert, alert_config.email_from or "factorymind",
                                 alert_config.email_to or ["(no recipients)"]).get_content())
        if not alert_config.channels:
            st.caption("Set `SLACK_WEBHOOK_URL` and/or the `ALERT_EMAIL_TO` and `SMTP_*` "
                       "settings to send this alert to Slack or email (see the README).")
        else:
            sent = st.session_state.setdefault("alerts_sent", set())
            alert_key = (bundle_key, tuple(sorted(assumptions.items())))
            label = "Send alert to " + " and ".join(alert_config.channels)
            if alert_key in sent:
                st.caption("✅ Alert already sent for this data and these assumptions.")
                label = "Send again"
            if st.button(label, key="send-alert"):
                with st.spinner("Sending…"):
                    results = send_alert(alert, alert_config)
                for channel, error in results.items():
                    if error:
                        _log.warning("Alert to %s failed: %s", channel, error)
                        st.error(f"Couldn't send to {channel}: {error}")
                    else:
                        _log.info("Alert sent to %s", channel)
                        st.success(f"Sent to {channel}.")
                if not any(results.values()):
                    sent.add(alert_key)

ui.section("gear", "Machine detail", "Pick a machine to see its quality, cascade and options.")
by_id = {m.machine_id: m for m in report.machines}
machine_id = st.selectbox(
    "Machine", list(by_id),
    format_func=lambda i: f"Machine {i} · {LEVEL_BADGES[by_id[i].risk_level]} failure risk",
)
st.slider("Hours ahead", min_value=6, max_value=168, step=6, key="horizon")

ui.section("drop", "Quality if it keeps running", "Projected defect rate and scrap cost.")
q = report.quality[machine_id]
horizon = q.horizon_hours[-1]
now_col, later_col, scrap_col = st.columns(3)
now_col.metric("Defect rate now", f"{q.defect_rate[0]:.1%}")
later_col.metric(f"Defect rate in {horizon}h", f"{q.defect_rate[-1]:.1%}",
                 delta=f"{(q.defect_rate[-1] - q.defect_rate[0]) * 100:+.1f} pts",
                 delta_color="inverse")
scrap_col.metric(f"Scrap cost over {horizon}h", f"${q.scrap_cost:,.0f}")
st.line_chart(
    pd.DataFrame({"Defect rate (%)": [r * 100 for r in q.defect_rate]},
                 index=pd.Index(q.horizon_hours, name="Hours from now")),
    y_label="Defect rate (%)",
)
growth = (f", widening {q.drift_growth_per_hour * 24:.1%} a day"
          if q.drift_growth_per_hour > 0 else "")
st.caption(
    f"Why: oldest component replaced {q.component_age_hours / 24:.0f} days ago · "
    f"{q.drift_sensor} is {q.drift:.1%} away from this machine's normal level{growth} · "
    f"{q.errors_24h} error(s) in the last 24h · {q.units_per_hour:.0f} units/hour scheduled · "
    f"scrap costs \\${report.assumptions_used['scrap_cost_per_unit']:,.0f} a unit"
)

with st.expander("How quality is projected"):
    st.markdown(
        "Public machine datasets have no defect records, so FactoryMind uses a transparent "
        "formula instead of a learned model. The defect rate *h* hours from now is:"
    )
    st.latex(
        r"\text{defect}(h) = \min\Big(\text{MAX\_RATE},\ \text{BASE\_RATE}"
        r" + \text{WEAR\_WEIGHT}\cdot\min(\text{wear}(h), \text{WEAR\_CAP})^2"
        r" + \text{DRIFT\_WEIGHT}\cdot\min\big(\max(\text{drift}(h) - \text{DRIFT\_DEADBAND}, 0),"
        r" \text{DRIFT\_CAP}\big) + \text{ERROR\_WEIGHT}\cdot\text{errors}_{24h}\Big)"
    )
    st.markdown(
        "- **wear(h)** = (hours since the oldest component was replaced + *h*) ÷ TYPICAL_LIFE_H\n"
        "- **drift** = the largest gap between a sensor's 24h average and this machine's own "
        "normal level (its median across the uploaded data), as a share of normal\n"
        "- **drift(h)** = drift now + its growth per hour over the last 24h × min(*h*, "
        "TREND_HORIZON_H): a day's trend is projected two days ahead, then held flat. "
        "Drift never shrinks.\n"
        "- **scrap cost** = Σ over the hours of units per hour × defect(h) × scrap cost per unit"
    )
    st.dataframe(
        pd.DataFrame([{"parameter": k, "value": v, "meaning": m}
                      for k, (v, m) in quality.PARAMETERS.items()]),
        hide_index=True, width="stretch",
    )

ui.section("cascade", "What a failure would cause",
           "Machine → quality → inventory → orders → money, over the next week.", ui.RED)
c = report.cascade[machine_id]
f, ok = c.if_failure, c.if_no_failure
p = c.failure_probability
p_text = ">99%" if p > 0.99 else "<1%" if p < 0.01 else f"{p:.0%}"


def breach_text(hours):
    return f"below safety stock in {hours:.0f}h" if hours is not None else "stays above safety stock"


def _synthetic(*tables: str) -> str | None:
    return ("synthetic data"
            if any(report.data_provenance.get(t) == "synthetic" for t in tables) else None)


def _tone(bad: bool, warn: bool = False) -> str:
    return "red" if bad else "amber" if warn else "teal"


m_ = ui.money
ui.cascade_chain([
    {"icon": "gear", "title": "Machine", "label": "Failure in 24h", "value": p_text,
     "tone": _tone(p >= 0.5, p >= 0.1), "alarm": p >= 0.5,
     "notes": [f"If it fails: down <b>{c.downtime_hours:g}h</b> from about hour "
               f"{c.failure_at_hours:g}"]},
    {"icon": "drop", "title": "Quality", "label": "Excess scrap if it fails",
     "value": m_(f.excess_scrap_cost), "tone": _tone(False, ok.excess_scrap_cost > 0),
     "notes": [f"<b>{m_(ok.excess_scrap_cost)}</b> if it keeps running for "
               f"{CASCADE_HORIZON_H}h", "A repair restores normal quality"]},
    {"icon": "box", "title": f"Inventory ({c.product or 'no product'})",
     "tag": _synthetic("inventory", "production_schedule"), "label": "Units not made",
     "value": f"{c.lost_units:,.0f}",
     "tone": _tone(f.inventory_breach_hours is not None, c.lost_units > 0),
     "notes": [f"If it fails: {breach_text(f.inventory_breach_hours)}",
               f"If not: {breach_text(ok.inventory_breach_hours)}"]},
    {"icon": "truck", "title": "Orders", "tag": _synthetic("orders"),
     "label": "Late if it fails", "value": f"{len(f.late_orders)}",
     "tone": _tone(len(f.late_orders) > 0),
     "notes": [f"<b>{m_(f.penalties)}</b> penalties if it fails",
               f"If not: {len(ok.late_orders)} late, {m_(ok.penalties)}"]},
    {"icon": "coin", "title": "Loss", "label": "Expected loss", "value": m_(c.expected_loss),
     "tone": "navy",
     "notes": [f"<b>{m_(f.total_loss)}</b> if it fails · <b>{m_(ok.total_loss)}</b> if not, "
               f"weighted by the {p_text} failure chance"]},
], loss_range={"lo": c.loss_p10, "mid": c.loss_p50, "hi": c.loss_p90})

if c.loss_p10 == c.loss_p90 and c.expected_loss > c.loss_p90:
    range_note = (
        f"In at least {c.loss_samples * 9 // 10:,} of {c.loss_samples:,} simulated weeks it "
        f"keeps running and "
        f"loses \\${c.loss_p50:,.0f}, so P10, P50 and P90 are all that figure. The expected loss is higher because the {p_text} chance "
        f"of a failure would cost \\${f.total_loss:,.0f}.")
elif c.loss_p10 == c.loss_p90:
    range_note = f"Every percentile is \\${c.loss_p50:,.0f}: the outcome barely varies."
else:
    range_note = (
        f"Across {c.loss_samples:,} simulated weeks, drawing whether and when in the next 24h it "
        f"fails: **P10 \\${c.loss_p10:,.0f} · P50 \\${c.loss_p50:,.0f} · "
        f"P90 \\${c.loss_p90:,.0f}**. One week in ten loses more than \\${c.loss_p90:,.0f}.")
st.markdown("**Loss range (Monte Carlo):** " + range_note)

def scenario_column(o):
    return [str(len(o.late_orders)), breach_text(o.inventory_breach_hours).capitalize()] + [
        f"${x:,.0f}" for x in (o.penalties, o.excess_scrap_cost, o.downtime_cost, o.repair_cost,
                               o.total_loss)]


st.dataframe(
    pd.DataFrame(
        {"If it fails": scenario_column(f), "If it keeps running": scenario_column(ok)},
        index=["Late orders", "Safety stock", "Penalties", "Excess scrap", "Downtime",
               "Repair", "Total loss"],
    ),
    width="stretch",
)

if f.late_orders:
    st.markdown("**Orders late if it fails**")
    st.dataframe(
        pd.DataFrame([
            {"Order": o.order_id, "Customer": o.customer, "Product": o.product,
             "Due": f"in {o.due_hours:.0f}h",
             "Hours late": o.hours_late if o.shipped else f"{o.hours_late:.0f}+ (not shipped)",
             "Penalty ($)": o.penalty}
            for o in f.late_orders
        ]),
        hide_index=True, width="stretch",
    )

with st.expander("How the cascade is simulated"):
    st.markdown(
        f"Hour 0 is the end of the uploaded data. FactoryMind simulates the next "
        f"{CASCADE_HORIZON_H}h hour by hour, twice:\n\n"
        f"- **If it fails**: it fails at hour {FAILURE_AT_H} (the middle of the 24h prediction "
        "window) and is down for the repair time. Repair restores normal quality.\n"
        "- **If it keeps running**: it keeps producing at its projected defect rate.\n\n"
        "Every machine making the same product adds its *good* output each hour. Orders ship "
        "first in, first out by due date when there is enough stock; safety stock can be used. "
        "Safety stock is breached when stock minus orders due but not yet shipped falls below "
        "it. A late order's penalty is penalty per day × days late (part days count) × the "
        "penalty multiplier.\n\n"
        "**Loss if it fails** = downtime + repair (part and labour) + penalties + excess scrap. "
        "**Loss if it keeps running** = penalties + excess scrap. **Expected loss** weights "
        "them by the failure chance. Excess scrap is scrap above the normal "
        f"{quality.BASE_RATE:.1%} defect rate.\n\n"
        f"**Loss range:** {MC_SAMPLES:,} seeded Monte Carlo draws. Each draw decides whether the "
        f"machine fails (with the predicted chance) and, if so, in which hour of the next "
        f"{PREDICTION_WINDOW_H}h (each equally likely), then re-runs the same week. P10/P50/P90 "
        "are the 10th, 50th and 90th percentile losses, each one an outcome that actually occurs "
        "in a draw. The same data and seed always give the same range."
    )
    st.caption("Costs use the plant assumptions in the sidebar: " + ", ".join(
        f"{k.replace('_', ' ')} {v:g}" for k, v in report.assumptions_used.items()))

ui.section("sliders", "What to do about it", "Every action re-simulated and ranked by net benefit.")
options = report.interventions[machine_id]
best = options[0]
if best.lever == "do_nothing":
    alt = options[1]
    effect = (f"to avoid \\${alt.loss_avoided:,.0f} of expected loss" if alt.loss_avoided > 0
              else "and wouldn't reduce the expected loss")
    st.success(f"**Best: do nothing.** No action pays for itself: the best alternative, "
               f"*{alt.label.lower()}*, would cost \\${alt.cost:,.0f} {effect}.")
else:
    st.success(f"**Best: {best.label}.** Costs \\${best.cost:,.0f} and avoids "
               f"\\${best.loss_avoided:,.0f} of expected loss: a net benefit of "
               f"**\\${best.net_benefit:,.0f}**. {best.note}")

DISRUPTION_LABELS = ["None", "Low", "Medium", "Medium-high", "High"]


def money(x: float) -> str:
    return f"−${-x:,.0f}" if x < 0 else f"${x:,.0f}"


st.dataframe(
    pd.DataFrame([
        {
            "Rank": o.rank,
            "Action": o.label,
            "Cost": money(o.cost),
            "Failure chance after": (
                (">99%" if o.failure_probability_after > 0.99
                 else "<1%" if o.failure_probability_after < 0.01
                 else f"{o.failure_probability_after:.0%}")
                + (" (assumed)" if o.effect_basis == "assumption" else "")),
            "Expected loss after": money(o.expected_loss_after),
            "Loss avoided": money(o.loss_avoided),
            "Net benefit": money(o.net_benefit),
            "Disruption": DISRUPTION_LABELS[o.disruption],
        }
        for o in options
    ]),
    hide_index=True, width="stretch",
)

with st.expander("How each action is simulated"):
    st.markdown(
        "Every action changes the inputs and re-runs the same failure/no-failure cascade as "
        "doing nothing. **Net benefit = loss avoided − the action's cost**. Ties go to the less "
        "disruptive action. One accounting rule applies everywhere: each hour of a machine's "
        "output that isn't made anywhere costs the downtime cost per hour.\n\n"
        + "\n".join(f"- **{o.label}**: {o.note}" for o in options)
    )


def gemini_unavailable(exc: Exception) -> None:
    """Explain a failed Gemini call in plain words; keep the details in the log."""
    logging.getLogger("factorymind.app").warning("Gemini call failed: %s", exc)
    busy = any(code in str(exc) for code in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
    st.error("Gemini is busy right now (overloaded or over quota). Please try again in a minute."
             if busy else f"Gemini couldn't answer right now: {exc}")

ui.section("spark", "Gemini's recommendation", "An AI explanation, with every figure checked.")
recs = st.session_state.setdefault("recommendations", {})
rec_key = (bundle_key, machine_id, tuple(sorted(assumptions.items())))
if st.button("Ask Gemini what to do", key=f"ask-{machine_id}", type="primary"):
    with st.spinner("Gemini is checking the risk, the cascade and the options…"):
        try:
            recs[rec_key] = recommend(files, machine_id,
                                      assumptions)
        except Exception as exc:  # noqa: BLE001 - shown to the operator, not swallowed
            gemini_unavailable(exc)
rec = recs.get(rec_key)
if rec is None:
    st.caption("Gemini reads the same analysis through tools and explains the best action. "
               "Every figure it quotes is checked against what the tools returned.")
else:
    with st.container(border=True):
        st.markdown(rec.text.replace("$", "\\$"))
    if rec.unverified_numbers:
        st.warning("Couldn't verify these figures against the tool results: "
                   + ", ".join(rec.unverified_numbers))
    else:
        st.caption(f"✅ Every figure was checked against the tool results"
                   f"{' after one rewrite' if rec.retried else ''} · {rec.model}"
                   f"{' · ⚡ cached answer, no new Gemini call' if rec.cached else ''}")
    with st.expander(f"Tools Gemini used ({len(rec.tool_calls)})"):
        for call in rec.tool_calls:
            args = ", ".join(f"{k}={v}" for k, v in call.args.items())
            st.markdown(f"**`{call.name}({args})`**")
            st.json(call.result, expanded=False)

ui.section("chat", "Ask FactoryMind", "Follow-up questions, answered from the same analysis.")
# One conversation per bundle and set of assumptions: if either changes, the data did too.
chat_key = (bundle_key, tuple(sorted(assumptions.items())))
if st.session_state.get("chat_key") != chat_key:
    st.session_state["chat_key"] = chat_key
    st.session_state["chat"] = Chat(files, assumptions)
    st.session_state["chat_log"] = []
chat = st.session_state["chat"]
chat.focus_machine = machine_id
chat_log = st.session_state["chat_log"]
st.caption(f"Ask follow-up questions. Gemini answers with the same tools and every figure is "
           f"checked. Questions are about machine {machine_id} unless you name another.")


def show_reply(reply):
    st.markdown(reply.text.replace("$", "\\$"))
    if reply.unverified_numbers:
        st.warning("Couldn't verify: " + ", ".join(reply.unverified_numbers))
    else:
        st.caption(f"✅ Figures checked{' after one rewrite' if reply.retried else ''} · "
                   f"{reply.model}")
    if reply.tool_calls:
        with st.expander(f"Tools used ({len(reply.tool_calls)})"):
            for call in reply.tool_calls:
                args = ", ".join(f"{k}={v}" for k, v in call.args.items())
                st.markdown(f"**`{call.name}({args})`**")
                st.json(call.result, expanded=False)


SUGGESTIONS = ["Which customers are affected if it fails?",
               "What if we wait until the weekend?",
               "Which machine should I worry about most?"]
picked = None
for col, suggestion in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
    if col.button(suggestion, key=f"suggest-{suggestion}", width="stretch"):
        picked = suggestion

for asked, reply in chat_log:
    with st.chat_message("user"):
        st.markdown(asked.replace("$", "\\$"))
    with st.chat_message("assistant"):
        show_reply(reply)

question = st.chat_input("Ask about the machines, the risks or the options…") or picked
if question:
    with st.chat_message("user"):
        st.markdown(question.replace("$", "\\$"))
    with st.chat_message("assistant"):
        with st.spinner("Checking the data…"):
            try:
                reply = chat.ask(question)
            except Exception as exc:  # noqa: BLE001 - shown to the operator, not swallowed
                gemini_unavailable(exc)
                reply = None
        if reply:
            chat_log.append((question, reply))
            show_reply(reply)

if chat_log and st.button("New conversation"):
    st.session_state.pop("chat_key")
    st.rerun()

ui.footer()
