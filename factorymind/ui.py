"""Presentation helpers for the Streamlit app: styles, illustrations and layout blocks.

Pure HTML/CSS/SVG, no analysis. Everything the app computes still comes from engine.analyze();
these functions only decide how it looks. All illustrations are original inline SVG, so the
app needs no image files or network access.
"""

from __future__ import annotations

import base64
from html import escape

import streamlit as st

NAVY = "#0b1f3a"
TEAL = "#0ea5a4"
AMBER = "#f59e0b"
RED = "#e5484d"

CSS = f"""
<style>
.block-container {{ padding-top: 3.2rem; max-width: 1280px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}

/* Hero */
.fm-hero {{ display: grid; grid-template-columns: 1.15fr 1fr; gap: 2rem; align-items: center;
  background: linear-gradient(135deg, {NAVY} 0%, #12325c 55%, #0f4c5c 100%);
  border-radius: 20px; padding: 2.4rem 2.6rem; color: #fff; margin-bottom: 1.4rem;
  box-shadow: 0 18px 40px rgba(11,31,58,.25); overflow: hidden; }}
.fm-hero .fm-eyebrow {{ display: inline-block; font-size: .78rem; font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase; color: #7ee7e6;
  background: rgba(14,165,164,.15); border: 1px solid rgba(126,231,230,.35);
  padding: .3rem .7rem; border-radius: 999px; margin-bottom: 1rem; }}
.fm-hero h1 {{ color: #fff; font-size: 2.5rem; line-height: 1.12; margin: 0 0 .8rem 0;
  padding: 0; font-weight: 800; }}
.fm-hero h1 span {{ color: #7ee7e6; }}
.fm-hero p {{ color: #c9d6e8; font-size: 1.05rem; line-height: 1.55; margin: 0 0 1.2rem 0; }}
.fm-pills {{ display: flex; flex-wrap: wrap; gap: .5rem; }}
.fm-pill {{ font-size: .85rem; color: #e6eef8; background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.16); padding: .35rem .75rem; border-radius: 999px; }}
.fm-hero .fm-art {{ width: 100%; height: auto; display: block; }}
@media (max-width: 900px) {{ .fm-hero {{ grid-template-columns: 1fr; padding: 1.6rem; }}
  .fm-hero h1 {{ font-size: 1.9rem; }} }}

/* Section headers */
.fm-section {{ display: flex; align-items: center; gap: .8rem; margin: 2rem 0 .6rem 0; }}
.fm-section .fm-icon {{ flex: none; width: 42px; height: 42px; border-radius: 12px;
  display: grid; place-items: center; background: #e7f6f6; }}
.fm-section .fm-icon img {{ width: 24px; height: 24px; }}
.fm-section h3 {{ margin: 0; padding: 0; font-size: 1.35rem; font-weight: 700; color: {NAVY}; }}
.fm-section p {{ margin: .1rem 0 0 0; color: #5b6b80; font-size: .92rem; }}

/* Cards: each has its own colour, set per card as --c (main) and --t (tint). */
@keyframes fmRise {{ from {{ opacity: 0; transform: translateY(18px); }}
  to {{ opacity: 1; transform: none; }} }}
@keyframes fmPulse {{ 0% {{ box-shadow: 0 0 0 0 var(--ring); }}
  70% {{ box-shadow: 0 0 0 10px transparent; }} 100% {{ box-shadow: 0 0 0 0 transparent; }} }}
@keyframes fmFlow {{ to {{ background-position: 24px 0; }} }}
@keyframes fmFloat {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-4px); }} }}
@keyframes fmShine {{ from {{ transform: translateX(-120%) skewX(-20deg); }}
  to {{ transform: translateX(260%) skewX(-20deg); }} }}

.fm-grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(3, 1fr);
  margin: .6rem 0 1rem 0; }}
@media (max-width: 900px) {{ .fm-grid {{ grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }} }}
.fm-card {{ position: relative; overflow: hidden; background: linear-gradient(180deg, var(--t), #fff 70%);
  border: 1px solid #e3e9f1; border-radius: 18px; padding: 1.3rem 1.2rem 1.2rem;
  box-shadow: 0 4px 14px rgba(11,31,58,.05);
  animation: fmRise .6s cubic-bezier(.2,.7,.2,1) both; animation-delay: var(--d);
  transition: transform .25s, box-shadow .25s, border-color .25s; }}
.fm-card::before {{ content: ""; position: absolute; inset: 0 0 auto 0; height: 4px;
  background: linear-gradient(90deg, var(--c), color-mix(in srgb, var(--c) 40%, #fff)); }}
.fm-card::after {{ content: ""; position: absolute; top: 0; left: 0; width: 40%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.65), transparent);
  transform: translateX(-120%) skewX(-20deg); pointer-events: none; }}
.fm-card:hover {{ transform: translateY(-6px); border-color: var(--c);
  box-shadow: 0 16px 32px color-mix(in srgb, var(--c) 22%, transparent); }}
.fm-card:hover::after {{ animation: fmShine .9s ease; }}
.fm-card .fm-icon {{ width: 48px; height: 48px; border-radius: 14px; display: grid;
  place-items: center; background: #fff; margin-bottom: .8rem;
  box-shadow: 0 6px 14px color-mix(in srgb, var(--c) 25%, transparent);
  transition: transform .3s; }}
.fm-card:hover .fm-icon {{ transform: rotate(-8deg) scale(1.1); }}
.fm-card .fm-icon img {{ width: 26px; height: 26px; }}
.fm-card h4 {{ margin: 0 0 .35rem 0; padding: 0; font-size: 1.04rem; color: {NAVY}; }}
.fm-card p {{ margin: 0; color: #5b6b80; font-size: .9rem; line-height: 1.45; }}

/* How it works */
.fm-steps {{ display: grid; grid-template-columns: repeat(5, 1fr);
  gap: 1.4rem; margin: 1rem 0 1rem 0; }}
.fm-step {{ position: relative; background: linear-gradient(160deg, var(--t), #fff 75%);
  border: 1px solid color-mix(in srgb, var(--c) 30%, #e3e9f1); border-radius: 18px;
  padding: 1.5rem 1rem 1.1rem;
  animation: fmRise .6s cubic-bezier(.2,.7,.2,1) both; animation-delay: var(--d);
  transition: transform .25s, box-shadow .25s; }}
.fm-step:hover {{ transform: translateY(-6px);
  box-shadow: 0 16px 30px color-mix(in srgb, var(--c) 25%, transparent); }}
.fm-step .fm-num {{ --ring: color-mix(in srgb, var(--c) 55%, transparent);
  position: absolute; top: -13px; left: 14px; width: 28px; height: 28px; border-radius: 50%;
  background: var(--c); color: #fff; font-size: .82rem; font-weight: 800;
  display: grid; place-items: center; animation: fmPulse 2.4s infinite;
  animation-delay: var(--d); }}
/* animated dashed arrow to the next step */
.fm-step:not(:last-child)::after {{ content: ""; position: absolute; top: 50%; right: -1.3rem;
  width: 1.2rem; height: 3px; border-radius: 2px;
  background: linear-gradient(90deg, var(--c) 50%, transparent 50%) 0 0 / 8px 3px repeat-x;
  animation: fmFlow .8s linear infinite; }}
.fm-step .fm-sicon {{ width: 52px; height: 52px; border-radius: 16px; display: grid;
  place-items: center; background: #fff; margin: .2rem 0 .7rem 0;
  box-shadow: 0 8px 18px color-mix(in srgb, var(--c) 25%, transparent);
  animation: fmFloat 3.2s ease-in-out infinite; animation-delay: var(--d); }}
.fm-step .fm-sicon img {{ width: 28px; height: 28px; }}
.fm-step h4 {{ margin: 0 0 .25rem 0; padding: 0; font-size: 1rem; color: var(--c); }}
.fm-step p {{ margin: 0; color: #5b6b80; font-size: .85rem; line-height: 1.4; }}
@media (max-width: 1000px) {{ .fm-steps {{ grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }}
  .fm-step::after {{ display: none; }} }}
@media (prefers-reduced-motion: reduce) {{
  .fm-card, .fm-step, .fm-step .fm-num, .fm-step .fm-sicon, .fm-step::after, .fm-table tbody tr,
  .fm-bar .fill, .fm-pill-risk::before, .fm-stage, .fm-stage .ic, .fm-stage::after,
  .fm-stage::before, .fm-range .band, .fm-range .mark {{ animation: none !important; }} }}

/* Demo bundle buttons: one colour per scenario (Streamlit tags each widget with an st-key- class) */
.st-key-demo-comp2_failure_building button, .st-key-demo-two_machines_at_risk button,
.st-key-demo-healthy_fleet button {{ color: #fff !important; border: none !important;
  font-weight: 700; padding: .65rem 1rem; border-radius: 12px;
  transition: transform .2s, box-shadow .2s, filter .2s; }}
.st-key-demo-comp2_failure_building button {{ background: linear-gradient(135deg, #b4232a, {RED});
  box-shadow: 0 8px 18px rgba(229,72,77,.35); }}
.st-key-demo-two_machines_at_risk button {{ background: linear-gradient(135deg, #b45309, {AMBER});
  box-shadow: 0 8px 18px rgba(245,158,11,.35); }}
.st-key-demo-healthy_fleet button {{ background: linear-gradient(135deg, #15803d, #22c55e);
  box-shadow: 0 8px 18px rgba(34,197,94,.35); }}
.st-key-demo-comp2_failure_building button:hover, .st-key-demo-two_machines_at_risk button:hover,
.st-key-demo-healthy_fleet button:hover {{ transform: translateY(-3px); filter: brightness(1.07); }}
.st-key-demo-comp2_failure_building button p, .st-key-demo-two_machines_at_risk button p,
.st-key-demo-healthy_fleet button p {{ color: #fff !important; }}
/* matching outlined download buttons */
.st-key-download-comp2_failure_building button {{ --dc: {RED}; }}
.st-key-download-two_machines_at_risk button {{ --dc: #d97706; }}
.st-key-download-healthy_fleet button {{ --dc: #16a34a; }}
[class*="st-key-download-"] button {{ background: #fff !important; border: 1.5px solid var(--dc) !important;
  border-radius: 12px; transition: background .2s, transform .2s; }}
[class*="st-key-download-"] button p {{ color: var(--dc) !important; font-weight: 600; }}
[class*="st-key-download-"] button:hover {{
  background: color-mix(in srgb, var(--dc) 10%, #fff) !important; transform: translateY(-2px); }}

/* Risk table */
.fm-table-wrap {{ overflow-x: auto; border: 1px solid #e3e9f1; border-radius: 16px;
  box-shadow: 0 6px 18px rgba(11,31,58,.06); margin: .4rem 0 .8rem 0; }}
.fm-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: .88rem;
  min-width: 760px; }}
.fm-table th {{ position: sticky; top: 0; background: {NAVY}; color: #fff; text-align: left;
  font-weight: 600; font-size: .78rem; letter-spacing: .04em; text-transform: uppercase;
  padding: .65rem .6rem; white-space: nowrap; }}
.fm-table td {{ padding: .6rem .6rem; white-space: nowrap; border-top: 1px solid #edf1f6; vertical-align: middle;
  color: {NAVY}; }}
.fm-table tbody tr {{ transition: background .15s; animation: fmRise .4s ease both; }}
.fm-table tbody tr:hover {{ background: #f4f8fc; }}
.fm-table tr.high {{ background: linear-gradient(90deg, #fdecec, #fff 60%); }}
.fm-table tr.high td:first-child {{ box-shadow: inset 4px 0 0 {RED}; }}
.fm-table tr.medium {{ background: linear-gradient(90deg, #fdf3e3, #fff 60%); }}
.fm-table tr.medium td:first-child {{ box-shadow: inset 4px 0 0 {AMBER}; }}
.fm-pill-risk {{ display: inline-flex; align-items: center; gap: .35rem; padding: .22rem .65rem;
  border-radius: 999px; font-size: .78rem; font-weight: 700; white-space: nowrap; }}
.fm-pill-risk::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }}
.fm-pill-risk.high {{ background: #fde2e3; color: #b4232a; }}
.fm-pill-risk.high::before {{ animation: fmPulse 1.6s infinite; --ring: rgba(229,72,77,.6); }}
.fm-pill-risk.medium {{ background: #fdecc8; color: #92400e; }}
.fm-pill-risk.low {{ background: #dcfce7; color: #15803d; }}
.fm-mid {{ font-weight: 700; }}
.fm-part {{ display: inline-block; padding: .15rem .5rem; border-radius: 6px; background: #eef2f7;
  font-family: ui-monospace, monospace; font-size: .8rem; }}
.fm-bar {{ width: 64px; }}
.fm-bar .track {{ height: 7px; border-radius: 99px; background: #edf1f6; overflow: hidden; }}
.fm-bar .fill {{ height: 100%; border-radius: 99px; background: var(--bc);
  animation: fmGrow .9s cubic-bezier(.2,.7,.2,1) both; transform-origin: left; }}
.fm-bar .num {{ font-size: .75rem; color: #5b6b80; margin-top: .2rem; font-variant-numeric: tabular-nums; }}
.fm-bar.hot .num {{ color: #b4232a; font-weight: 700; }}
@keyframes fmGrow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
.fm-table td.fm-why {{ color: #5b6b80; font-size: .8rem; line-height: 1.35; white-space: normal;
  min-width: 200px; }}

/* Cascade chain: machine -> quality -> inventory -> orders -> money */
.fm-chain {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 1.6rem;
  margin: .6rem 0 1rem 0; }}
.fm-stage {{ position: relative; border-radius: 18px; padding: 1rem 1rem 1.05rem;
  background: linear-gradient(170deg, var(--t), #fff 70%);
  border: 1px solid color-mix(in srgb, var(--c) 30%, #e3e9f1);
  box-shadow: 0 6px 16px rgba(11,31,58,.06);
  animation: fmRise .55s cubic-bezier(.2,.7,.2,1) both; animation-delay: var(--d);
  transition: transform .25s, box-shadow .25s; }}
.fm-stage:hover {{ transform: translateY(-5px);
  box-shadow: 0 14px 28px color-mix(in srgb, var(--c) 25%, transparent); }}
.fm-stage .head {{ display: flex; align-items: center; gap: .55rem; margin-bottom: .7rem; }}
.fm-stage .ic {{ position: relative; flex: none; width: 38px; height: 38px; border-radius: 12px;
  display: grid; place-items: center; background: #fff;
  box-shadow: 0 5px 12px color-mix(in srgb, var(--c) 25%, transparent); }}
.fm-stage .ic img {{ width: 21px; height: 21px; }}
.fm-stage.alarm .ic {{ --ring: color-mix(in srgb, var(--c) 60%, transparent);
  animation: fmPulse 1.8s infinite; }}
.fm-stage .ttl {{ font-weight: 700; color: {NAVY}; font-size: .95rem; line-height: 1.2; }}
.fm-stage .tag {{ display: inline-block; margin-top: .15rem; font-size: .66rem; font-weight: 700;
  color: #b45309; background: #fef3c7; border-radius: 6px; padding: .05rem .4rem; }}
.fm-stage .lbl {{ font-size: .75rem; color: #5b6b80; text-transform: uppercase;
  letter-spacing: .04em; }}
.fm-stage .val {{ font-size: 1.85rem; font-weight: 800; color: var(--c); line-height: 1.15;
  margin: .1rem 0 .55rem 0; font-variant-numeric: tabular-nums; }}
.fm-stage .note {{ font-size: .8rem; color: #5b6b80; line-height: 1.4; margin-top: .25rem; }}
.fm-stage .note b {{ color: {NAVY}; }}
/* animated connector: a dashed line with a travelling dot */
.fm-stage:not(:last-child)::after {{ content: ""; position: absolute; top: 34px; right: -1.55rem;
  width: 1.5rem; height: 3px; border-radius: 2px;
  background: linear-gradient(90deg, #94a3b8 50%, transparent 50%) 0 0 / 7px 3px repeat-x;
  animation: fmFlow .7s linear infinite; }}
.fm-stage:not(:last-child)::before {{ content: ""; position: absolute; top: 30px; right: -1.55rem;
  width: 10px; height: 10px; border-radius: 50%; background: var(--c);
  box-shadow: 0 0 8px var(--c); animation: fmTravel 1.6s ease-in-out infinite;
  animation-delay: var(--d); z-index: 1; }}
@keyframes fmTravel {{ 0% {{ transform: translateX(-4px); opacity: 0; }}
  20% {{ opacity: 1; }} 80% {{ opacity: 1; }} 100% {{ transform: translateX(22px); opacity: 0; }} }}
/* the loss stage is the destination: filled */
.fm-stage.money {{ background: linear-gradient(150deg, {NAVY}, #1d4677); border: none; color: #fff; }}
.fm-stage.money .ttl, .fm-stage.money .val {{ color: #fff; }}
.fm-stage.money .lbl, .fm-stage.money .note {{ color: #c9d6e8; }}
.fm-stage.money .note b {{ color: #fff; }}
.fm-range {{ margin-top: .7rem; }}
.fm-range .scale {{ position: relative; height: 10px; border-radius: 99px;
  background: rgba(255,255,255,.15); }}
.fm-range .band {{ position: absolute; top: 0; bottom: 0; border-radius: 99px;
  background: linear-gradient(90deg, #7ee7e6, {AMBER});
  animation: fmGrow .9s cubic-bezier(.2,.7,.2,1) both; animation-delay: .5s;
  transform-origin: left; }}
.fm-range .mark {{ position: absolute; top: -4px; width: 4px; height: 18px; margin-left: -2px;
  border-radius: 2px; background: #fff; box-shadow: 0 0 6px rgba(255,255,255,.8);
  animation: fmRise .5s ease both; animation-delay: 1.1s; }}
.fm-range .cap {{ display: flex; justify-content: space-between; font-size: .72rem;
  color: #c9d6e8; margin-top: .35rem; font-variant-numeric: tabular-nums; }}
@media (max-width: 1100px) {{ .fm-chain {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
  .fm-stage::after, .fm-stage::before {{ display: none; }} }}

/* KPI tiles */
.fm-kpis {{ display: grid; gap: .9rem; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  margin: .6rem 0 1.2rem 0; }}
.fm-kpi {{ border-radius: 16px; padding: 1rem 1.15rem; color: #fff;
  box-shadow: 0 8px 20px rgba(11,31,58,.12); }}
.fm-kpi .fm-label {{ font-size: .8rem; opacity: .85; text-transform: uppercase;
  letter-spacing: .05em; }}
.fm-kpi .fm-value {{ font-size: 1.9rem; font-weight: 800; line-height: 1.15; margin-top: .2rem; }}
.fm-kpi .fm-note {{ font-size: .82rem; opacity: .85; margin-top: .15rem; }}
.fm-kpi.navy {{ background: linear-gradient(135deg, {NAVY}, #1d4677); }}
.fm-kpi.red {{ background: linear-gradient(135deg, #b4232a, {RED}); }}
.fm-kpi.amber {{ background: linear-gradient(135deg, #b45309, {AMBER}); }}
.fm-kpi.teal {{ background: linear-gradient(135deg, #0b7d7c, {TEAL}); }}

/* Native widgets */
div[data-testid="stFileUploader"] section {{ border-radius: 14px; border: 2px dashed #b9d7e3;
  background: #f6fbfd; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 10px; font-weight: 600; }}
div[data-testid="stExpander"] details {{ border-radius: 12px; }}
section[data-testid="stSidebar"] {{ background: #f4f7fb; }}

/* Footer */
.fm-footer {{ margin-top: 3rem; padding: 1.4rem 0 .4rem; border-top: 1px solid #e3e9f1;
  color: #7a8899; font-size: .85rem; display: flex; justify-content: space-between;
  flex-wrap: wrap; gap: .6rem; }}
</style>
"""

# --- icons (24×24, stroke style) ------------------------------------------------------------

def _img(svg: str, alt: str = "", cls: str = "") -> str:
    """st.html strips inline <svg>, so SVGs are embedded as data-URI images."""
    data = base64.b64encode(svg.strip().encode()).decode()
    return (f'<img class="{cls}" alt="{escape(alt)}" '
            f'src="data:image/svg+xml;base64,{data}"/>')


def _icon(paths: str, color: str = TEAL) -> str:
    return _img(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
                f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
                f'stroke-linejoin="round">{paths}</svg>')


ICONS = {
    "upload": '<path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    "pulse": '<path d="M3 12h4l2.5-6 4 12 2.5-6H21"/>',
    "cascade": '<rect x="3" y="3" width="6" height="5" rx="1"/><rect x="15" y="9.5" width="6" height="5" rx="1"/><rect x="3" y="16" width="6" height="5" rx="1"/><path d="M9 5.5h3v13H9"/><path d="M12 12h3"/>',
    "sliders": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "spark": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4"/><path d="m6 6 2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/>',
    "bell": '<path d="M6 8a6 6 0 1 1 12 0c0 7 3 8 3 8H3s3-1 3-8"/><path d="M10 20a2 2 0 0 0 4 0"/>',
    "gear": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    "drop": '<path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z"/><path d="M9.5 14.5a2.5 2.5 0 0 0 2.5 2.5"/>',
    "box": '<path d="M21 8 12 3 3 8v8l9 5 9-5z"/><path d="m3 8 9 5 9-5M12 13v8"/>',
    "truck": '<path d="M3 6h11v10H3zM14 10h4l3 3v3h-7"/><circle cx="7" cy="17.5" r="1.8"/><circle cx="17" cy="17.5" r="1.8"/>',
    "coin": '<circle cx="12" cy="12" r="9"/><path d="M15 9.5c-.5-1-1.6-1.5-3-1.5-1.7 0-3 .8-3 2s1.3 1.7 3 2 3 .8 3 2-1.3 2-3 2c-1.4 0-2.5-.5-3-1.5M12 6.5v11"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/>',
    "chat": '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/><path d="M8.5 11h7M8.5 14h4"/>',
    "shield": '<path d="M12 3 4.5 6v6c0 4.5 3.2 7.8 7.5 9 4.3-1.2 7.5-4.5 7.5-9V6z"/><path d="m9 12 2 2 4-4"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r="1"/><circle cx="4.5" cy="12" r="1"/><circle cx="4.5" cy="18" r="1"/>',
}


def icon(name: str, color: str = TEAL) -> str:
    return _icon(ICONS[name], color)


# --- hero illustration ----------------------------------------------------------------------

HERO_ART = f"""
<svg viewBox="0 0 520 340" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="A factory line with a dashboard flagging one machine at risk">
  <defs>
    <linearGradient id="fmPanel" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffffff" stop-opacity=".14"/>
      <stop offset="1" stop-color="#ffffff" stop-opacity=".05"/>
    </linearGradient>
    <linearGradient id="fmMachine" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#2b5c8f"/><stop offset="1" stop-color="#1a3d66"/>
    </linearGradient>
    <linearGradient id="fmRisk" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#f87171"/><stop offset="1" stop-color="#b4232a"/>
    </linearGradient>
  </defs>
  <!-- floor grid -->
  <g stroke="#7ee7e6" stroke-opacity=".12">
    <path d="M0 300h520M0 320h520M40 280 0 340M140 280 110 340M240 280 230 340M340 280 350 340M440 280 470 340"/>
  </g>
  <!-- conveyor -->
  <rect x="20" y="262" width="480" height="14" rx="7" fill="#0d2a4d" stroke="#7ee7e6" stroke-opacity=".35"/>
  <g fill="#7ee7e6" fill-opacity=".55">
    <circle cx="50" cy="269" r="3"/><circle cx="110" cy="269" r="3"/><circle cx="170" cy="269" r="3"/>
    <circle cx="230" cy="269" r="3"/><circle cx="290" cy="269" r="3"/><circle cx="350" cy="269" r="3"/>
    <circle cx="410" cy="269" r="3"/><circle cx="470" cy="269" r="3"/>
  </g>
  <g fill="{AMBER}" fill-opacity=".9">
    <rect x="70" y="246" width="18" height="16" rx="3"/><rect x="200" y="246" width="18" height="16" rx="3"/>
    <rect x="390" y="246" width="18" height="16" rx="3"/>
  </g>
  <!-- machines -->
  <g>
    <rect x="40" y="170" width="92" height="88" rx="10" fill="url(#fmMachine)" stroke="#7ee7e6" stroke-opacity=".4"/>
    <rect x="54" y="184" width="64" height="26" rx="5" fill="#0b1f3a"/>
    <path d="M58 200h10l4-8 6 14 4-6h32" stroke="{TEAL}" stroke-width="2" fill="none"/>
    <circle cx="66" cy="232" r="9" fill="none" stroke="#9fb6d3" stroke-width="3"/>
    <circle cx="104" cy="232" r="9" fill="none" stroke="#9fb6d3" stroke-width="3"/>
    <circle cx="124" cy="176" r="6" fill="{TEAL}"/>
  </g>
  <g>
    <rect x="200" y="150" width="110" height="108" rx="10" fill="url(#fmRisk)" stroke="#fecaca" stroke-opacity=".6"/>
    <rect x="216" y="166" width="78" height="30" rx="5" fill="#3b0d10"/>
    <path d="M220 184h12l5-10 7 18 5-12 6 8 6-14 5 10h15" stroke="#fecaca" stroke-width="2" fill="none"/>
    <circle cx="236" cy="226" r="12" fill="none" stroke="#fecaca" stroke-width="3"/>
    <circle cx="274" cy="226" r="12" fill="none" stroke="#fecaca" stroke-width="3"/>
    <circle cx="302" cy="156" r="7" fill="#fff"/>
    <circle cx="302" cy="156" r="14" fill="none" stroke="#fff" stroke-opacity=".5">
      <animate attributeName="r" values="8;20;8" dur="2.2s" repeatCount="indefinite"/>
      <animate attributeName="stroke-opacity" values=".7;0;.7" dur="2.2s" repeatCount="indefinite"/>
    </circle>
  </g>
  <g>
    <rect x="378" y="170" width="92" height="88" rx="10" fill="url(#fmMachine)" stroke="#7ee7e6" stroke-opacity=".4"/>
    <rect x="392" y="184" width="64" height="26" rx="5" fill="#0b1f3a"/>
    <path d="M396 198h14l4-6 6 10 4-4h28" stroke="{TEAL}" stroke-width="2" fill="none"/>
    <circle cx="404" cy="232" r="9" fill="none" stroke="#9fb6d3" stroke-width="3"/>
    <circle cx="442" cy="232" r="9" fill="none" stroke="#9fb6d3" stroke-width="3"/>
    <circle cx="462" cy="176" r="6" fill="{TEAL}"/>
  </g>
  <!-- link from risky machine to dashboard -->
  <path d="M255 150 C 255 110, 300 96, 330 92" stroke="#fecaca" stroke-width="2" stroke-dasharray="5 5" fill="none"/>
  <!-- dashboard card -->
  <g>
    <rect x="300" y="14" width="200" height="124" rx="14" fill="url(#fmPanel)" stroke="#ffffff" stroke-opacity=".25"/>
    <rect x="316" y="30" width="80" height="8" rx="4" fill="#ffffff" fill-opacity=".55"/>
    <rect x="316" y="46" width="120" height="6" rx="3" fill="#ffffff" fill-opacity=".25"/>
    <rect x="316" y="66" width="54" height="56" rx="8" fill="#ffffff" fill-opacity=".08"/>
    <rect x="326" y="96" width="8" height="18" rx="2" fill="{TEAL}"/>
    <rect x="338" y="86" width="8" height="28" rx="2" fill="{TEAL}"/>
    <rect x="350" y="76" width="8" height="38" rx="2" fill="{RED}"/>
    <rect x="380" y="66" width="104" height="24" rx="8" fill="{RED}" fill-opacity=".9"/>
    <circle cx="393" cy="78" r="5" fill="#fff"/>
    <rect x="404" y="74" width="66" height="8" rx="4" fill="#fff" fill-opacity=".9"/>
    <rect x="380" y="98" width="104" height="24" rx="8" fill="{TEAL}" fill-opacity=".9"/>
    <path d="m389 110 4 4 7-8" stroke="#fff" stroke-width="2.2" fill="none" stroke-linecap="round"/>
    <rect x="406" y="106" width="64" height="8" rx="4" fill="#fff" fill-opacity=".9"/>
  </g>
  <!-- AI spark -->
  <g transform="translate(36 40)">
    <circle r="30" fill="#7ee7e6" fill-opacity=".12"/>
    <path d="M0 -16 4 -4 16 0 4 4 0 16 -4 4 -16 0 -4 -4z" fill="#7ee7e6"/>
  </g>
</svg>
"""


# --- blocks ---------------------------------------------------------------------------------

def inject_css() -> None:
    st.html(CSS)


def hero() -> None:
    pills = ["Failure risk in 24h", "Quality & scrap", "Inventory & orders", "Best action",
             "Gemini explains", "Slack & email alerts"]
    st.html(f"""
<div class="fm-hero">
  <div>
    <div class="fm-eyebrow">AI predictive maintenance</div>
    <h1>Stop machine failures <span>before they cost you</span></h1>
    <p>FactoryMind predicts which machines will fail in the next 24 hours, traces the
       cascade to quality, inventory and customer orders, and finds the smallest action that
       prevents the largest loss.</p>
    <div class="fm-pills">{''.join(f'<span class="fm-pill">{escape(p)}</span>' for p in pills)}</div>
  </div>
  <div>{_img(HERO_ART, "A factory line with a dashboard flagging one machine at risk", "fm-art")}</div>
</div>""")


def section(icon_name: str, title: str, subtitle: str = "", color: str = TEAL) -> None:
    sub = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.html(f"""<div class="fm-section"><div class="fm-icon">{icon(icon_name, color)}</div>
<div><h3>{escape(title)}</h3>{sub}</div></div>""")


STEPS = [
    ("upload", "Upload", "Telemetry, errors, maintenance, schedule, inventory and orders, or a demo."),
    ("pulse", "Predict", "A boosted-tree model scores each component's 24h failure risk."),
    ("cascade", "Trace the cascade", "Quality, stock, late orders and money, hour by hour."),
    ("sliders", "Simulate actions", "Slow down, replace, service or reroute, each re-simulated."),
    ("spark", "Recommend", "Gemini explains the best action, quoting only verified figures."),
]

FEATURES = [
    ("target", "Component-level risk", "Which part is failing, how likely, and the sensor signals behind it."),
    ("drop", "Quality projection", "Defect rate and scrap cost if the machine keeps running."),
    ("box", "Inventory impact", "When the product falls below safety stock, with and without a failure."),
    ("truck", "Late orders", "Which customers wait, for how long, and the penalties."),
    ("coin", "Loss with a range", "Expected loss plus a Monte Carlo P10/P50/P90 range."),
    ("bell", "Alerts", "Send at-risk machines and the best action to Slack or email."),
]


# (main, tint) per card, in order
PALETTE = [("#0ea5a4", "#e6f7f7"), ("#2563eb", "#e8effd"), ("#7c3aed", "#f1ebfd"),
           ("#d97706", "#fdf3e3"), ("#e11d48", "#fde9ee"), ("#16a34a", "#e7f6ec")]


def _style(i: int) -> str:
    c, t = PALETTE[i % len(PALETTE)]
    return f'style="--c:{c};--t:{t};--d:{i * 0.09:.2f}s"'


def how_it_works() -> None:
    section("list", "How it works", "From raw sensor data to a costed decision in five steps.")
    st.html('<div class="fm-steps">' + "".join(
        f'<div class="fm-step" {_style(n)}><div class="fm-num">{n + 1}</div>'
        f'<div class="fm-sicon">{icon(i, PALETTE[n % len(PALETTE)][0])}</div>'
        f'<h4>{escape(t)}</h4><p>{escape(d)}</p></div>'
        for n, (i, t, d) in enumerate(STEPS)) + "</div>")


def features() -> None:
    section("shield", "What you get", "Every number is simulated, explained and editable.")
    cards(FEATURES)


def cards(items) -> None:
    st.html('<div class="fm-grid">' + "".join(
        f'<div class="fm-card" {_style(n)}><div class="fm-icon">'
        f'{icon(i, PALETTE[n % len(PALETTE)][0])}</div><h4>{escape(t)}</h4>'
        f'<p>{escape(d)}</p></div>' for n, (i, t, d) in enumerate(items)) + "</div>")


def kpis(tiles: list[tuple[str, str, str, str]]) -> None:
    """tiles: (colour class, label, value, note)."""
    st.html('<div class="fm-kpis">' + "".join(
        f'<div class="fm-kpi {c}"><div class="fm-label">{escape(label)}</div>'
        f'<div class="fm-value">{escape(value)}</div><div class="fm-note">{escape(note)}</div>'
        f'</div>' for c, label, value, note in tiles) + "</div>")


def _bar(p: float) -> str:
    """A component's 24h failure chance as a coloured bar."""
    colour = RED if p >= 0.5 else AMBER if p >= 0.1 else TEAL
    text = ">99%" if p > 0.99 else f"{p * 100:.1f}%"
    hot = " hot" if p >= 0.5 else ""
    width = max(p * 100, 1.5)  # keep a sliver visible for tiny risks
    return (f'<div class="fm-bar{hot}" style="--bc:{colour}"><div class="track">'
            f'<div class="fill" style="width:{width:.1f}%"></div></div>'
            f'<div class="num">{text}</div></div>')


def risk_table(machines, components: list[str]) -> None:
    """The fleet risk table: riskiest first, a coloured row and pill per risk level."""
    labels = {"high": "High", "medium": "Medium", "low": "Low"}
    head = ("<tr><th>Risk</th><th>Machine</th><th>Model</th><th>Age</th><th>Riskiest part</th>"
            + "".join(f"<th>{escape(c)}</th>" for c in components) + "<th>Why</th></tr>")
    rows = []
    for n, m in enumerate(machines):
        low = m.risk_level == "low"
        # For low-risk machines the "riskiest" part and its reasons are noise.
        part = "—" if low else f'<span class="fm-part">{escape(m.top_component)}</span>'
        why = "" if low else escape(" · ".join(d.description for d in m.top_drivers))
        rows.append(
            f'<tr class="{m.risk_level}" style="animation-delay:{n * 0.04:.2f}s">'
            f'<td><span class="fm-pill-risk {m.risk_level}">{labels[m.risk_level]}</span></td>'
            f'<td class="fm-mid">{m.machine_id}</td><td>{escape(m.model)}</td>'
            f'<td>{m.age} yrs</td><td>{part}</td>'
            + "".join(f"<td>{_bar(m.component_risk[c])}</td>" for c in components)
            + f'<td class="fm-why">{why}</td></tr>')
    st.html(f'<div class="fm-table-wrap"><table class="fm-table"><thead>{head}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


TONES = {"red": (RED, "#fde9ee"), "amber": ("#d97706", "#fdf3e3"),
         "teal": (TEAL, "#e6f7f7"), "navy": (NAVY, "#e8effd")}


def money(x: float) -> str:
    return f"${x:,.0f}"


def cascade_chain(stages: list[dict], loss_range: dict | None = None) -> None:
    """The five linked cascade cards. Each stage: icon, title, label, value, notes (HTML
    strings, already escaped), tone (red|amber|teal|navy), and optionally tag and alarm. The last
    stage is drawn as the filled 'money' card, with the Monte Carlo range bar when given:
    {lo, mid, hi} in USD."""
    cards = []
    for n, st_ in enumerate(stages):
        c, t = TONES[st_["tone"]]
        last = n == len(stages) - 1
        classes = "fm-stage" + (" money" if last else "") + (" alarm" if st_.get("alarm") else "")
        tag = f'<div class="tag">{escape(st_["tag"])}</div>' if st_.get("tag") else ""
        notes = "".join(f'<div class="note">{note}</div>' for note in st_["notes"])
        extra = ""
        if last and loss_range:
            # Zoom the scale onto the range so its spread is visible.
            low, high = loss_range["lo"] * 0.85, loss_range["hi"] * 1.05
            span = (high - low) or 1

            def pos(x):
                return 100 * (x - low) / span

            left = pos(loss_range["lo"])
            width = max(pos(loss_range["hi"]) - left, 1.5)
            mid = pos(loss_range["mid"])
            extra = (f'<div class="fm-range"><div class="scale">'
                     f'<div class="band" style="left:{left:.1f}%;width:{width:.1f}%"></div>'
                     f'<div class="mark" style="left:{mid:.1f}%"></div></div>'
                     f'<div class="cap"><span>P10 {money(loss_range["lo"])}</span>'
                     f'<span>P50 {money(loss_range["mid"])}</span>'
                     f'<span>P90 {money(loss_range["hi"])}</span></div></div>')
        icon_colour = "#ffffff" if last else c
        ic_bg = ' style="background:rgba(255,255,255,.12)"' if last else ""
        cards.append(
            f'<div class="{classes}" style="--c:{c};--t:{t};--d:{n * 0.12:.2f}s">'
            f'<div class="head"><div class="ic"{ic_bg}>{icon(st_["icon"], icon_colour)}</div>'
            f'<div><div class="ttl">{escape(st_["title"])}</div>{tag}</div></div>'
            f'<div class="lbl">{escape(st_["label"])}</div>'
            f'<div class="val">{escape(st_["value"])}</div>{notes}{extra}</div>')
    st.html(f'<div class="fm-chain">{"".join(cards)}</div>')


def footer() -> None:
    st.html("""<div class="fm-footer"><span>🏭 <b>FactoryMind</b> · decision intelligence for
the factory floor</span><span>Sensor data: Azure Predictive Maintenance · Built on Google Cloud,
BigQuery ML and Gemini</span></div>""")
