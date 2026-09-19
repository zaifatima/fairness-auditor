# app.py
from pathlib import Path
import os
import time
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# SECURITY: Never collect or display the API key in the Streamlit UI.
# The dashboard reads it from openai_key.txt (local, gitignored) or OPENAI_API_KEY.

st.set_page_config(
    page_title="Proactive Fairness Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Styling
# ---------------------------
st.markdown("""
<style>
:root{
  --ink:#0f172a; --muted:#64748b; --line:#e4e9f1;
  --blue:#2f80ed; --purple:#7c5cff; --teal:#0ea5a5; --orange:#d99017; --green:#16a37a;
}
html,body,[class*="css"]{
  font-family:"Aptos","Segoe UI",Inter,system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
  color:var(--ink);
}
.stApp{
  background:
    radial-gradient(circle at 6% 0%,rgba(47,128,237,.08),transparent 23%),
    radial-gradient(circle at 95% 3%,rgba(124,92,255,.07),transparent 21%),
    linear-gradient(180deg,#fbfcfe 0%,#f1f4f8 100%);
}
.block-container{max-width:1540px;padding-top:.8rem;padding-bottom:2.4rem;}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#091428 0%,#102640 100%);
  border-right:1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] *{color:#eef4ff !important;}
[data-testid="stSidebar"] .stRadio label{font-weight:650;}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.08);}
[data-testid="stSidebar"] [data-baseweb="select"]>div{
  background:#fff !important;border:1px solid #cbd5e1 !important;border-radius:12px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#15243d !important;}
[data-testid="stSidebar"] [data-baseweb="select"] svg{color:#53657d !important;fill:#53657d !important;}
button{border-radius:12px !important;font-weight:760 !important;}

.hero{
  position:relative;overflow:hidden;padding:32px 38px 29px;border-radius:28px;color:#fff;
  background:
    radial-gradient(circle at 90% 25%,rgba(46,201,255,.24),transparent 18%),
    radial-gradient(circle at 75% 120%,rgba(124,92,255,.28),transparent 34%),
    linear-gradient(130deg,#09162d 0%,#17355d 55%,#2f80ed 100%);
  box-shadow:0 22px 62px rgba(9,29,57,.22);margin-bottom:15px;animation:heroIn .65s ease-out;
}
.hero:before,.hero:after{
  content:"";position:absolute;border-radius:50%;border:1px solid rgba(255,255,255,.13);pointer-events:none;
}
.hero:before{width:290px;height:290px;right:-90px;top:-150px;animation:floatRing 14s linear infinite;}
.hero:after{width:460px;height:460px;right:-180px;top:-260px;animation:floatRing 20s linear infinite reverse;}
.hero-inner{position:relative;z-index:2;max-width:1050px;}
.eyebrow{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:#b9d7ff;font-weight:850;margin-bottom:8px;}
.hero h1{margin:0 0 8px;font-size:2.55rem;letter-spacing:-.045em;line-height:1.04;}
.hero p{margin:0;max-width:1080px;color:#f3f7ff;opacity:.94;line-height:1.5;font-size:.98rem;}

.dashboard-band{
  display:flex;align-items:center;justify-content:space-between;gap:18px;padding:12px 15px;margin:0 0 20px;
  border-radius:15px;background:linear-gradient(100deg,#112545 0%,#1f416f 100%);color:#fff;
  box-shadow:0 8px 25px rgba(18,39,70,.12);
}
.dashboard-band .title{font-weight:900;font-size:.88rem;}
.dashboard-band .sub{font-size:.70rem;color:#cfddf0;margin-top:2px;}
.dashboard-band .signal{padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.12);font-size:.67rem;font-weight:850;white-space:nowrap;}
.live-dot{width:8px;height:8px;border-radius:50%;display:inline-block;background:#53efb3;box-shadow:0 0 0 0 rgba(83,239,179,.55);animation:pulse 2s infinite;margin-right:6px;}

.page-header{
  position:relative;overflow:hidden;padding:14px 18px 13px;margin:0 0 13px;border-radius:19px;
  border:1px solid rgba(228,233,241,.98);background:rgba(255,255,255,.90);
  box-shadow:0 9px 28px rgba(18,38,63,.055);
}
.page-header:after{content:"";position:absolute;left:0;right:0;top:0;height:4px;}
.page-header.blue:after{background:linear-gradient(90deg,#2f80ed,#63adff);}
.page-header.purple:after{background:linear-gradient(90deg,#7957ff,#b18aff);}
.page-header.teal:after{background:linear-gradient(90deg,#0ea5a5,#46d4c5);}
.page-header.orange:after{background:linear-gradient(90deg,#d99017,#f2be67);}
.page-header.green:after{background:linear-gradient(90deg,#16a37a,#55d4ad);}
.page-tag{
  display:inline-block;padding:5px 9px;border-radius:999px;background:#f1f5f9;color:#52647c;
  font-size:.62rem;font-weight:900;letter-spacing:.09em;text-transform:uppercase;margin-bottom:7px;
}
.page-header h2{margin:0;color:#14213d;font-size:1.42rem;letter-spacing:-.03em;font-weight:900;}
.page-header p{margin:5px 0 0;color:#68778d;font-size:.79rem;line-height:1.45;max-width:1100px;}

.section-title{color:var(--ink);font-size:1.16rem;font-weight:900;letter-spacing:-.025em;margin:17px 0 10px;}
.small-note{color:var(--muted);font-size:.82rem;line-height:1.5;}
.kicker{color:#52647c;font-size:.68rem;font-weight:850;text-transform:uppercase;letter-spacing:.105em;}

.glass{
  background:rgba(255,255,255,.84);backdrop-filter:blur(12px);
  border:1px solid rgba(228,233,241,.98);border-radius:20px;box-shadow:0 10px 34px rgba(18,39,70,.065);
}
.card{padding:18px 20px;}

.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:15px 0;}
.metric-card{
  position:relative;overflow:hidden;min-height:205px;padding:18px;border-radius:19px;background:#fff;border:1px solid #e4e9f1;
  box-shadow:0 8px 28px rgba(19,44,76,.055);transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease;
  animation:cardIn .55s ease-out both;
}
.metric-card:nth-child(2){animation-delay:.05s}.metric-card:nth-child(3){animation-delay:.10s}.metric-card:nth-child(4){animation-delay:.15s}
.metric-card:hover{transform:translateY(-5px);box-shadow:0 18px 40px rgba(19,44,76,.11);border-color:#c9d8ee;}
.metric-card:before{content:"";position:absolute;width:90px;height:90px;right:-26px;top:-26px;border-radius:50%;background:radial-gradient(circle,rgba(52,120,246,.12),transparent 66%);}
.metric-label{color:#667085;font-size:.69rem;font-weight:850;text-transform:uppercase;letter-spacing:.075em;}
.metric-value{color:var(--ink);font-size:2rem;font-weight:950;margin-top:7px;letter-spacing:-.045em;}
.metric-help{color:#6a7890;font-size:.75rem;margin-top:6px;line-height:1.4;min-height:34px;}
.metric-badge{display:inline-block;margin-top:8px;padding:5px 9px;border-radius:999px;background:#eef5ff;color:#315f9d;font-size:.66rem;font-weight:850;}
.metric-reference{margin-top:10px;padding:9px 10px;border-radius:11px;background:#f8fafc;border:1px solid #e7ecf3;color:#52627a;font-size:.69rem;line-height:1.35;}
.reference-label{display:block;color:#73829a;font-size:.61rem;text-transform:uppercase;letter-spacing:.08em;font-weight:900;margin-bottom:2px;}
.good-chip,.no-universal{display:inline-block;margin-top:6px;padding:4px 8px;border-radius:999px;font-size:.63rem;font-weight:850;}
.good-chip{background:#eef9f5;border:1px solid #cdeee1;color:#17735e;}
.no-universal{background:#f1f4f8;border:1px solid #e2e7ee;color:#586a82;}
.metric-scale{height:7px;border-radius:999px;background:linear-gradient(90deg,#c8efdf 0%,#f7e6ae 52%,#efb2b2 100%);margin-top:10px;position:relative;}
.metric-scale .needle{position:absolute;top:-4px;width:3px;height:15px;background:#1a2d49;border-radius:3px;box-shadow:0 1px 3px rgba(0,0,0,.18);}

.insight{padding:16px 18px;border-left:4px solid var(--blue);border-radius:14px;background:linear-gradient(90deg,#eef6ff 0%,rgba(239,247,255,.56) 100%);}
.insight strong{color:#17385f;}
.reference-box{margin-top:15px;background:linear-gradient(135deg,#fbfdff 0%,#eef5ff 100%);border:1px solid #dce7f5;border-radius:18px;padding:15px 16px;}
.reference-box-title{color:var(--ink);font-weight:900;font-size:.96rem;margin-bottom:5px;}
.reference-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px;}
.reference-mini{background:#fff;border:1px solid #e3eaf3;border-radius:14px;padding:10px;}
.reference-mini b{color:#203553;display:block;font-size:.74rem;margin-bottom:3px;}
.reference-mini span{color:#63738a;font-size:.69rem;line-height:1.35;}

.arch-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}
.arch-card{padding:19px;border:1px solid #e3e9f2;background:#fff;border-radius:18px;min-height:180px;transition:.22s ease;}
.arch-card:hover{transform:translateY(-4px);box-shadow:0 15px 34px rgba(19,44,76,.08);}
.arch-icon{width:44px;height:44px;border-radius:13px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#e8f3ff,#f0ebff);font-size:1.2rem;margin-bottom:12px;}

.sticky-explain{padding:14px 16px;background:#fff;border:1px solid #e2e8f0;border-radius:16px;box-shadow:0 7px 22px rgba(18,38,63,.05);}
.explain-row{display:grid;grid-template-columns:165px 1fr;gap:16px;align-items:start;padding:10px 0;border-bottom:1px solid #edf0f4;}
.explain-row:last-child{border-bottom:none;}
.explain-row b{color:#243b5a;font-size:.76rem;}
.explain-row span{color:#66758a;font-size:.73rem;line-height:1.45;}

.warn{border-left:4px solid #d99017;background:#fff8e9;padding:13px 15px;border-radius:12px;color:#6d4d09;}
.success{border-left:4px solid #16a37a;background:#effbf7;padding:13px 15px;border-radius:12px;color:#0e5d4d;}
.footer{color:#718096;font-size:.72rem;margin-top:24px;padding-top:14px;border-top:1px solid #dfe5ee;}

@keyframes heroIn{from{opacity:0;transform:translateY(-12px)}to{opacity:1;transform:translateY(0)}}
@keyframes cardIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(83,239,179,.5)}70%{box-shadow:0 0 0 8px rgba(83,239,179,0)}100%{box-shadow:0 0 0 0 rgba(83,239,179,0)}}
@keyframes floatRing{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@media(max-width:1050px){.metric-grid{grid-template-columns:repeat(2,1fr)}.reference-grid{grid-template-columns:repeat(2,1fr)}.arch-grid{grid-template-columns:1fr}}
@media(max-width:720px){.metric-grid,.reference-grid{grid-template-columns:1fr}.hero h1{font-size:2rem}.explain-row{grid-template-columns:1fr;gap:4px}}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Data
# ---------------------------
ROOT = Path(__file__).parent
DATA = ROOT / "data"
temporal = pd.read_csv(DATA / "temporal_audit.csv")
static_perf = pd.read_csv(DATA / "static_performance.csv")
model_info = pd.read_csv(DATA / "model_info.csv")

MODELS = ["NeuMF", "SASRec", "LightGCN"]

# ---------------------------
# Helpers
# ---------------------------
def fit_projection(d, col, horizon=3):
    """Simple historical trend projection for monitoring. Not a trained predictive model."""
    x = d["year"].to_numpy(dtype=float)
    y = d[col].to_numpy(dtype=float)
    if len(x) < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    last = int(d["year"].max())
    future = np.arange(last + 1, last + horizon + 1, dtype=float)
    pred = slope * future + intercept
    # Clamp only for bounded metrics / human readability.
    if col == "gini":
        pred = np.clip(pred, 0, 1)
    if col == "coverage_pct":
        pred = np.clip(pred, 0, None)
    return pd.DataFrame({"year": future.astype(int), "pred": pred, "slope": slope})

def direction_text(slope, metric):
    if abs(slope) < 1e-9:
        return "roughly stable"
    if metric in {"gini", "popular_exposure_pct", "amplification_x"}:
        return "increasing concentration"
    if metric == "coverage_pct":
        return "declining catalogue breadth"
    return "changing"

def model_story(model, d):
    first = d.iloc[0]
    last = d.iloc[-1]
    if model == "NeuMF":
        return (
            f"Across the retained period, exposure Gini rose from {first.gini:.3f} "
            f"to {last.gini:.3f}, while catalogue coverage fell from "
            f"{first.coverage_pct:.1f}% to {last.coverage_pct:.1f}%. "
            "The pattern is consistent with increasing exposure concentration, "
            "although the early years contain a temporary deviation from the longer-term trend."
        )
    if model == "SASRec":
        return (
            f"Exposure Gini increased from {first.gini:.3f} to {last.gini:.3f}. "
            f"Coverage declined from {first.coverage_pct:.1f}% to {last.coverage_pct:.1f}%, "
            "while popular-item exposure increased strongly. The overall pattern is a "
            "substantial move towards concentrated exposure."
        )
    return (
        f"At the earliest retained checkpoint, exposure Gini was already {first.gini:.4f} "
        f"and catalogue coverage was only {first.coverage_pct:.2f}%. By the final checkpoint "
        f"Gini reached {last.gini:.4f}. The main signal is persistence of extreme concentration "
        "rather than a large deterioration from a low-concentration starting point."
    )

def plain_interpretation(row):
    g = float(row.gini)
    cov = float(row.coverage_pct)
    pop = float(row.popular_exposure_pct)
    amp = float(row.amplification_x)
    messages = []
    if g >= 0.99:
        messages.append("Exposure is extremely concentrated across the catalogue.")
    elif g >= 0.95:
        messages.append("Exposure is highly concentrated across the catalogue.")
    elif g >= 0.85:
        messages.append("Exposure is fairly concentrated across the catalogue.")
    else:
        messages.append("Exposure is more distributed than in the high-concentration range.")
    messages.append(f"Only {cov:.2f}% of the catalogue appears in recommendations at this checkpoint.")
    messages.append(f"Popular items receive about {pop:.1f}% of recommendation exposure.")
    messages.append(f"Popular-item exposure is about {amp:.2f}× the group's catalogue-share baseline.")
    return " ".join(messages)

def safe_ai_explanation(model, row, trend_text, api_key, ai_model):
    if not api_key:
        return None, "No API key entered."
    if OpenAI is None:
        return None, "The `openai` package is not installed. Run `pip install -r requirements.txt`."
    prompt = f"""
You are the plain-English explanation layer of a recommender-system fairness auditing dashboard.
Do NOT rank models, name a winner, or declare a system ethically fair/unfair.
Explain the measured exposure signals and their practical meaning to a non-technical product manager.

Model: {model}
Year: {int(row.year)}
Exposure Gini: {row.gini:.6f}
Catalogue coverage: {row.coverage_pct:.3f}%
Popular-item exposure: {row.popular_exposure_pct:.3f}%
Exposure amplification: {row.amplification_x:.3f}x
Trend summary: {trend_text}

Return:
1) What the metrics say, in plain English.
2) What a practitioner should watch next.
3) One limitation: these metrics are exposure/concentration signals, not universal moral judgements of fairness.
Keep it under 180 words.
"""
    try:
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(model=ai_model, input=prompt)
        return resp.output_text.strip(), None
    except Exception as e:
        return None, f"OpenAI call failed: {type(e).__name__}: {e}"

def page_header(title, subtitle, tag, accent):
    st.markdown(
        f'''<div class="page-header {accent}">
            <div class="page-tag">{tag}</div>
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>''',
        unsafe_allow_html=True
    )

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("### ⚖️ Audit controls")
    view = st.radio(
        "View",
        ["Overview", "Temporal Audit", "Trend Outlook", "Model Explorer", "Metric Guide", "AI Explainer"],
        index=0,
    )
    st.markdown("---")
    selected_model = st.selectbox("Model", MODELS, index=0)
    model_df = temporal[temporal["model"] == selected_model].sort_values("year").copy()
    selected_year = st.select_slider(
        "Evaluation year",
        options=model_df["year"].astype(int).tolist(),
        value=int(model_df["year"].max()),
    )
    st.markdown("---")
    st.markdown(
        '<div class="small-note">Data source: completed temporal audit outputs. '
        'The dashboard is descriptive and uses the native checkpoint cadence; '
        'LightGCN is not interpolated.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div class="eyebrow">Practitioner-facing recommender audit toolkit</div>
    <h1>Proactive Fairness Auditor</h1>
    <p>See how recommendation exposure is distributed, how it changes over time, and where the system may need closer monitoring — explained in plain language.</p>
    <div class="live-pill"><span class="live-dot"></span> Audit workspace online</div>
  </div>
</div>
""", unsafe_allow_html=True)

selected_row = model_df[model_df["year"] == selected_year].iloc[0]
info = model_info[model_info["model"] == selected_model].iloc[0]



# ---------------------------
# Overview
# ---------------------------
if view == "Overview":
    page_header("Overview", "A high-level view of recommendation exposure and the signals that deserve attention.", "System snapshot", "blue")
    st.markdown(
        f"""<div class="glass card">
            <div class="kicker">Selected system</div>
            <div class="section-title" style="margin-top:4px;">{selected_model}</div>
            <div class="small-note"><b>{info.architecture}</b> — {info.plain}</div>
            <div class="stat-strip">
                <span class="stat-chip">Checkpoint: {selected_year}</span>
                <span class="stat-chip">Temporal span: {int(model_df.year.min())}–{int(model_df.year.max())}</span>
                <span class="stat-chip">{info.temporal_story}</span>
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    metric_cards = [
        ("Exposure Gini", f"{selected_row.gini:.4f}",
         "Unequal recommendation exposure across catalogue items.",
         "Lower = more evenly distributed",
         "0 = perfectly equal exposure; 1 = maximum inequality.",
         "0 = equality reference", "good-chip",
         max(0,min(100,float(selected_row.gini)*100))),
        ("Catalogue Coverage", f"{selected_row.coverage_pct:.2f}%",
         "Share of the available catalogue that appears in recommendations.",
         "Higher = broader catalogue reach",
         "100% is maximum breadth, but there is no universal minimum that is always fair.",
         "No universal target", "no-universal",
         max(0,min(100,float(selected_row.coverage_pct)))),
        ("Popular Exposure", f"{selected_row.popular_exposure_pct:.1f}%",
         "Share of recommendation exposure going to the popular-item group.",
         "Compare with catalogue share",
         "This audit uses the top 20% as the popular group, so 20% is the proportional-exposure reference.",
         "20% = proportional reference", "good-chip",
         max(0,min(100,float(selected_row.popular_exposure_pct)))),
        ("Amplification", f"{selected_row.amplification_x:.2f}×",
         "Popular-item exposure relative to the group's catalogue share.",
         "1× = proportional exposure",
         "1× means exposure is proportional to catalogue share; values above 1× indicate amplification.",
         "1× = proportional reference", "good-chip",
         max(0,min(100,float(selected_row.amplification_x)/5*100))),
    ]

    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    for label, value, helptext, badge, reference, chip, chip_class, needle in metric_cards:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-help">{helptext}</div>
                <div class="metric-badge">{badge}</div>
                <div class="metric-reference">
                    <span class="reference-label">Reference point</span>
                    {reference}
                    <span class="{chip_class}">{chip}</span>
                </div>
                <div class="metric-scale"><span class="needle" style="left:{needle:.1f}%"></span></div>
            </div>""",
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">What the dashboard sees</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="insight"><strong>{selected_model} at {selected_year}:</strong> '
        f'{plain_interpretation(selected_row)}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="reference-box">'
        '<div class="reference-box-title">🎯 What counts as a “good” score?</div>'
        '<div class="small-note">There is no single universal fairness score. These are practical reference points, not pass/fail rules.</div>'
        '<div class="reference-grid">'
        '<div class="reference-mini"><b>Gini</b><span>Closer to 0 = more equal exposure.</span></div>'
        '<div class="reference-mini"><b>Coverage</b><span>Closer to 100% = broader reach, but the right target depends on the application.</span></div>'
        '<div class="reference-mini"><b>Popular exposure</b><span>Compare with the popular group’s catalogue share. Here, 20% is proportional.</span></div>'
        '<div class="reference-mini"><b>Amplification</b><span>1× = exposure proportional to catalogue share.</span></div>'
        '</div></div>',
        unsafe_allow_html=True
    )

    d = model_df
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.year, y=d.gini, mode="lines+markers", name="Exposure Gini",
        line=dict(width=4), marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Exposure Gini: %{y:.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[int(selected_year)], y=[float(selected_row.gini)],
        mode="markers", name="Selected checkpoint",
        marker=dict(size=16, line=dict(width=3)),
        hovertemplate="Selected year: %{x}<br>Gini: %{y:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Exposure concentration through time", x=0.01),
        xaxis_title="Evaluation year", yaxis_title="Exposure Gini",
        yaxis=dict(range=[0, 1.02], gridcolor="#edf1f6"),
        height=420, margin=dict(l=20,r=20,t=65,b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="small-note"><b>How to read this:</b> a higher Gini indicates a more uneven exposure distribution. '
        'It is an audit signal about recommendation concentration, not a universal fairness verdict.</div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "⬇️ Download temporal audit data (CSV)",
        data=temporal.to_csv(index=False).encode("utf-8"),
        file_name="temporal_audit.csv",
        mime="text/csv"
    )

    st.markdown(
        '''<div class="sticky-explain">
            <div class="kicker">Plain-English reading guide</div>
            <div class="explain-row"><b>Exposure Gini</b><span>How unevenly recommendation exposure is distributed across catalogue items.</span></div>
            <div class="explain-row"><b>Coverage</b><span>How much of the catalogue receives any recommendation exposure at all.</span></div>
            <div class="explain-row"><b>Popular exposure</b><span>How much of the recommendation space is going to the popular-item group.</span></div>
            <div class="explain-row"><b>Amplification</b><span>How strongly recommendation exposure is magnifying the group's catalogue share.</span></div>
        </div>''',
        unsafe_allow_html=True
    )


# ---------------------------
# Temporal audit
# ---------------------------
elif view == "Temporal Audit":
    page_header("Temporal Audit", f"Follow how exposure concentration changes across the observed history for {selected_model}.", "Time & behaviour", "purple")
    st.markdown(
        f'<div class="small-note">Native checkpoint cadence for <b>{selected_model}</b>. '
        'LightGCN checkpoints are shown as collected; no missing years are interpolated.</div>',
        unsafe_allow_html=True
    )

    d = model_df

    def make_gini_figure(view_df, highlight_year=None):
        fig_local = go.Figure()
        fig_local.add_trace(go.Scatter(
            x=view_df.year, y=view_df.gini, mode="lines+markers",
            name="Exposure Gini", line=dict(width=4), marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>Exposure Gini: %{y:.4f}<extra></extra>"
        ))
        if highlight_year is not None:
            hit = view_df[view_df.year == highlight_year]
            if not hit.empty:
                fig_local.add_trace(go.Scatter(
                    x=hit.year, y=hit.gini, mode="markers", name="Current checkpoint",
                    marker=dict(size=17, line=dict(width=3)),
                    hovertemplate="Checkpoint %{x}<br>Gini: %{y:.4f}<extra></extra>"
                ))
        fig_local.update_layout(
            title=dict(text=f"{selected_model} — exposure Gini timeline", x=0.01),
            xaxis=dict(title="Evaluation year", gridcolor="#edf1f6"),
            yaxis=dict(title="Exposure Gini", range=[0,1.02], gridcolor="#edf1f6"),
            height=440, margin=dict(l=20,r=20,t=65,b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
            legend=dict(orientation="h", y=1.08, x=0),
            hovermode="x unified",
        )
        return fig_local

    animation_col1, animation_col2, animation_col3 = st.columns([1,1,2])
    with animation_col1:
        play_clicked = st.button("▶ Play timeline", type="primary", use_container_width=True)
    with animation_col2:
        show_all = st.button("↺ Reset view", use_container_width=True)
    with animation_col3:
        st.markdown(
            '<div class="small-note" style="padding-top:7px;">'
            '<b>Play</b> replays every checkpoint. <b>Reset</b> returns to the complete observed series.</div>',
            unsafe_allow_html=True
        )

    chart_placeholder = st.empty()
    progress_placeholder = st.empty()

    if play_clicked:
        for i in range(1, len(d) + 1):
            current = d.iloc[:i]
            current_year = int(current.iloc[-1].year)
            chart_placeholder.plotly_chart(
                make_gini_figure(current, highlight_year=current_year),
                use_container_width=True,
                key=f"gini_animation_{selected_model}_{current_year}"
            )
            progress_placeholder.progress(i / len(d), text=f"Replaying checkpoint: {current_year}")
            time.sleep(0.35)
        progress_placeholder.empty()
    else:
        chart_placeholder.plotly_chart(
            make_gini_figure(d, highlight_year=int(selected_year)),
            use_container_width=True,
            key=f"gini_static_{selected_model}_{selected_year}"
        )

    c1, c2 = st.columns(2)
    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=d.year, y=d.coverage_pct, mode="lines+markers",
            name="Catalogue coverage", line=dict(width=3)
        ))
        fig2.update_layout(
            title=dict(text="Catalogue breadth",x=.01),
            xaxis_title="Year", yaxis_title="Coverage (%)", height=340,
            margin=dict(l=20,r=20,t=55,b=20),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white",
            yaxis=dict(gridcolor="#edf1f6")
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=d.year, y=d.popular_exposure_pct, mode="lines+markers",
            name="Popular exposure", line=dict(width=3)
        ))
        fig3.update_layout(
            title=dict(text="Popular-item exposure",x=.01),
            xaxis_title="Year", yaxis_title="Exposure (%)", height=340,
            margin=dict(l=20,r=20,t=55,b=20),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white",
            yaxis=dict(gridcolor="#edf1f6")
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        f'<div class="glass card"><div class="kicker">Model reading</div>'
        f'<div class="section-title">{info.temporal_story}</div>'
        f'<div class="small-note">{model_story(selected_model,d)}</div></div>',
        unsafe_allow_html=True
    )

    with st.expander("🧠 Why might this pattern occur?"):
        if selected_model == "NeuMF":
            st.write("NeuMF learns user–item representations from historical interactions. A plausible interpretation is that repeated historical popularity patterns can become increasingly represented as more interaction data accumulate. This is a hypothesis, not causal proof.")
        elif selected_model == "SASRec":
            st.write("SASRec learns from the order of interactions in a user's history. A plausible interpretation is that repeated popular-item patterns in sequences may reinforce their future recommendation likelihood. This is a hypothesis, not causal proof.")
        else:
            st.write("LightGCN learns on a user–item graph. A plausible interpretation is that concentration already present in the interaction graph can be strongly reflected in recommendations. This is a hypothesis, not causal proof.")

# ---------------------------
# Trend outlook
# ---------------------------
elif view == "Trend Outlook":
    page_header("Trend Outlook", f"Project the direction of the observed audit signals for {selected_model}; use this as a monitoring aid.", "Early warning", "teal")
    st.markdown("### 🔮 Trend Outlook")
    st.markdown(
        '<div class="warn"><b>Interpretation:</b> the dotted line is a simple historical trend projection used as an early-warning monitoring signal, not a guaranteed future value.</div>',
        unsafe_allow_html=True
    )

    d = model_df
    future_g = fit_projection(d, "gini", horizon=3)
    future_c = fit_projection(d, "coverage_pct", horizon=3)

    if future_g is not None and future_c is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d.year,y=d.gini,mode="lines+markers",name="Observed",
            line=dict(width=4),marker=dict(size=7),
            hovertemplate="Observed %{x}<br>Gini %{y:.4f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=future_g.year,y=future_g.pred,mode="lines+markers",name="Trend projection",
            line=dict(width=3,dash="dash"),marker=dict(symbol="diamond",size=8),
            hovertemplate="Projected %{x}<br>Gini %{y:.4f}<extra></extra>"
        ))
        fig.update_layout(
            title=dict(text=f"{selected_model} — 3-year concentration outlook",x=.01),
            xaxis_title="Year", yaxis_title="Exposure Gini",
            yaxis=dict(range=[0,1.02],gridcolor="#edf1f6"),
            height=430,margin=dict(l=20,r=20,t=65,b=20),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white",
            legend=dict(orientation="h",y=1.08,x=0),
        )
        st.plotly_chart(fig,use_container_width=True)

        last_proj_g = float(future_g.iloc[-1].pred)
        last_proj_c = float(future_c.iloc[-1].pred)
        g_slope = float(future_g.iloc[0].slope)
        c_slope = float(future_c.iloc[0].slope)

        cols = st.columns(3)
        with cols[0]:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">3-year Gini outlook</div>'
                f'<div class="metric-value">{last_proj_g:.4f}</div>'
                f'<div class="metric-help">Trend-only estimate based on observed checkpoints.</div></div>',
                unsafe_allow_html=True)
        with cols[1]:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">3-year coverage outlook</div>'
                f'<div class="metric-value">{last_proj_c:.2f}%</div>'
                f'<div class="metric-help">Trend-only estimate; values below zero are clipped.</div></div>',
                unsafe_allow_html=True)
        with cols[2]:
            direction = "Concentration rising" if g_slope > 0 else "Concentration not rising"
            breadth = "Coverage declining" if c_slope < 0 else "Coverage not declining"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Monitoring direction</div>'
                f'<div class="metric-value" style="font-size:1.25rem">{direction}</div>'
                f'<div class="metric-help">{breadth}. Treat this as a trigger to inspect the next audit cycle.</div></div>',
                unsafe_allow_html=True)

        st.markdown('<div class="section-title">Suggested monitoring action</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight"><strong>Watch the next audit window.</strong> '
            f'The historical direction for {selected_model} suggests monitoring both exposure concentration '
            f'and catalogue breadth rather than relying on a single point-in-time measurement.</div>',
            unsafe_allow_html=True
        )

elif view == "Model Explorer":
    page_header("Model Explorer", "See how each recommender architecture works and how the observed temporal patterns differ.", "Architecture", "orange")
    st.markdown('<div class="arch-grid">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    arch_cards = [
        ("NeuMF","🧠","Neural collaborative filtering",
         "Uses user and item representations plus a neural interaction layer.",
         "Emerging concentration over time."),
        ("SASRec","🔄","Sequential self-attention",
         "Learns from the order of a user's past interactions.",
         "Strong increase in concentration over time."),
        ("LightGCN","🕸️","Graph-based collaborative filtering",
         "Propagates information through the user–item interaction graph.",
         "Persistent extreme concentration from the earliest retained checkpoint."),
    ]
    for col,(name,icon,arch,desc,story) in zip([c1,c2,c3],arch_cards):
        with col:
            st.markdown(
                f'<div class="arch-card"><div class="arch-icon">{icon}</div>'
                f'<div class="kicker">{name}</div><div class="section-title" style="margin-top:5px">{arch}</div>'
                f'<div class="small-note">{desc}</div>'
                f'<div class="stat-chip" style="margin-top:13px;display:inline-block">{story}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">The audit journey</div>', unsafe_allow_html=True)
    steps = [
        ("Audit","Measure exposure concentration, catalogue breadth and popularity amplification."),
        ("Observe","Repeat the audit across historical checkpoints."),
        ("Diagnose","Look for changing concentration, shrinking coverage or persistent concentration."),
        ("Monitor","Use the direction of the historical series to decide what deserves another audit.")
    ]
    for i,(title,desc) in enumerate(steps):
        st.markdown(
            f'<div class="timeline-step"><div class="step-dot"></div><div><b>{title}</b>'
            f'<div class="small-note">{desc}</div></div></div>',
            unsafe_allow_html=True
        )
        if i < len(steps)-1:
            st.markdown('<div class="step-line"></div>',unsafe_allow_html=True)

    latest_rows = []
    for m in MODELS:
        md = temporal[temporal["model"] == m].sort_values("year")
        lr = md.iloc[-1]
        latest_rows.append({
            "Model":m, "Latest year":int(lr.year), "Gini":f"{lr.gini:.4f}",
            "Coverage":f"{lr.coverage_pct:.2f}%", "Popular exposure":f"{lr.popular_exposure_pct:.1f}%",
            "Amplification":f"{lr.amplification_x:.2f}×"
        })
    st.dataframe(pd.DataFrame(latest_rows), use_container_width=True, hide_index=True)

# ---------------------------
# Metric guide
# ---------------------------
elif view == "Metric Guide":
    page_header("Metric Guide", "A plain-English guide to the metrics, their reference points and how to interpret them.", "Understand the numbers", "blue")
    metrics = [
        ("Exposure Gini",
         "How evenly recommendation exposure is distributed across catalogue items.",
         "Closer to 0 means more equal exposure; closer to 1 means greater inequality.",
         "0 is the mathematical equality point."),
        ("Catalogue Coverage",
         "The percentage of the available catalogue that appears in recommendations.",
         "Higher means broader catalogue reach.",
         "100% is maximum breadth; there is no universal minimum target."),
        ("Popular-item Exposure",
         "The share of recommendation slots going to the popular-item group.",
         "Compare the exposure share with that group's catalogue share.",
         "This audit uses the top 20% as the popular group; 20% is the proportional baseline."),
        ("Exposure Amplification",
         "How much more exposure popular items receive relative to their catalogue share.",
         "1× means proportional exposure; values above 1× indicate amplification.",
         "1× is the proportional reference."),
        ("Hit@10",
         "Whether at least one relevant item appears in a user's top 10 recommendations.",
         "Higher means more users received at least one relevant item in the top 10.",
         "No universal good score; compare within the same evaluation setup."),
        ("Recall@10",
         "How much of a user's relevant set was retrieved in the top 10.",
         "Higher means more relevant items were retrieved.",
         "No universal good score; compare within the same evaluation setup."),
        ("MRR",
         "Mean Reciprocal Rank: how high the first relevant recommendation appears.",
         "Higher means the first relevant item tends to appear nearer the top.",
         "No universal good score across tasks."),
        ("Trend projection",
         "A simple projection of the historical metric direction.",
         "Use it to see whether concentration appears to be rising or falling.",
         "Monitoring aid, not a validated prediction."),
    ]

    for title, what, read, reference in metrics:
        with st.expander(title):
            st.markdown(f"**What it measures:** {what}")
            st.markdown(f"**How to read it:** {read}")
            st.markdown(f"**Reference:** {reference}")

    st.markdown("#### 📏 Reference points at a glance")
    reference_table = pd.DataFrame([
        ["Exposure Gini", "0", "Equal exposure", "1 = maximum inequality"],
        ["Catalogue Coverage", "100%", "Maximum breadth", "No universal minimum"],
        ["Popular Exposure", "20%*", "Proportional exposure", "*Top 20% popular group in this audit"],
        ["Amplification", "1×", "Proportional exposure", ">1× indicates amplification"],
        ["Hit@10", "No universal threshold", "Higher = more users get a relevant top-10 item", "Same evaluation setup required"],
        ["Recall@10", "No universal threshold", "Higher = more relevant items retrieved", "Same evaluation setup required"],
        ["MRR", "No universal threshold", "Higher = relevant items tend to appear higher", "Same evaluation setup required"],
        ["Trend projection", "No target", "Direction of historical movement", "Not validated forecasting"],
    ], columns=["Metric", "Reference", "Meaning", "Caveat"])
    st.dataframe(reference_table, use_container_width=True, hide_index=True)

# ---------------------------
# AI Explainer
# ---------------------------
else:
    page_header("AI-Assisted Interpretation", "Translate measured audit signals into clear language for non-technical users.", "Assisted interpretation", "green")
    st.markdown(
        '<div class="ai-box"><div class="small-note"><b>Purpose:</b> translate the selected audit signals into plain English. '
        'The assistant explains the evidence; it does not make the fairness decision.</div></div>',
        unsafe_allow_html=True
    )

    def load_openai_key():
        # Local development: read from the gitignored key file.
        key_file = Path(__file__).parent / "openai_key.txt"
        if key_file.exists():
            key = key_file.read_text(encoding="utf-8").strip()
            if key:
                return key

        # Streamlit Community Cloud: read from app secrets.
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
            if key:
                return key
        except Exception:
            pass

        # Fallback for other deployment platforms / local environment variables.
        return os.getenv("OPENAI_API_KEY", "")


    api_key = load_openai_key()
    ai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    d = model_df
    latest = d.iloc[-1]
    trend_g = fit_projection(d, "gini", horizon=3)
    trend_c = fit_projection(d, "coverage_pct", horizon=3)
    trend_text = (
        f"Gini slope={float(trend_g.iloc[0].slope):.6f} per year; "
        f"coverage slope={float(trend_c.iloc[0].slope):.6f} percentage-points per year."
        if trend_g is not None and trend_c is not None else "Trend unavailable."
    )

    ai_status = "AI explanation ready" if api_key else "AI explanation unavailable — configure local API key"

    st.markdown(
        f'<div class="small-note"><b>Selected model:</b> {selected_model} &nbsp; '
        f'<b>Selected year:</b> {selected_year} &nbsp; '
        f'<b>Status:</b> {ai_status} &nbsp; '
        f'<b>Input:</b> four audit metrics + trend direction.</div>',
        unsafe_allow_html=True
    )

    if not api_key:
        st.markdown(
            '<div class="warn"><b>AI explanation is not configured.</b> '
            'Place your OpenAI key in <code>openai_key.txt</code> in the dashboard folder, '
            'then refresh the page.</div>',
            unsafe_allow_html=True
        )

    if st.button("✨ Explain this audit in plain English", type="primary"):
        with st.spinner("Generating explanation..."):
            answer, err = safe_ai_explanation(
                selected_model, selected_row, trend_text, api_key, ai_model
            )
        if answer:
            st.markdown(
                f'<div class="success"><b>AI explanation</b><br><br>{answer}</div>',
                unsafe_allow_html=True
            )
        else:
            st.error(err)

    st.markdown("#### Quick audit interpretation")
    st.write(plain_interpretation(selected_row))


# ---------------------------
# Downloads + footer
# ---------------------------
st.markdown(
    '<div class="footer">Project prototype: “Towards a Proactive Fairness Auditor for Recommender Systems”. '
    'Activity-based user groups are not protected demographic groups. '
    'The dashboard intentionally distinguishes measured exposure concentration from broader normative claims about fairness.</div>',
    unsafe_allow_html=True
)
