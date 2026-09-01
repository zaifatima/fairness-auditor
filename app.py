import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.metrics import evaluate_checkpoint_fairness

st.set_page_config(page_title="Proactive Fairness Auditor", layout="wide")

st.title("🛡️ Proactive Fairness Auditor (PFA)")
st.caption("Local Execution Mode — Offline Evaluation & Longitudinal Fairness Forecasting")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Model & Weights Selection")
    selected_model = st.selectbox("Select Recommendation Model", ["LightGCN", "SASRec"])
    
    st.header("2. Audit Parameters")
    top_k = st.slider("Top-K Recommendation Cutoff", min_value=5, max_value=50, value=10)
    alert_threshold = st.slider("Max Permissible Gini Disparity (α)", 0.10, 0.90, 0.45)

# --- Simulated Evaluation Pipeline (Replacing with .pth / .npz outputs) ---
# Generating sample prediction matrix shape (Users x Top-K) to simulate loaded weights
np.random.seed(42 if selected_model == "LightGCN" else 101)
n_users = 1000
# LightGCN tends to have higher popularity bias than SASRec
item_alpha = 0.5 if selected_model == "LightGCN" else 1.2 
simulated_topk = np.random.zipf(a=1.5, size=(n_users, top_k)) % 5000

metrics = evaluate_checkpoint_fairness(simulated_topk, total_catalog_items=5000)

# --- Executive Banner KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("Selected Architecture", selected_model)
col2.metric("Gini Exposure Index", f"{metrics['gini_index']}")
col3.metric("Catalog Coverage", f"{metrics['catalog_coverage_pct']}%")

if metrics['gini_index'] > alert_threshold:
    st.error(f"🔴 ALERT: Disparity threshold exceeded ({metrics['gini_index']} > {alert_threshold})")
else:
    st.success("🟢 NORMAL: Disparity within acceptable operational limits")

st.divider()

# --- Main Analytics Tab Setup ---
tab1, tab2 = st.tabs(["📈 Drift Forecasting & Trajectory", "📋 Mitigation Advisory"])

with tab1:
    st.subheader(f"Projected Fairness Drift ({selected_model})")
    
    epochs = np.arange(1, 21)
    # Simulate drift over epochs
    base_gini = metrics['gini_index']
    gini_drift = base_gini + (0.012 * epochs) + np.random.normal(0, 0.01, len(epochs))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs[:12], y=gini_drift[:12], mode='lines+markers', name='Observed Disparity'))
    fig.add_trace(go.Scatter(x=epochs[11:], y=gini_drift[11:], mode='lines+markers', name='Projected Drift', line=dict(dash='dash', color='red')))
    fig.add_hline(y=alert_threshold, line_dash="dot", line_color="orange", annotation_text="Alert Threshold (α)")

    fig.update_layout(xaxis_title="Epoch / Recommendation Cycle", yaxis_title="Gini Disparity Index", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Automated Mitigation & Calibration Strategy")
    st.dataframe(pd.DataFrame({
        "Model Component": [f"{selected_model} Embeddings", "Top-K Re-ranker"],
        "Detected Issue": ["Head-item over-representation", "Tail-item suppression"],
        "Recommended Action": ["Apply post-hoc re-ranking score adjustment", "Increase tail-item exploration bonus (+0.12)"]
    }), use_container_width=True)