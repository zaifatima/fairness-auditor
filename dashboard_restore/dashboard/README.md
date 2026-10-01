# Proactive Fairness Auditor â€” Dashboard Prototype

A practitioner-facing Streamlit dashboard for the recommender-system fairness audit results.

## What is included

- NeuMF, SASRec and LightGCN temporal audit results
- Exposure Gini
- Catalogue coverage
- Popular-item exposure
- Exposure amplification
- Interactive temporal charts
- A simple 3-year trend-based outlook
- Plain-English metric guide
- Optional OpenAI explanation layer
- CSV download of the temporal audit data

## Important methodological note

The â€œTrend Outlookâ€ is a simple trend projection based on the observed temporal series. It is **not** the Random Forest forecasting model proposed in the original dissertation design, and it is not a validated predictive model. For the dissertation, describe it as a prototype early-warning/trend view unless a formally evaluated forecasting experiment is completed.

The dashboard does not rank models or declare a universal fairness winner. The metrics are exposure/concentration signals.

## Run locally

```powershell
cd proactive_fairness_dashboard
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Then open the local address shown by Streamlit.

## Optional OpenAI explainer

Use the â€œAI Explainerâ€ page and enter an API key at runtime. The key is not stored in the project files. The app uses the OpenAI Responses API and defaults to `gpt-5.6-luna`; change the model field when needed.

## Screenshots for dissertation

Recommended screenshots:
1. Overview page with the four headline audit metrics.
2. Temporal Audit page for NeuMF.
3. Temporal Audit page for SASRec.
4. Temporal Audit page for LightGCN.
5. Trend Outlook page.
6. Metric Guide page.

Do not present the trend projection as validated forecasting accuracy.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io and choose **Create app**.
3. Select the repository, branch (`main`) and `app.py`.
4. In **Advanced settings â†’ Secrets**, add:
   ```
   OPENAI_API_KEY=YOUR_OPENAI_API_KEY
   ```
5. Deploy. The app reads the key from Streamlit secrets in the cloud; `openai_key.txt` remains local and should never be committed.

The dashboard does not require the OpenAI feature to display the core audit results.
