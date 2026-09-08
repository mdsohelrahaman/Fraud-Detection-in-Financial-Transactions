"""Streamlit dashboard for transaction-level fraud alerts."""
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🚨", layout="wide")

st.title("🚨 Financial Fraud Detection Dashboard")
st.caption("Anomaly scores and fraud alerts for anonymized financial transactions")

DATA_PATH = Path("data/processed/scored_transactions.csv")
RAW_PATH = Path("data/raw/transactions.csv")

path = DATA_PATH if DATA_PATH.exists() else RAW_PATH

if not path.exists():
    st.warning(
        "No transaction file found. Add an anonymized CSV to "
        "data/raw/transactions.csv or export scored_transactions.csv from the modeling notebook."
    )
    st.stop()

df = pd.read_csv(path)
st.sidebar.header("Filters")

score_cols = [c for c in df.columns if c in {"fraud_score", "anomaly_score", "reconstruction_score"}]
score_col = st.sidebar.selectbox("Score column", score_cols, index=0) if score_cols else None

if "alert_level" in df.columns:
    levels = st.sidebar.multiselect(
        "Alert level",
        sorted(df["alert_level"].dropna().unique().tolist()),
        default=sorted(df["alert_level"].dropna().unique().tolist()),
    )
    filtered = df[df["alert_level"].isin(levels)]
else:
    filtered = df.copy()

if score_col:
    threshold = st.sidebar.slider("Minimum score", 0.0, 1.0, 0.50, 0.01)
    filtered = filtered[filtered[score_col] >= threshold]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(filtered):,}")
c2.metric("Alerts", f"{int(filtered.get('review_required', pd.Series(False, index=filtered.index)).sum()):,}")
if "is_fraud" in filtered.columns:
    fraud_rate = 100 * filtered["is_fraud"].mean()
    c3.metric("Observed fraud rate", f"{fraud_rate:.2f}%")
else:
    c3.metric("Observed fraud rate", "N/A")
if score_col:
    c4.metric("Average score", f"{filtered[score_col].mean():.3f}")
else:
    c4.metric("Average score", "N/A")

st.subheader("Alert distribution")
if "alert_level" in filtered.columns:
    st.bar_chart(filtered["alert_level"].value_counts())

st.subheader("Priority transactions")
st.dataframe(filtered.head(200), use_container_width=True, hide_index=True)
