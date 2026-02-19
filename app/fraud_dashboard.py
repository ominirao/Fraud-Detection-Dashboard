# app/fraud_dashboard.py
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Fraud Monitoring Dashboard", layout="wide")
st.title("Fraud Monitoring & Risk Analytics")

MODEL_PATH = "models/fraud_model.joblib"
SCALER_PATH = "models/scaler.joblib"

# Check
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    st.error("Model or scaler missing. Put fraud_model.joblib and scaler.joblib in models/")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.sidebar.header("Data input")
upload = st.sidebar.file_uploader("Upload transactions CSV (creditcard.csv) or leave empty to use sample", type=["csv"])
if upload:
    df = pd.read_csv(upload)
else:
    st.sidebar.info("No file uploaded — using a small sample from the dataset (first 5000 rows).")
    # load bundled sample if present in repo data/sample.csv else fail gracefully
    sample_path = "data/sample_creditcard.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        st.error("No sample data found. Upload 'creditcard.csv' from Kaggle in the sidebar.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("Select time window for KPI aggregation")
window = st.sidebar.selectbox("Window", ["All data", "Last 24 hours (simulated)", "Last 7 days (simulated)"])

# Basic KPIs
total_tx = len(df)
fraud_tx = int(df['Class'].sum())
fraud_rate = fraud_tx / total_tx if total_tx else 0
total_fraud_amount = df.loc[df['Class']==1, 'Amount'].sum()

k1, k2, k3 = st.columns(3)
k1.metric("Total transactions", f"{total_tx:,}")
k2.metric("Fraud transactions", f"{fraud_tx:,}", f"{fraud_rate*100:.3f}%")
k3.metric("Fraud amount (sum)", f"${total_fraud_amount:,.2f}")

st.markdown("## Fraud distribution and analysis")
col1, col2 = st.columns(2)
with col1:
    st.write("Fraud vs Non-Fraud counts")
    counts = df['Class'].value_counts().rename({0:'Non-Fraud',1:'Fraud'})
    st.bar_chart(pd.DataFrame(counts))

with col2:
    st.write("Transaction amount (log scale)")
    st.write(df['Amount'].describe())
    st.line_chart(df['Amount'].rolling(100).mean())

st.markdown("## Top suspicious transactions (by amount among frauds)")
fraud_top = df[df['Class']==1].sort_values('Amount', ascending=False).head(10)
st.table(fraud_top[['Time','Amount']].assign(Amount=lambda x: x['Amount'].map("${:,.2f}".format)))

st.markdown("## Run risk scoring for a sample of transactions")
sample = df.sample(min(500, len(df)), random_state=42).reset_index(drop=True)
# prepare features consistent with training: scale Amount and Time
X_sample = sample.copy()
X_sample['Amount'] = scaler.transform(X_sample[['Amount']])
try:
    X_sample_scaled = X_sample.drop(columns=['Class'])
except:
    X_sample_scaled = X_sample
probs = model.predict_proba(X_sample_scaled)[:,1] if hasattr(model, "predict_proba") else model.predict(X_sample_scaled)
sample['fraud_prob'] = probs
st.dataframe(sample[['Time','Amount','fraud_prob']].head(20))

st.markdown("## Download scored sample")
st.download_button("Download sample with fraud_prob", sample.to_csv(index=False).encode('utf-8'), "scored_sample.csv")
