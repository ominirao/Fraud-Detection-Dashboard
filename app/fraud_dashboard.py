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
# Automatically load dataset from repo
DATA_PATH = "Dataset/creditcard.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.warning("Local dataset not found. Please upload creditcard.csv.")
    upload = st.sidebar.file_uploader("Upload creditcard.csv", type=["csv"])
    if upload:
        df = pd.read_csv(upload)
    else:
        st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("Select time window for KPI aggregation")
window = st.sidebar.selectbox("Window", ["All data", "Last 24 hours (simulated)", "Last 7 days (simulated)"])

# Basic KPIs
total_tx = len(df)
fraud_tx = int(df['Class'].sum()) if 'Class' in df.columns else 0
fraud_rate = fraud_tx / total_tx if total_tx else 0
total_fraud_amount = df.loc[df['Class'] == 1, 'Amount'].sum() if 'Class' in df.columns else 0.0

k1, k2, k3 = st.columns(3)
k1.metric("Total transactions", f"{total_tx:,}")
k2.metric("Fraud transactions", f"{fraud_tx:,}", f"{fraud_rate*100:.3f}%")
k3.metric("Fraud amount (sum)", f"${total_fraud_amount:,.2f}")

st.markdown("## Fraud distribution and analysis")
col1, col2 = st.columns(2)
with col1:
    st.write("Fraud vs Non-Fraud counts")
    if 'Class' in df.columns:
        counts = df['Class'].value_counts().rename({0: 'Non-Fraud', 1: 'Fraud'})
        st.bar_chart(pd.DataFrame(counts))
    else:
        st.write("No 'Class' column in dataset.")

with col2:
    st.write("Transaction amount (log scale)")
    if 'Amount' in df.columns:
        st.write(df['Amount'].describe())
        st.line_chart(df['Amount'].rolling(100).mean())
    else:
        st.write("No 'Amount' column in dataset.")

st.markdown("## Top suspicious transactions (by amount among frauds)")
if 'Class' in df.columns and 'Amount' in df.columns and 'Time' in df.columns:
    fraud_top = df[df['Class'] == 1].sort_values('Amount', ascending=False).head(10)
    st.table(fraud_top[['Time', 'Amount']].assign(Amount=lambda x: x['Amount'].map("${:,.2f}".format)))
else:
    st.write("Dataset missing required columns for top suspicious transactions (need 'Class','Amount','Time').")

st.markdown("## Run risk scoring for a sample of transactions")

# Sample selection
sample = df.sample(min(500, len(df)), random_state=42).reset_index(drop=True)
X_sample = sample.copy()

# Debug info (useful if something goes wrong)
with st.expander("Debug: model & scaler info"):
    st.write("scaler type:", type(scaler))
    st.write("scaler.feature_names_in_ (if present):", getattr(scaler, "feature_names_in_", None))
    st.write("model type:", type(model))
    st.write("model.feature_names_in_ (if present):", getattr(model, "feature_names_in_", None))
    st.write("sample columns:", list(X_sample.columns))

# Try to scale Amount robustly and create Amount_scaled
try:
    # Case: scaler was fitted on DataFrame with named columns
    if hasattr(scaler, "feature_names_in_"):
        scaler_features = list(scaler.feature_names_in_)
        # select features present in the sample that the scaler expects
        present = [f for f in scaler_features if f in X_sample.columns]
        if not present:
            raise ValueError(f"Scaler expects features {scaler_features} but none of those exist in the sample.")
        X_to_transform = X_sample[present]
        transformed = scaler.transform(X_to_transform)

        # transformed could be 1D-like or 2D. Extract Amount column if possible.
        if transformed.ndim == 1 or (transformed.ndim == 2 and transformed.shape[1] == 1):
            X_sample['Amount_scaled'] = np.ravel(transformed)
        else:
            if 'Amount' in present:
                amt_idx = present.index('Amount')
                X_sample['Amount_scaled'] = transformed[:, amt_idx]
            else:
                # fallback: take the first transformed column
                X_sample['Amount_scaled'] = transformed[:, 0]

    # Case: scaler was fitted on numpy arrays (no feature names) - assume single column scaling
    else:
        if 'Amount' not in X_sample.columns:
            raise ValueError("No 'Amount' column in sample to scale.")
        transformed = scaler.transform(X_sample[['Amount']].to_numpy())
        X_sample['Amount_scaled'] = np.ravel(transformed)

except Exception as e:
    st.warning("Scaling failed; proceeding with unscaled Amount. See debug info above for details.")
    st.write("Scaling error:", e)
    # fallback to original Amount
    if 'Amount' in X_sample.columns:
        X_sample['Amount_scaled'] = X_sample['Amount']
    else:
        # create a zero column if Amount missing
        X_sample['Amount_scaled'] = 0.0

# Prepare features for the model
X_for_model = X_sample.copy()

# Use scaled Amount as the 'Amount' feature so models expecting 'Amount' find it
X_for_model['Amount'] = X_for_model['Amount_scaled']

# Remove label column if present
if 'Class' in X_for_model.columns:
    X_for_model = X_for_model.drop(columns=['Class'])

# If the model records feature names, try to restrict to those columns
if hasattr(model, "feature_names_in_"):
    model_features = [f for f in model.feature_names_in_ if f in X_for_model.columns]
    if model_features:
        X_for_model = X_for_model[model_features]
    else:
        st.warning("No overlapping feature names found between model.feature_names_in_ and sample columns. Using available columns as-is.")

# Ensure numeric input (convert non-numeric where possible)
# Keep index alignment for mapping predictions back to sample
try:
    X_numeric = X_for_model.select_dtypes(include=[np.number])
    # If no numeric columns found, attempt to coerce all to numeric (may introduce NaNs)
    if X_numeric.shape[1] == 0:
        X_for_model = X_for_model.apply(pd.to_numeric, errors='coerce').fillna(0)
    else:
        # keep numeric columns + any remaining columns coerced to numeric
        non_numeric = [c for c in X_for_model.columns if c not in X_numeric.columns]
        if non_numeric:
            X_for_model[non_numeric] = X_for_model[non_numeric].apply(pd.to_numeric, errors='coerce').fillna(0)

except Exception as e:
    st.warning("Failed to coerce features to numeric; model input may fail.")
    st.write("Coercion error:", e)

# Predict probabilities / scores
try:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_for_model)[:, 1]
    else:
        # if model only supports predict (e.g., it's a regressor), use that as "score"
        probs = model.predict(X_for_model)
except Exception as e:
    st.error("Model prediction failed. Check feature alignment and types between your sample and the trained model.")
    st.write("Prediction error:", e)
    probs = np.zeros(len(X_for_model))

# Attach probabilities back to the original sample DataFrame (keeps original Amount)
sample['fraud_prob'] = probs

# Show scored sample (original Amount retained)
cols_to_show = [c for c in ['Time', 'Amount', 'fraud_prob'] if c in sample.columns]
st.dataframe(sample[cols_to_show].head(20))

st.markdown("## Download scored sample")
csv_bytes = sample.to_csv(index=False).encode('utf-8')
st.download_button("Download sample with fraud_prob", csv_bytes, "scored_sample.csv")
