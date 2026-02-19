# 🛡️ Fraud Detection & Risk Monitoring Dashboard

🔗 Live Demo: https://your-streamlit-link.streamlit.app  
💻 GitHub Repository: https://github.com/your-username/fraud-detection-dashboard  
📊 Dataset Source: 

---

## 📌 Project Overview

This project demonstrates an end-to-end fraud detection and monitoring system using transaction-level credit card data.

The objective was to:

- Detect fraudulent transactions using machine learning
- Monitor fraud KPIs through a dashboard
- Translate technical outputs into business-friendly risk insights

The solution integrates:

- Data preprocessing & exploratory analysis
- Imbalanced data handling (SMOTE / class weighting)
- Predictive modeling (Logistic Regression & Random Forest)
- Performance evaluation using Precision, Recall, F1, ROC-AUC
- Interactive Streamlit dashboard for fraud monitoring

---

## 📂 Dataset

Dataset Source: Kaggle – Credit Card Fraud Detection Dataset  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

Dataset characteristics:

- 284,807 transactions  
- Highly imbalanced dataset (~0.17% fraud rate)  
- PCA-transformed features (V1–V28)  
- Time and transaction amount features  

---

## ⚙️ Machine Learning Pipeline

### 1️⃣ Data Preprocessing
- Stratified 80/20 train-test split
- Handled class imbalance using:
  - SMOTE oversampling
  - Class-weight balancing
- Standardized numerical features

### 2️⃣ Models Implemented
- Logistic Regression (interpretable baseline)
- Random Forest (non-linear ensemble model)

### 3️⃣ Evaluation Metrics
Due to extreme class imbalance, performance was measured using:

- Precision
- Recall
- F1-score
- ROC-AUC

---

## 📊 Model Performance

| Metric        | Logistic Regression | Random Forest |
|--------------|--------------------|---------------|
| Precision    | XX%                | XX%           |
| Recall       | XX%                | XX%           |
| F1 Score     | XX%                | XX%           |
| ROC-AUC      | X.XX               | X.XX          |

> Focus was placed on Precision and Recall rather than Accuracy due to imbalance.

---

## 📈 Fraud Monitoring Dashboard Features

The deployed Streamlit dashboard includes:

### 🔹 KPI Metrics
- Total Transactions
- Fraud Transactions
- Fraud Rate (%)
- Total Fraud Amount at Risk

### 🔹 Visualizations
- Fraud vs Non-Fraud distribution
- Transaction amount distribution
- Rolling fraud trend
- Top high-risk transactions

### 🔹 Risk Scoring
- Individual transaction probability scoring
- Downloadable scored dataset

---

## 🌍 Deployment

The dashboard is publicly deployed using Streamlit Cloud.

To access the live application:

👉 Click the Live Demo link at the top.

---

## 🖥️ Running Locally (Mac)

1️⃣ Clone the repository:

git clone https://github.com/your-username/fraud-detection-dashboard.git
cd fraud-detection-dashboard

2️⃣ Create virtual environment:

python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies:

pip install -r requirements.txt

4️⃣ Run the dashboard:

streamlit run app/fraud_dashboard.py

5️⃣ Open browser:

http://localhost:8501

---

## 📁 Project Structure

fraud-detection-dashboard/                                                    
│                                                                               
├── app/                                                                       
│ └── fraud_dashboard.py                                                               
├── models/                                                                     
│ ├── fraud_model.joblib                                                                      
│ └── scaler.joblib                                                               
├── notebooks/                                                           
│ ├── 01_EDA.ipynb                                                               
│ └── 02_model_training.ipynb                                                                 
├── data/                                                                           
│ └── README.md                                                                         
├── requirements.txt                                                        
└── README.md

---

## 🎯 Business Impact

This project demonstrates how transaction-level data can be transformed into actionable fraud monitoring insights.

Key value delivered:

- Identified fraud patterns in highly imbalanced data
- Developed calibrated probability-based risk scoring
- Built dashboard to support operational fraud monitoring
- Enabled decision-making through KPI visualization

This reflects a real-world fraud analytics workflow:

Data Collection → Data Cleaning → Risk Modeling → KPI Reporting → Decision Support

---

## ⚠️ Disclaimer

This project is built for educational and portfolio purposes only.  
It is not intended for real-world financial deployment.

---

## 👤 Author

Omini Rao  
Business Intelligence | Data Analytics | Machine Learning

