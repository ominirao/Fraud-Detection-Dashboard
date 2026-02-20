# 💳 Fraud Detection & Risk Analytics Dashboard

🔗 **Live Demo:** https://fraud-detection-dashboard-dtnkhbqjlwzcq5swldkkla.streamlit.app                                     
💻 **GitHub Repository:** https://github.com/ominirao/Fraud-Detection-Dashboard                                                          
📊 **Dataset Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

---

## 📌 Project Overview

This project is a Machine Learning-powered fraud detection system designed to identify high-risk financial transactions in real time.

The model predicts the probability of fraud using transaction-level behavioral and financial features.

The web application provides:

- Real-time fraud probability scoring
- Risk-based transaction ranking
- Fraud rate monitoring KPIs
- Transaction trend analysis
- Public cloud deployment

---

## 📸 Application Preview

### Dashboard Overview
<img width="2000" height="1000" alt="1258BF87-C133-400D-B7A6-8B84605498B0" src="https://github.com/user-attachments/assets/86c3c8a7-8d9c-4ced-80de-efb4f8ec3f71" />


### Fraud Probability Output
<img width="2939" height="1658" alt="D9E9A705-31A2-445F-9600-ED58C410D96C" src="https://github.com/user-attachments/assets/24653258-eb5f-414f-9d61-dfd1f57bbafa" />

---

## 🎯 Business Impact

This project demonstrates how transaction-level financial data can be transformed into actionable fraud risk intelligence.

Key outcomes:

- Performed exploratory data analysis on highly imbalanced financial data
- Applied SMOTE to address class imbalance
- Engineered transaction-level risk indicators
- Built a calibrated probability model for fraud scoring
- Designed a real-time dashboard for operational monitoring
- Translated ML outputs into business-friendly KPIs

This project showcases an end-to-end analytics pipeline:

Data Collection → Data Cleaning → Feature Engineering → Model Development → Evaluation → Deployment

---

## ⚙️ Machine Learning Pipeline

The model was built using:

- Random Forest Classifier
- SMOTE (Class Imbalance Handling)
- StandardScaler
- Stratified Train-Test Split
- Probability-based Risk Scoring

---

## 📊 Model Performance

- Accuracy: 99.2%
- Precision: 91%
- Recall: 84%
- F1 Score: 87%
- ROC-AUC: 0.96

*(Metrics based on validation dataset — see training notebook for full evaluation.)*

---

## 📈 Dashboard Features

The Streamlit dashboard includes:

- Total Transactions KPI
- Fraud Transactions KPI
- Fraud Rate %
- Total Fraud Amount
- Transaction Trend Analysis
- Top Suspicious Transactions by Amount
- Downloadable scored transaction samples

---

## 🌍 Deployment

The application is deployed using **Streamlit Cloud**.

👉 Click the Live Demo link above to explore the dashboard.

---

## 🖥️ Running Locally

1️⃣ Clone the repository  
git clone https://github.com/yourusername/fraud-detection-dashboard.git                                                
cd fraud-detection-dashboard  

2️⃣ Create virtual environment  
python3 -m venv venv                                                                                     
source venv/bin/activate  

3️⃣ Install dependencies  
pip install -r requirements.txt  

4️⃣ Run the app  
streamlit run app.py  

---

## 📁 Project Structure

fraud-detection-dashboard/                                                                   
│                                                                                                      
├── app.py                                                                                                  
├── models/                                                                                                  
│   ├── fraud_model.joblib                                                                               
│   └── scaler.joblib                                                                           
├── Dataset/                                                                                  
│   └── creditcard.csv                                                                           
├── requirements.txt                                                                             
├── README.md                                                                           
└── notebooks/                                                                                        
    └── training_notebook.ipynb                                                                       

---

## 👤 Author

Omini Rao  
Machine Learning | Business Intelligence | Risk Analytics
