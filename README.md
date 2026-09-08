# Fraud Detection in Financial Transactions

An end-to-end machine learning project for identifying suspicious financial transactions with supervised classification and unsupervised anomaly detection.

## Goal
Detect potentially fraudulent transactions while handling severe class imbalance and converting model outputs into actionable alerts.

## Features
- EDA for transaction volume, fraud rate, amount patterns and categorical risk signals
- Class-imbalance strategies using class weights and SMOTE
- Logistic Regression baseline and Random Forest classifier
- Isolation Forest anomaly detection
- Autoencoder reconstruction-error anomaly detection
- Evaluation with precision, recall, F1, ROC-AUC and PR-AUC
- Fraud-score threshold tuning
- Alert severity engine for operational review
- Streamlit alert dashboard

## Dataset
Place an anonymized CSV at `data/raw/transactions.csv`. Raw data is ignored by Git to avoid committing sensitive or large files.

Expected target column: `is_fraud` with values `0` and `1`. For purely unsupervised experiments, the target can be omitted.

## Structure
```text
Fraud-Detection-in-Financial-Transactions/
├── data/raw/transactions.csv
├── data/processed/
├── models/
├── notebooks/01_fraud_eda_and_modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── anomaly_detection.py
│   ├── supervised_models.py
│   └── alerting.py
├── dashboard/app.py
├── reports/README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Modeling workflow
1. Load and validate anonymized transaction data.
2. Run the EDA notebook to identify fraud patterns and data-quality issues.
3. Preprocess numeric and categorical features.
4. Compare Logistic Regression and Random Forest with imbalance handling.
5. Train Isolation Forest and Autoencoder anomaly detectors.
6. Evaluate using fraud-focused metrics, especially recall and PR-AUC.
7. Tune thresholds and map scores into LOW/MEDIUM/HIGH/CRITICAL alerts.
8. Review alerts through the Streamlit dashboard.

## Why not accuracy alone?
Fraud is typically a minority class, so a model can achieve high accuracy while missing many fraudulent transactions. This project emphasizes recall, precision, F1, PR-AUC and alert volume.

## Disclaimer
For educational and portfolio use. Model outputs are risk signals and should not be treated as definitive proof of fraud.
