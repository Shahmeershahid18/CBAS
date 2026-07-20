"""
Canonical feature definition for the Churn Prediction model.

Shared by the training pipeline and the inference service. The API accepts the
clean `internal` keys; training also knows how to rename the raw Kaggle Telco
Customer Churn column names onto those keys.
Dataset: blastchar/telco-customer-churn (7,043 customers).
"""

# Internal feature key -> raw column name in "WA_Fn-UseC_-Telco-Customer-Churn.csv".
KAGGLE_COLUMN_MAP = {
    "tenure": "tenure",
    "monthly_charges": "MonthlyCharges",
    "total_charges": "TotalCharges",
    "contract": "Contract",
    "payment_method": "PaymentMethod",
    "internet_service": "InternetService",
    "paperless_billing": "PaperlessBilling",
    "tech_support": "TechSupport",
    "online_security": "OnlineSecurity",
    "senior_citizen": "SeniorCitizen",
}

NUMERIC_FEATURES = [
    "tenure",           # months as a customer
    "monthly_charges",
    "total_charges",
]

CATEGORICAL_FEATURES = [
    "contract",          # Month-to-month / One year / Two year
    "payment_method",
    "internet_service",  # DSL / Fiber optic / No
    "paperless_billing", # Yes / No
    "tech_support",      # Yes / No / No internet service
    "online_security",   # Yes / No / No internet service
    "senior_citizen",    # 0 / 1
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "churn"          # internal name (1 = churned)
KAGGLE_TARGET = "Churn"   # raw column in the CSV ("Yes"/"No")
