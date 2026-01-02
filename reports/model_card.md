# Model Card — Week 3 (Draft)

## 1) What is the prediction?
- **Target (y):** `is_high_value`
- **Unit of analysis:** one row = one user
- **Decision supported:** Identify high-value customers (total_amount > $50)

## 2) Data contract (inference)
- **ID passthrough columns:** `user_id`
- **Required feature columns (X):** `country`, `n_orders`, `total_amount`
- **Forbidden columns:** target + leakage fields

## 3) Evaluation plan
- **Split strategy:** random holdout (stratified)
- **Test size:** 0.2 (20%)
- **Random seed:** 42
- **Primary metric:** ROC-AUC
- **Secondary metrics:** accuracy, precision, recall, F1-score
- **Baseline model:** Logistic Regression with standard scaling (numeric) and one-hot encoding (categorical)