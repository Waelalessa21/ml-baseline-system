# Evaluation Summary

## Dataset & Task
- **Dataset:** Sample customer data (100 users)
- **Target:** `is_high_value` (binary: 1 if total_amount > $50, else 0)
- **Unit of analysis:** One row = one user
- **Features:** country, n_orders, total_amount
- **Split:** 80/20 random holdout (stratified), seed=42

## Baseline vs Model Performance

**Primary Metric: ROC-AUC**

| Metric | Baseline (Majority Class) | Model (LogisticRegression) | Improvement |
|--------|---------------------------|---------------------------|-------------|
| **ROC-AUC** | 0.5000 | **1.0000** | +0.5000 |
| Accuracy | 0.8000 | 0.9500 | +0.1500 |
| Precision | 0.8000 | 1.0000 | +0.2000 |
| Recall | 1.0000 | 0.9375 | -0.0625 |
| F1-Score | 0.8889 | 0.9677 | +0.0788 |

## Model Details
- **Model:** Logistic Regression (L2 regularization, max_iter=1000)
- **Preprocessing:** StandardScaler for numeric features, OneHotEncoder for categorical
- **Train size:** 80 samples
- **Test size:** 20 samples
- **Target distribution:** 80% positive (high_value=1), 20% negative

## Caveats & Likely Failure Modes

1. **Small dataset (n=100)**: Model may not generalize well to larger populations or different customer segments
2. **High class imbalance (80/20)**: Model may struggle with minority class (low-value customers) and could overfit to high-value patterns
3. **Feature leakage risk**: `total_amount` directly determines the target - model may just be learning a threshold rule rather than predictive patterns from country/n_orders
4. **No temporal validation**: Random split doesn't account for time-based drift in customer behavior
5. **Perfect test metrics (ROC-AUC=1.0)**: Suspiciously high performance suggests potential overfitting or data leakage

## Recommendation
- Collect more data before production deployment
- Investigate feature importance to confirm model isn't just using total_amount threshold
- Consider time-based split to validate temporal stability
