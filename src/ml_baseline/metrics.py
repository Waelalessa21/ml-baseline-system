from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))

    return metrics


def print_metrics(metrics, title="Metrics"):
    print(f"\n{title}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
