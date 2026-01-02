import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from .splits import get_splitter
from .metrics import calculate_classification_metrics, print_metrics
from .pipeline import build_baseline_pipeline
from .schema import InputSchema
from .io import write_tabular, best_effort_ext


def train_model(
    data_path,
    target,
    id_col="user_id",
    test_size=0.2,
    random_state=42,
    split_strategy="random",
    **split_kwargs,
):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"models/runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training run: {run_id}")
    print(f"Run directory: {run_dir}")

    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    y = df[target]
    X = df.drop(columns=[target, id_col] if id_col in df.columns else [target])

    print(f"\nTarget: {target}")
    print(f"Distribution: {y.value_counts().to_dict()}")
    print(f"\nFeatures: {list(X.columns)}")
    print(f"Shape: {X.shape}")

    print(f"\nSplitting data (test_size={test_size}, strategy={split_strategy})...")
    splitter = get_splitter(split_strategy)

    split_params = {"test_size": test_size}
    if split_strategy == "random":
        split_params["random_state"] = random_state
    elif split_strategy == "time":
        split_params["time_col"] = split_kwargs.get("time_col")
    elif split_strategy == "group":
        split_params["group_col"] = split_kwargs.get("group_col")
        split_params["random_state"] = random_state

    X_train, X_test, y_train, y_test = splitter(X, y, **split_params)
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")

    pipeline = build_baseline_pipeline(X, random_state=random_state)

    print("\nTraining baseline model (LogisticRegression)...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    print("\nEvaluating on holdout set...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
    print_metrics(metrics, "Holdout Metrics")

    results = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "target": target,
        "features": list(X.columns),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "test_size": test_size,
        "random_state": random_state,
        "split_strategy": split_strategy,
        "model_type": "LogisticRegression",
        "metrics": metrics,
    }

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "baseline_holdout.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics: {metrics_dir / 'baseline_holdout.json'}")

    with open(metrics_dir / "holdout_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_dir / 'holdout_metrics.json'}")

    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    ext = best_effort_ext()

    predictions_df = X_test.copy()
    if id_col in df.columns:
        test_indices = X_test.index
        predictions_df[id_col] = df.loc[test_indices, id_col].values
    predictions_df[f"{target}_true"] = y_test.values
    predictions_df[f"{target}_pred"] = y_pred
    predictions_df[f"{target}_pred_proba"] = y_pred_proba
    
    preds_path = tables_dir / f"holdout_predictions{ext}"
    write_tabular(predictions_df, preds_path)
    print(f"Saved predictions: {preds_path}")

    holdout_input = X_test.copy()
    if id_col in df.columns:
        holdout_input[id_col] = df.loc[test_indices, id_col].values
    
    input_path = tables_dir / f"holdout_input{ext}"
    write_tabular(holdout_input, input_path)
    print(f"Saved holdout input: {input_path}")

    schema_dir = run_dir / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)

    id_cols = [id_col] if id_col in df.columns else []
    schema = InputSchema.from_training_df(df=df, target=target, id_cols=id_cols)
    schema.dump(schema_dir / "input_schema.json")
    print(f"Saved schema: {schema_dir / 'input_schema.json'}")

    run_meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "target": target,
            "data_path": data_path,
            "test_size": test_size,
            "random_state": random_state,
            "split_strategy": split_strategy,
            "id_column": id_col,
        },
        "data_stats": {
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
            "target_distribution": y.value_counts().to_dict(),
        },
        "model": {"type": "LogisticRegression", "sklearn_version": "1.3+"},
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Saved run metadata: {run_dir / 'run_meta.json'}")

    model_path = run_dir / "model" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved model: {model_path}")

    registry_dir = Path("models/registry")
    registry_dir.mkdir(parents=True, exist_ok=True)
    with open(registry_dir / "latest.txt", "w") as f:
        f.write(run_id)
    print(f"Updated registry: {registry_dir / 'latest.txt'}")

    print(f"\nTraining complete! Run ID: {run_id}")
    print(f"All artifacts saved to: {run_dir}")

    return results
