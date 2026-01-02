"""Prediction logic for trained models."""

from __future__ import annotations

import joblib
from pathlib import Path

import pandas as pd

from .io import read_tabular, write_tabular, best_effort_ext
from .schema import InputSchema, validate_and_align


def resolve_run_dir(run: str, root: Path = Path(".")) -> Path:
    if run == "latest":
        registry_file = root / "models" / "registry" / "latest.txt"
        if not registry_file.exists():
            raise ValueError("No 'latest' run found. Train a model first.")
        run_id = registry_file.read_text().strip()
    else:
        run_id = run
    
    run_dir = root / "models" / "runs" / run_id
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")
    
    return run_dir


def run_predict(
    run: str,
    input_path: Path,
    output_path: Path | None = None,
    root: Path = Path("."),
) -> pd.DataFrame:
   
    run_dir = resolve_run_dir(run, root=root)
    run_id = run_dir.name
    print(f"Using run: {run_id}")
    
    # Load model artifacts
    model_path = run_dir / "model" / "model.joblib"
    schema_path = run_dir / "schema" / "input_schema.json"
    
    if not model_path.exists():
        raise ValueError(f"Model not found: {model_path}")
    if not schema_path.exists():
        raise ValueError(f"Schema not found: {schema_path}")
    
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading schema from: {schema_path}")
    schema = InputSchema.load(schema_path)
    
    print(f"Loading input data from: {input_path}")
    input_df = read_tabular(input_path)
    print(f"Input shape: {input_df.shape}")
    
    print("Validating input schema...")
    X, passthrough = validate_and_align(input_df, schema)
    print(f"Validation successful. Features shape: {X.shape}")
    
    print("Making predictions...")
    predictions = model.predict(X)
    pred_proba = model.predict_proba(X)[:, 1]
    
    output_df = passthrough.copy()
    output_df["prediction"] = predictions
    output_df["prediction_proba"] = pred_proba
    
    if output_path is None:
        ext = best_effort_ext()
        output_path = root / "output" / f"predictions{ext}"
        
    print(f"Writing predictions to: {output_path}")
    write_tabular(output_df, output_path)
    print(f"Predictions saved! Shape: {output_df.shape}")
    print(f"Columns: {list(output_df.columns)}")
    
    return output_df





