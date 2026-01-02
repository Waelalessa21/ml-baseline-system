"""Tests for the predict command."""

import pandas as pd
import pytest
from pathlib import Path
from typer.testing import CliRunner
from ml_baseline.cli import app

runner = CliRunner()


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "user_id": ["u001", "u002", "u003"],
        "country": ["US", "GB", "CA"],
        "n_orders": [5, 10, 15],
        "total_amount": [25.5, 75.0, 125.0],
    })


@pytest.fixture
def sample_data_no_id():
    return pd.DataFrame({
        "country": ["US", "GB", "CA"],
        "n_orders": [5, 10, 15],
        "total_amount": [25.5, 75.0, 125.0],
    })


@pytest.fixture
def sample_data_with_target():
    return pd.DataFrame({
        "user_id": ["u001", "u002", "u003"],
        "country": ["US", "GB", "CA"],
        "n_orders": [5, 10, 15],
        "total_amount": [25.5, 75.0, 125.0],
        "is_high_value": [0, 1, 1],  # Forbidden column
    })


@pytest.fixture
def sample_data_missing_feature():
    return pd.DataFrame({
        "user_id": ["u001", "u002", "u003"],
        "country": ["US", "GB", "CA"],
        "n_orders": [5, 10, 15],
    })


def test_predict_writes_output_file(tmp_path, sample_data):
    input_file = tmp_path / "test_input.csv"
    sample_data.to_csv(input_file, index=False)
    
    output_file = tmp_path / "predictions.csv"
    
    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
        "--output-path", str(output_file),
    ])
    
    assert result.exit_code == 0, f"Command failed: {result.stdout}"
    
    assert output_file.exists(), "Output file was not created"
    
    predictions = pd.read_csv(output_file)
    assert len(predictions) == 3, "Output should have 3 rows"
    assert "prediction" in predictions.columns, "Output should have 'prediction' column"
    assert "prediction_proba" in predictions.columns, "Output should have 'prediction_proba' column"


def test_predict_forbidden_column_error(tmp_path, sample_data_with_target):
 
    input_file = tmp_path / "test_input_with_target.csv"
    sample_data_with_target.to_csv(input_file, index=False)
    
    output_file = tmp_path / "predictions.csv"
    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
        "--output-path", str(output_file),
    ])
    assert result.exit_code == 1, "Command should fail with forbidden column"
    
    assert "Validation Error" in result.stdout or "Forbidden columns" in result.stdout, \
        f"Error message should mention forbidden columns. Got: {result.stdout}"
    assert "is_high_value" in result.stdout, \
        f"Error should mention the forbidden column name. Got: {result.stdout}"


def test_predict_missing_feature_error(tmp_path, sample_data_missing_feature):
    input_file = tmp_path / "test_input_missing.csv"
    sample_data_missing_feature.to_csv(input_file, index=False)
    
    output_file = tmp_path / "predictions.csv"

    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
        "--output-path", str(output_file),
    ])
    
    assert result.exit_code == 1, "Command should fail with missing feature"
    
    assert "Validation Error" in result.stdout or "Missing required" in result.stdout, \
        f"Error message should mention missing features. Got: {result.stdout}"
    assert "total_amount" in result.stdout, \
        f"Error should mention the missing feature name. Got: {result.stdout}"


def test_predict_includes_optional_ids(tmp_path, sample_data):
    input_file = tmp_path / "test_input_with_id.csv"
    sample_data.to_csv(input_file, index=False)
    
    output_file = tmp_path / "predictions.csv"
    
    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
        "--output-path", str(output_file),
    ])
    
    assert result.exit_code == 0, f"Command failed: {result.stdout}"
    
    predictions = pd.read_csv(output_file)
    assert "user_id" in predictions.columns, "Output should include optional ID column"
    assert predictions["user_id"].tolist() == ["u001", "u002", "u003"], \
        "ID values should match input"


def test_predict_without_optional_ids(tmp_path, sample_data_no_id):
    input_file = tmp_path / "test_input_no_id.csv"
    sample_data_no_id.to_csv(input_file, index=False)
    
    output_file = tmp_path / "predictions.csv"
    
    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
        "--output-path", str(output_file),
    ])
    
    assert result.exit_code == 0, f"Command failed: {result.stdout}"
    
    predictions = pd.read_csv(output_file)
    assert "user_id" not in predictions.columns, "Output should not have ID when input lacks it"
    assert "prediction" in predictions.columns, "Output should have predictions"
    assert len(predictions) == 3, "Output should have 3 rows"


def test_predict_default_output_path(tmp_path, sample_data):
    input_file = tmp_path / "test_input.csv"
    sample_data.to_csv(input_file, index=False)
    
    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
    ])
    
    assert result.exit_code == 0, f"Command failed: {result.stdout}"

    assert "output/predictions" in result.stdout, \
        "Output should be written to default output/predictions path"
