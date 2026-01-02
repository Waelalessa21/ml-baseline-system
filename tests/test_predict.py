"""Tests for the predict command."""

import pandas as pd
import pytest
from pathlib import Path
from typer.testing import CliRunner
from ml_baseline.cli import app

runner = CliRunner()


@pytest.fixture
def sample_data():
    """Create sample data that matches the trained model's schema."""
    return pd.DataFrame({
        "user_id": ["u001", "u002", "u003"],
        "country": ["US", "GB", "CA"],
        "n_orders": [5, 10, 15],
        "total_amount": [25.5, 75.0, 125.0],
    })


@pytest.fixture
def sample_data_no_id():
    """Sample data without ID column."""
    return pd.DataFrame({
        "country": ["US", "GB", "CA"],
        "n_orders": [5, 10, 15],
        "total_amount": [25.5, 75.0, 125.0],
    })


@pytest.fixture
def sample_data_with_target():
    """Sample data with forbidden target column."""
    return pd.DataFrame({
        "user_id": ["u001", "u002", "u003"],
        "country": ["US", "GB", "CA"],
        "n_orders": [5, 10, 15],
        "total_amount": [25.5, 75.0, 125.0],
        "is_high_value": [0, 1, 1],  # Forbidden column
    })


@pytest.fixture
def sample_data_missing_feature():
    """Sample data missing a required feature."""
    return pd.DataFrame({
        "user_id": ["u001", "u002", "u003"],
        "country": ["US", "GB", "CA"],
        "n_orders": [5, 10, 15],
        # Missing total_amount
    })


def test_predict_writes_output_file(tmp_path, sample_data):
    """Test that predict command writes an output file."""
    # Create test input file
    input_file = tmp_path / "test_input.csv"
    sample_data.to_csv(input_file, index=False)
    
    # Create output directory
    output_file = tmp_path / "predictions.csv"
    
    # Run predict command
    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
        "--output-path", str(output_file),
    ])
    
    # Check command succeeded
    assert result.exit_code == 0, f"Command failed: {result.stdout}"
    
    # Check output file exists
    assert output_file.exists(), "Output file was not created"
    
    # Check output file has correct structure
    predictions = pd.read_csv(output_file)
    assert len(predictions) == 3, "Output should have 3 rows"
    assert "prediction" in predictions.columns, "Output should have 'prediction' column"
    assert "prediction_proba" in predictions.columns, "Output should have 'prediction_proba' column"


def test_predict_forbidden_column_error(tmp_path, sample_data_with_target):
    """Test that forbidden target column produces a clear error."""
    # Create test input file with forbidden column
    input_file = tmp_path / "test_input_with_target.csv"
    sample_data_with_target.to_csv(input_file, index=False)
    
    output_file = tmp_path / "predictions.csv"
    
    # Run predict command
    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
        "--output-path", str(output_file),
    ])
    
    # Check command failed
    assert result.exit_code == 1, "Command should fail with forbidden column"
    
    # Check error message is clear
    assert "Validation Error" in result.stdout or "Forbidden columns" in result.stdout, \
        f"Error message should mention forbidden columns. Got: {result.stdout}"
    assert "is_high_value" in result.stdout, \
        f"Error should mention the forbidden column name. Got: {result.stdout}"


def test_predict_missing_feature_error(tmp_path, sample_data_missing_feature):
    """Test that missing required feature produces a clear error."""
    # Create test input file missing a required feature
    input_file = tmp_path / "test_input_missing.csv"
    sample_data_missing_feature.to_csv(input_file, index=False)
    
    output_file = tmp_path / "predictions.csv"
    
    # Run predict command
    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
        "--output-path", str(output_file),
    ])
    
    # Check command failed
    assert result.exit_code == 1, "Command should fail with missing feature"
    
    # Check error message is clear
    assert "Validation Error" in result.stdout or "Missing required" in result.stdout, \
        f"Error message should mention missing features. Got: {result.stdout}"
    assert "total_amount" in result.stdout, \
        f"Error should mention the missing feature name. Got: {result.stdout}"


def test_predict_includes_optional_ids(tmp_path, sample_data):
    """Test that output includes optional ID columns if they exist in input."""
    # Create test input file with ID column
    input_file = tmp_path / "test_input_with_id.csv"
    sample_data.to_csv(input_file, index=False)
    
    output_file = tmp_path / "predictions.csv"
    
    # Run predict command
    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
        "--output-path", str(output_file),
    ])
    
    # Check command succeeded
    assert result.exit_code == 0, f"Command failed: {result.stdout}"
    
    # Check output includes ID column
    predictions = pd.read_csv(output_file)
    assert "user_id" in predictions.columns, "Output should include optional ID column"
    assert predictions["user_id"].tolist() == ["u001", "u002", "u003"], \
        "ID values should match input"


def test_predict_without_optional_ids(tmp_path, sample_data_no_id):
    """Test that predict works when optional ID columns are absent."""
    # Create test input file without ID column
    input_file = tmp_path / "test_input_no_id.csv"
    sample_data_no_id.to_csv(input_file, index=False)
    
    output_file = tmp_path / "predictions.csv"
    
    # Run predict command
    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
        "--output-path", str(output_file),
    ])
    
    # Check command succeeded
    assert result.exit_code == 0, f"Command failed: {result.stdout}"
    
    # Check output does not have ID column
    predictions = pd.read_csv(output_file)
    assert "user_id" not in predictions.columns, "Output should not have ID when input lacks it"
    assert "prediction" in predictions.columns, "Output should have predictions"
    assert len(predictions) == 3, "Output should have 3 rows"


def test_predict_default_output_path(tmp_path, sample_data, monkeypatch):
    """Test that predict uses default output path when not specified."""
    # Change to tmp directory
    monkeypatch.chdir(tmp_path)
    
    # Create test input file
    input_file = tmp_path / "test_input.csv"
    sample_data.to_csv(input_file, index=False)
    
    # Run predict command without output path
    result = runner.invoke(app, [
        "predict",
        "--run", "latest",
        "--input-path", str(input_file),
    ])
    
    # Check command succeeded
    assert result.exit_code == 0, f"Command failed: {result.stdout}"
    
    # Check default output file exists (should be in output/ directory)
    default_output = tmp_path / "output" / "predictions.csv"
    if not default_output.exists():
        # Try parquet extension
        default_output = tmp_path / "output" / "predictions.parquet"
    
    assert default_output.exists(), "Default output file should be created in output/ directory"





