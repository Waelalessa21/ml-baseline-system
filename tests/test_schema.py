import pandas as pd
import pytest
from ml_baseline.schema import InputSchema, validate_and_align


@pytest.fixture
def sample_schema():
    return InputSchema(
        required_feature_columns=["country", "n_orders", "total_amount"],
        feature_dtypes={
            "country": "object",
            "n_orders": "int64",
            "total_amount": "float64",
        },
        optional_id_columns=["user_id"],
        forbidden_columns=["is_high_value"],
    )


def test_validate_forbidden_columns_raises_error(sample_schema):
    df = pd.DataFrame({
        "user_id": ["u001", "u002"],
        "country": ["US", "GB"],
        "n_orders": [5, 10],
        "total_amount": [25.5, 75.0],
        "is_high_value": [0, 1],  # Forbidden!
    })
    
    with pytest.raises(AssertionError) as exc_info:
        validate_and_align(df, sample_schema)
    
    assert "Forbidden columns" in str(exc_info.value)
    assert "is_high_value" in str(exc_info.value)


def test_validate_missing_required_columns_raises_error(sample_schema):
    df = pd.DataFrame({
        "user_id": ["u001", "u002"],
        "country": ["US", "GB"],
        "n_orders": [5, 10],
        # Missing total_amount!
    })
    
    with pytest.raises(AssertionError) as exc_info:
        validate_and_align(df, sample_schema)
    
    assert "Missing required" in str(exc_info.value)
    assert "total_amount" in str(exc_info.value)


def test_validate_returns_features_in_schema_order(sample_schema):
    df = pd.DataFrame({
        "total_amount": [25.5, 75.0, 125.0], 
        "user_id": ["u001", "u002", "u003"],
        "n_orders": [5, 10, 15], 
        "country": ["US", "GB", "CA"], 
    })
    
    X, ids = validate_and_align(df, sample_schema)
    assert list(X.columns) == ["country", "n_orders", "total_amount"]
    assert len(X) == 3


def test_validate_returns_optional_ids_when_present(sample_schema):
    df = pd.DataFrame({
        "user_id": ["u001", "u002", "u003"],
        "country": ["US", "GB", "CA"],
        "n_orders": [5, 10, 15],
        "total_amount": [25.5, 75.0, 125.0],
    })
    
    X, ids = validate_and_align(df, sample_schema)
    
    assert "user_id" in ids.columns
    assert ids["user_id"].tolist() == ["u001", "u002", "u003"]
    assert len(ids) == 3


def test_validate_returns_empty_ids_when_absent(sample_schema):
    df = pd.DataFrame({
        "country": ["US", "GB", "CA"],
        "n_orders": [5, 10, 15],
        "total_amount": [25.5, 75.0, 125.0],
    })
    
    X, ids = validate_and_align(df, sample_schema)
    
    assert len(ids.columns) == 0
    assert len(ids) == 3


def test_validate_dtype_conversion():
    schema = InputSchema(
        required_feature_columns=["age", "score", "name"],
        feature_dtypes={
            "age": "int64",
            "score": "float64",
            "name": "object",
        },
        optional_id_columns=[],
        forbidden_columns=[],
    )
    
    df = pd.DataFrame({
        "age": ["25", "30", "35"],
        "score": ["85.5", "90.0", "95.5"],
        "name": ["Alice", "Bob", "Charlie"],
    })
    
    X, ids = validate_and_align(df, schema)
    
    assert pd.api.types.is_numeric_dtype(X["age"])
    assert pd.api.types.is_numeric_dtype(X["score"])
    assert pd.api.types.is_string_dtype(X["name"])


def test_validate_allows_extra_columns_not_in_schema(sample_schema):
    df = pd.DataFrame({
        "user_id": ["u001", "u002"],
        "country": ["US", "GB"],
        "n_orders": [5, 10],
        "total_amount": [25.5, 75.0],
        "extra_column": ["foo", "bar"],  
    })
    
    X, ids = validate_and_align(df, sample_schema)
    
    assert "extra_column" not in X.columns
    assert len(X) == 2


def test_schema_round_trip(tmp_path):
    schema = InputSchema(
        required_feature_columns=["a", "b", "c"],
        feature_dtypes={"a": "int64", "b": "float64", "c": "object"},
        optional_id_columns=["id"],
        forbidden_columns=["target"],
    )
    
    path = tmp_path / "schema.json"
    schema.dump(path)
    
    loaded_schema = InputSchema.load(path)
    
    assert loaded_schema.required_feature_columns == schema.required_feature_columns
    assert loaded_schema.feature_dtypes == schema.feature_dtypes
    assert loaded_schema.optional_id_columns == schema.optional_id_columns
    assert loaded_schema.forbidden_columns == schema.forbidden_columns






