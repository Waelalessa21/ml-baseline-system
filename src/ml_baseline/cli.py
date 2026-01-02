import typer
import pandas as pd
import numpy as np
import json
from pathlib import Path
from .train import train_model
from .predict import run_predict, resolve_run_dir

app = typer.Typer()


@app.command()
def hello():
    print("ML baseline CLI is working!")


@app.command()
def make_sample_data(seed: int = 42):
    np.random.seed(seed)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame(
        {
            "user_id": [f"u{i:03d}" for i in range(1, 101)],
            "country": np.random.choice(["US", "GB", "CA"], 100),
            "n_orders": np.random.randint(1, 20, 100),
            "total_amount": np.round(np.random.uniform(10, 200, 100), 2),
        }
    )
    data["is_high_value"] = (data["total_amount"] > 50).astype(int)

    data.to_csv("data/processed/features.csv", index=False)
    print(f"Sample data created at data/processed/features.csv, shape: {data.shape}")


@app.command()
def train(
    target: str = typer.Option(..., help="Target column name"),
    id_col: str = typer.Option("user_id", help="ID column to exclude from features"),
    data_path: str = typer.Option(
        "data/processed/features.csv", help="Path to features file"
    ),
    test_size: float = typer.Option(0.2, help="Proportion of data for test set"),
    random_state: int = typer.Option(42, help="Random seed for reproducibility"),
    split_strategy: str = typer.Option(
        "random", help="Split strategy: random, time, or group"
    ),
    time_col: str = typer.Option(None, help="Time column for time-based split"),
    group_col: str = typer.Option(None, help="Group column for group-based split"),
):
    try:
        train_model(
            data_path=data_path,
            target=target,
            id_col=id_col,
            test_size=test_size,
            random_state=random_state,
            split_strategy=split_strategy,
            time_col=time_col,
            group_col=group_col,
        )
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@app.command()
def predict(
    run: str = typer.Option(..., help="Run ID or 'latest'"),
    input_path: str = typer.Option(..., "--input", "--input-path", help="Path to input data file"),
    output_path: str = typer.Option(None, "--output", "--output-path", help="Path to save predictions (default: output/predictions.{csv|parquet})"),
):
    try:
        output_path_obj = Path(output_path) if output_path else None
        run_predict(
            run=run,
            input_path=Path(input_path),
            output_path=output_path_obj,
        )
    except AssertionError as e:
        print(f"Validation Error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


@app.command()
def show_run(
    run: str = typer.Argument("latest", help="Run ID or 'latest'"),
):
    """Display run metadata."""
    try:
        run_dir = resolve_run_dir(run)
        meta_path = run_dir / "run_meta.json"
        
        if not meta_path.exists():
            print(f"Error: run_meta.json not found in {run_dir}")
            raise typer.Exit(1)
        
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        print(json.dumps(meta, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
