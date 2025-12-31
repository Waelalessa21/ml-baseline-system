import typer
import pandas as pd
import numpy as np
from pathlib import Path

app = typer.Typer()

@app.command()
def hello():
    print("ML baseline CLI is working!")

@app.command()
def make_sample_data(seed: int = 42):
    np.random.seed(seed)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    data = pd.DataFrame({
        "user_id": [f"u{i:03d}" for i in range(1, 11)],
        "country": np.random.choice(["US", "GB", "CA"], 10),
        "n_orders": np.random.randint(1, 10, 10),
        "total_amount": np.round(np.random.uniform(10, 100, 10), 2),
    })
    data["is_high_value"] = (data["total_amount"] > 50).astype(int)
    
    sliced = data[:5]
    sliced.to_csv("data/processed/features.csv", index=False)
    print(f"Sample data created at data/processed/features.csv, shape: {sliced.shape}")

if __name__ == "__main__":
    app()
