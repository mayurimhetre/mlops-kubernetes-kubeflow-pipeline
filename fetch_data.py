# fetch_data.py
import argparse
import pandas as pd
from sklearn.datasets import load_iris

def fetch_data(output_path: str):
    """Load the Iris dataset and save it as a CSV."""
    iris = load_iris(as_frame=True)
    df = iris.frame  # includes feature columns + 'target' column

    df.to_csv(output_path, index=False)
    print(f"Fetched {len(df)} rows, {len(df.columns)} columns.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    fetch_data(args.output_path)