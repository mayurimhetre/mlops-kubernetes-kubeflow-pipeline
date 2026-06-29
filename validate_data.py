# validate_data.py
import argparse
import sys
import pandas as pd

EXPECTED_COLUMNS = [
    "sepal length (cm)", "sepal width (cm)",
    "petal length (cm)", "petal width (cm)", "target"
]
EXPECTED_CLASSES = {0, 1, 2}
MIN_ROWS = 50  # sanity floor — Iris should always be ~150

def validate_data(input_path: str):
    df = pd.read_csv(input_path)
    errors = []

    # 1. Schema check — right columns, right names
    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(f"Column mismatch. Got: {list(df.columns)}")

    # 2. Row count check — did we actually get real data?
    if len(df) < MIN_ROWS:
        errors.append(f"Too few rows: {len(df)} (expected >= {MIN_ROWS})")

    # 3. Null check — no missing values allowed
    if df.isnull().values.any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"Null values found in columns: {null_cols}")

    # 4. Range check — feature values should be positive, sane measurements
    feature_cols = EXPECTED_COLUMNS[:-1]  # everything except 'target'
    for col in feature_cols:
        if col in df.columns:
            if (df[col] <= 0).any() or (df[col] > 20).any():
                errors.append(f"Out-of-range values in '{col}' (expected 0-20 cm)")

    # 5. Target/class check — only valid class labels
    if "target" in df.columns:
        unexpected = set(df["target"].unique()) - EXPECTED_CLASSES
        if unexpected:
            errors.append(f"Unexpected target classes: {unexpected}")

    # Report
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)  # non-zero exit = pipeline stops here
    else:
        print(f"VALIDATION PASSED: {len(df)} rows, schema OK, no nulls, classes OK.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    args = parser.parse_args()
    validate_data(args.input_path)