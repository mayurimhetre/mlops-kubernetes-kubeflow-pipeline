# pipeline.py
from kfp import dsl
from kfp import compiler

# ---------- Component 1: Fetch ----------
@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas", "scikit-learn"])
def fetch_data(output_csv: dsl.Output[dsl.Dataset]):
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(output_csv.path, index=False)
    print(f"Fetched {len(df)} rows.")


# ---------- Component 2: Validate ----------
@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas"])
def validate_data(input_csv: dsl.Input[dsl.Dataset]):
    import pandas as pd
    import sys

    df = pd.read_csv(input_csv.path)
    errors = []

    expected_columns = [
        "sepal length (cm)", "sepal width (cm)",
        "petal length (cm)", "petal width (cm)", "target"
    ]
    if list(df.columns) != expected_columns:
        errors.append(f"Column mismatch: {list(df.columns)}")
    if len(df) < 50:
        errors.append(f"Too few rows: {len(df)}")
    if df.isnull().values.any():
        errors.append("Null values found")
    feature_cols = expected_columns[:-1]
    for col in feature_cols:
        if (df[col] <= 0).any() or (df[col] > 20).any():
            errors.append(f"Out-of-range values in '{col}'")
    if set(df["target"].unique()) - {0, 1, 2}:
        errors.append("Unexpected target classes")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"VALIDATION PASSED: {len(df)} rows OK.")


# ---------- Component 3: Train ----------
@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas", "scikit-learn", "joblib"])
def train_model(
    input_csv: dsl.Input[dsl.Dataset],
    model_output: dsl.Output[dsl.Model],
    accuracy_metric: dsl.Output[dsl.Metrics],
):
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    df = pd.read_csv(input_csv.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    joblib.dump(model, model_output.path)
    accuracy_metric.log_metric("accuracy", accuracy)
    accuracy_metric.log_metric("f1_macro", f1)


# ---------- Pipeline definition: wires the DAG ----------
@dsl.pipeline(name="iris-fetch-validate-train", description="Fetch, validate, and train on Iris data")
def iris_pipeline():
    fetch_task = fetch_data()
    validate_task = validate_data(input_csv=fetch_task.outputs["output_csv"])
    train_task = train_model(input_csv=fetch_task.outputs["output_csv"])
    train_task.after(validate_task)  # train only runs if validate succeeds


if __name__ == "__main__":
    compiler.Compiler().compile(iris_pipeline, "iris_pipeline.yaml")
    print("Compiled to iris_pipeline.yaml")