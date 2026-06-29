# train_model.py
import argparse
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_model(input_path: str, model_output_path: str, metrics_output_path: str):
    df = pd.read_csv(input_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    # Stratify keeps class proportions balanced between train/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, model_output_path)
    print(f"Model saved to: {model_output_path}")

    # Save metrics as JSON — useful for the pipeline UI and for your README
    metrics = {"accuracy": accuracy, "f1_macro": f1}
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_output_path", type=str, required=True)
    parser.add_argument("--metrics_output_path", type=str, required=True)
    args = parser.parse_args()
    train_model(args.input_path, args.model_output_path, args.metrics_output_path)