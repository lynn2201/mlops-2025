import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


def evaluate(model_path: str, input_path: str, metrics_output: str | None = None) -> None:
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Reading evaluation data from: {input_path}")
    df = pd.read_csv(input_path)

    # Simple example: use the same features as in train.py
    if {"Pclass", "Age"}.issubset(df.columns):
        X = df[["Pclass", "Age"]]
    else:
        # Fallback: use all columns except the label
        X = df.drop(columns=["Survived"])

    y = df["Survived"]

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    print(f"Accuracy: {acc:.4f}")

    if metrics_output is not None:
        metrics_path = Path(metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({"accuracy": acc}, f)
        print(f"Metrics saved to {metrics_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Titanic model")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--input", required=True, help="Path to features CSV with labels")
    parser.add_argument(
        "--metrics_output",
        required=False,
        help="Optional path to save metrics as JSON",
    )
    args = parser.parse_args()

    evaluate(args.model, args.input, args.metrics_output)


if __name__ == "__main__":
    main()
