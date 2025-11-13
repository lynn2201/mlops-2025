import argparse
from pathlib import Path

import joblib
import pandas as pd


def predict(model_path: str, input_path: str, output_path: str) -> None:
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Reading features from: {input_path}")
    df = pd.read_csv(input_path)

    # Use same features as in train.py.
    # If 'Survived' is present (train-like file), drop it before predicting.
    X = df.copy()
    if "Survived" in X.columns:
        X = X.drop(columns=["Survived"])

    if {"Pclass", "Age"}.issubset(X.columns):
        X = X[["Pclass", "Age"]]

    preds = model.predict(X)

    out_df = df.copy()
    out_df["prediction"] = preds

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained Titanic model")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--input", required=True, help="Path to features CSV (no label needed)")
    parser.add_argument("--output", required=True, help="Path to CSV with predictions")
    args = parser.parse_args()

    predict(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
