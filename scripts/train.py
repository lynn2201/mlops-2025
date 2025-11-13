import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train(input_path: str, model_path: str) -> None:
    print(f"Reading features from: {input_path}")
    df = pd.read_csv(input_path)

    # Very simple example: adjust column names to your CSV
    X = df[["Pclass", "Age"]]  # change if your columns differ
    y = df["Survived"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a simple Titanic model")
    parser.add_argument("--input", required=True, help="Path to features CSV")
    parser.add_argument("--model", required=True, help="Path to output model .pkl")
    args = parser.parse_args()

    train(args.input, args.model)


if __name__ == "__main__":
    main()
