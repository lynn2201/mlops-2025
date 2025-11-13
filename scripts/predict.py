import argparse
import pandas as pd
from pathlib import Path
import pickle

def main():
    parser = argparse.ArgumentParser(description="Run prediction using trained model")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print("Loading input features...")
    X = pd.read_csv(args.input)

    # Remove target column if it exists
    if "Survived" in X.columns:
        X = X.drop(columns=["Survived"])

    print("Loading model...")
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    print("Predicting...")
    preds = model.predict(X)

    out = pd.DataFrame({"prediction": preds})
    out.to_csv(args.output, index=False)

    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
