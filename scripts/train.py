import argparse
import pickle
import pandas as pd

from mlops_2025.model.model import SimpleRandomForestModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest model on Titanic features"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to features CSV (must contain a 'Survived' target column)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path where the trained model will be saved (e.g. models/model.pkl)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("Loading feature data...")
    df = pd.read_csv(args.input)

    # Split into X (features) and y (target)
    if "Survived" not in df.columns:
        raise ValueError("Expected a 'Survived' column in the input features CSV.")

    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    print("Training Random Forest model...")
    model = SimpleRandomForestModel()
    model.train(X, y)

    print(f"Saving trained model to {args.model} ...")
    with open(args.model, "wb") as f:
        pickle.dump(model, f)

    print("Done.")


if __name__ == "__main__":
    main()
