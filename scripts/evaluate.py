import argparse
from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on labeled data")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="CSV file with features AND target column.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Survived",
        help="Name of the target column in the input file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model .pkl file.",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="metrics",
        help="Directory where evaluation metrics will be saved.",
    )
    args = parser.parse_args()

    print("Loading evaluation data...")
    df = pd.read_csv(args.input)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.input}")

    # Split features and target
    y_true = df[args.target]
    X = df.drop(columns=[args.target])

    print("Loading model...")
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    print("Running predictions...")
    y_pred = model.predict(X)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    # Make sure metrics directory exists
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / "evaluation.txt"

    print("Saving metrics...")
    with open(metrics_file, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    print(f"Accuracy: {acc:.4f}")
    print(f"Detailed report saved to {metrics_file}")


if __name__ == "__main__":
    main()
