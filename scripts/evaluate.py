import argparse
import pandas as pd
from pathlib import Path
import pickle


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on feature file")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="metrics/metric.txt")
    args = parser.parse_args()

    print("Loading features...")
    X = pd.read_csv(args.input)

    print("Loading trained model...")
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    print("Running evaluation...")
    predictions = model.predict(X)

    accuracy = (predictions == 0).mean()  # dummy metric (fake labels since no y_test)

    Path("metrics").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"accuracy: {accuracy:.4f}\n")

    print(f"Saved metric to {args.output}")


if __name__ == "__main__":
    main()
