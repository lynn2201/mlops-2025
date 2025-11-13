import argparse
import pandas as pd

from mlops_2025.features.features_computer import SimpleFeaturesComputer


def build_parser():
    p = argparse.ArgumentParser(description="Compute features for Titanic dataset")
    p.add_argument("--input", required=True, help="Processed input CSV")
    p.add_argument("--output", required=True, help="Output features CSV")
    return p


def main():
    args = build_parser().parse_args()

    print("Loading processed data...")
    df = pd.read_csv(args.input)

    print("Computing features...")
    fe = SimpleFeaturesComputer()
    df_features = fe.compute_features(df)

    print(f"Saving features to {args.output}")
    df_features.to_csv(args.output, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
