import argparse
import pandas as pd

from mlops_2025.preprocessing.preprocessor import SimplePreprocessor


def build_parser():
    p = argparse.ArgumentParser(description="Preprocess raw Titanic data")
    p.add_argument("--input", required=True, help="Path to input raw CSV")
    p.add_argument("--output", required=True, help="Path to output processed CSV")
    return p


def main():
    args = build_parser().parse_args()

    print("Loading raw data...")
    df = pd.read_csv(args.input)

    print("Applying preprocessing...")
    pre = SimplePreprocessor()
    df_processed = pre.process(df)

    print(f"Saving processed data to {args.output}")
    df_processed.to_csv(args.output, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
