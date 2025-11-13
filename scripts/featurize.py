import argparse
import pandas as pd


def featurize(input_path: str, output_path: str) -> None:
    print(f"Reading processed data from: {input_path}")
    df = pd.read_csv(input_path)

    # Example feature: FamilySize (if these columns exist)
    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    print(f"Writing features to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Create features for Titanic data")
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--output", required=True, help="Path to features CSV")
    args = parser.parse_args()

    featurize(args.input, args.output)


if __name__ == "__main__":
    main()
