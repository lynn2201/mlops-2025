import argparse
import pandas as pd

def featurize(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    # Minimal feature engineering example:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    featurize(args.input, args.output)
