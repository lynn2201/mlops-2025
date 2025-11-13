import argparse
import pandas as pd

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Featurize Titanic data")
    parser.add_argument("--input", required=True, help="Path to processed CSV")
    parser.add_argument("--output", required=True, help="Path to save feature CSV")
    return parser

def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic features.
    This version is safe even if Titanic columns do not exist (fake data).
    """

    # Only create family_size if columns exist
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["family_size"] = df["SibSp"] + df["Parch"] + 1
    else:
        print("⚠️  'SibSp' or 'Parch' not found — skipping family_size feature")

    # Example categorical encoding (only if column exists)
    if "Sex" in df.columns:
        df["sex_is_female"] = (df["Sex"] == "female").astype(int)
    else:
        print("⚠️  'Sex' column not found — skipping sex_is_female feature")

    return df


def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)
    df_features = featurize(df)
    df_features.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
