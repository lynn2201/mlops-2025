import argparse
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Titanic data")
    parser.add_argument("--input", required=True, help="Path to raw input CSV")
    parser.add_argument("--output", required=True, help="Path to save processed CSV")
    return parser


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: put your Titanic preprocessing logic here later.
    For now, we just return the dataframe unchanged.
    """
    return df


def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)
    df_processed = preprocess(df)
    df_processed.to_csv(args.output, index=False)
    print(f"Saved processed data to: {args.output}")



if __name__ == "__main__":
    main()
