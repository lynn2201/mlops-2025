import argparse
import pandas as pd


def preprocess(input_path: str, output_path: str) -> None:
    print(f"Reading raw data from: {input_path}")
    df = pd.read_csv(input_path)

    # Minimal cleaning (you can customize later)
    df = df.dropna()

    print(f"Writing processed data to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic data")
    parser.add_argument("--input", required=True, help="Path to raw input CSV")
    parser.add_argument("--output", required=True, help="Path to processed output CSV")
    args = parser.parse_args()

    preprocess(args.input, args.output)


if __name__ == "__main__":
    main()
