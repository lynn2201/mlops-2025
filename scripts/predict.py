import argparse
import pandas as pd
import joblib

def predict(model_path: str, input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    X = df[["Pclass", "Age"]]

    model = joblib.load(model_path)
    preds = model.predict(X)

    df["Prediction"] = preds
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predict(args.model, args.input, args.output)
