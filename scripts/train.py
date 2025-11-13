import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train(input_path: str, model_path: str):
    df = pd.read_csv(input_path)

    # Basic example: predict Survived using Pclass & Age
    X = df[["Pclass", "Age"]]
    y = df["Survived"]

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    train(args.input, args.model)
