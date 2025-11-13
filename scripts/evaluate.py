import argparse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def evaluate(model_path: str, input_path: str):
    df = pd.read_csv(input_path)

    X = df[["Pclass", "Age"]]
    y = df["Survived"]

    model = joblib.load(model_path)
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    evaluate(args.model, args.input)
