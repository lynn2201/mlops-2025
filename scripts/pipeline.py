import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    # 1. Preprocess
    run_cmd("python scripts/preprocess.py --input data/fake_titanic.csv --output data/preprocessed.csv")

    # 2. Featurize
    run_cmd("python scripts/featurize.py --input data/preprocessed.csv --output data/features.csv")

    # 3. Train
    run_cmd("python scripts/train.py --input data/features.csv --model model.pkl")

    # 4. Evaluate
    run_cmd("python scripts/evaluate.py --input data/features.csv --model model.pkl")

    # 5. Predict
    run_cmd("python scripts/predict.py --input data/features.csv --model model.pkl --output predictions.csv")
