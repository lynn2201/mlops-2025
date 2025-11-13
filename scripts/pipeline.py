import subprocess


def run_cmd(cmd: str) -> None:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    # 1. Preprocess: raw → processed
    run_cmd(
        "uv run python scripts/preprocess.py "
        "--input data/fake_titanic.csv "
        "--output data/processed.csv"
    )

    # 2. Featurize: processed → features
    run_cmd(
        "uv run python scripts/featurize.py "
        "--input data/processed.csv "
        "--output data/features.csv"
    )

    # 3. Train: features → model
    run_cmd(
        "uv run python scripts/train.py "
        "--input data/features.csv "
        "--model models/model.pkl"
    )

    # 4. Evaluate (optional, but nice): model + features → metrics
    run_cmd(
        "uv run python scripts/evaluate.py "
        "--model models/model.pkl "
        "--input data/features.csv "
        "--metrics_output metrics/metrics.json"
    )

    # 5. Predict: model + features → predictions
    run_cmd(
        "uv run python scripts/predict.py "
        "--model models/model.pkl "
        "--input data/features.csv "
        "--output data/predictions.csv"
    )
