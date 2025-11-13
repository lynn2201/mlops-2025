import pandas as pd
from .base_features_computer import BaseFeaturesComputer


class SimpleFeaturesComputer(BaseFeaturesComputer):
    """Simple Titanic feature engineering."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()

        # very minimal example features – adapt later if needed
        if "Sex" in features.columns:
            features["Sex_is_female"] = (features["Sex"] == "female").astype(int)

        if "Age" in features.columns:
            features["Age"] = features["Age"].fillna(features["Age"].median())

        for col in ["Name", "Ticket", "Cabin"]:
            if col in features.columns:
                features = features.drop(columns=[col])

        return features
