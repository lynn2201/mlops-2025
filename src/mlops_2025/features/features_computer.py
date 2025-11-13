import pandas as pd
from .base_features_computer import BaseFeaturesComputer


class SimpleFeaturesComputer(BaseFeaturesComputer):
    """Convert categorical features into numeric encodings."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Example Titanic transformations
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

        if "Embarked" in df.columns:
            df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

        # Fill missing numeric fields
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].fillna(df[col].median())

        return df
