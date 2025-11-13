import pandas as pd
from .base_preprocessor import BasePreprocessor


class SimplePreprocessor(BasePreprocessor):
    """Simple Titanic preprocessor."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Minimal cleaning (this can get more complex later)
        df = df.dropna()
        return df
