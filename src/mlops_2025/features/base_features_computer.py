from abc import ABC, abstractmethod
import pandas as pd


class BaseFeaturesComputer(ABC):
    """Abstract base class for feature engineering."""

    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takes a DataFrame and returns a feature matrix."""
        pass
