"""Module for loading and handling the marked personas dataset."""

from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np


class MarkedPersonasDataset():
    """Class to handle loading of the marked personas dataset."""

    def __init__(self, data_path: Union[str, Path]):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert data_path.exists(), f"Data path {str(data_path)} does not exist."

        self.data_path = data_path
        df = pd.read_csv(str(data_path))
        self._data = df

    def get_hover_text(self) -> pd.Series:
        """Get hover text for each entry in the dataset."""
        return self._data['text'].apply(
            lambda x: str(x)[:200] + '...' if len(str(x)) > 200 else str(x))

    def get_texts(self) -> np.ndarray:
        """Get the texts from the dataset."""
        return self._data['text'].to_numpy()

    def get_labels(self) -> np.ndarray:
        """Get the labels from the dataset."""
        return self._data['label'].to_numpy()

    def get_data(self) -> pd.DataFrame:
        """Get the entire dataset as a DataFrame."""
        return self._data
