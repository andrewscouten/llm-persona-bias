import numpy as np
from abc import ABC


class FeatureExtractor(ABC):
    """Abstract base class for feature extraction methods."""

    def __init__(self, algorithm, fit_transform_fn) -> None:
        """
        Initialize the feature extractor.

        Args:
            algorithm: An instance of the feature extraction method.
            fit_transform_fn: A callable for fitting and transforming data that
                                the algorithm provides.
        """
        if not callable(fit_transform_fn):
            raise ValueError(f"fit_transform_fn must be a callable.")
        self.algorithm = algorithm
        self.fit_transform_fn = fit_transform_fn

    def __call__(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fit the feature extractor and transform data.

        Args:
            data: Input data array
            **kwargs: Additional parameters for the feature extraction method

        Returns:
            Extracted features
        """
        return self.fit_transform_fn(data, **kwargs)

