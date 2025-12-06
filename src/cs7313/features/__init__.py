"""Feature extraction, dimensionality reduction, and clustering modules."""

from .extractor import FeatureExtractor
from .reduction import (
    DimensionalityReducer,
    PCAReducer,
    TSNEReducer,
    UMAPReducer,
    TruncatedSVDReducer,
    NMFReducer,
)
from .clustering import (
    ClusteringAlgorithm,
    KMeansClustering,
    DBSCANClustering,
    AgglomerativeClustering,
    SpectralClustering,
    GaussianMixtureClustering,
    HDBSCANClustering,
)

__all__ = [
    # Base classes
    "FeatureExtractor",
    "DimensionalityReducer",
    "ClusteringAlgorithm",
    # Dimensionality reduction
    "PCAReducer",
    "TSNEReducer",
    "UMAPReducer",
    "TruncatedSVDReducer",
    "NMFReducer",
    # Clustering
    "KMeansClustering",
    "DBSCANClustering",
    "AgglomerativeClustering",
    "SpectralClustering",
    "GaussianMixtureClustering",
    "HDBSCANClustering",
]
