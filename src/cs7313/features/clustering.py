import hdbscan
from sklearn.cluster import (
    KMeans as SKLearnKMeans,
    DBSCAN as SKLearnDBSCAN,
    AgglomerativeClustering as SKLearnAgglomerativeClustering,
    SpectralClustering as SKLearnSpectralClustering,
)
from sklearn.mixture import GaussianMixture

from .extractor import FeatureExtractor


class ClusteringAlgorithm(FeatureExtractor):
    """Class for clustering algorithms."""

    def __init__(self, algorithm, fit_predict_fn):
        fit_predict = getattr(algorithm, "fit_predict", None)
        if not callable(fit_predict):
            raise ValueError("Algorithm must have a callable 'fit_predict' method.")
        super().__init__(algorithm, fit_predict_fn)


class KMeansClustering(ClusteringAlgorithm):
    """K-Means clustering algorithm.
    
    K-Means partitions data into k clusters by minimizing within-cluster variance.
    Useful for identifying distinct groups in text embeddings.
    """

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, 
                 random_state=42, **kwargs):
        """
        Initialize K-Means clusterer.
        
        Args:
            n_clusters: Number of clusters to form (default: 8)
            init: Method for initialization (default: 'k-means++')
            n_init: Number of times to run with different seeds (default: 10)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to KMeans
        """
        algorithm = SKLearnKMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            random_state=random_state,
            **kwargs
        )
        super().__init__(algorithm, algorithm.fit_predict)


class DBSCANClustering(ClusteringAlgorithm):
    """Density-Based Spatial Clustering of Applications with Noise.
    
    DBSCAN finds clusters of arbitrary shape and identifies outliers.
    Excellent for finding biased language patterns that form dense regions
    in embedding space.
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', **kwargs):
        """
        Initialize DBSCAN clusterer.
        
        Args:
            eps: Maximum distance between samples in a neighborhood (default: 0.5)
            min_samples: Minimum samples in a neighborhood for core point (default: 5)
            metric: Distance metric to use (default: 'euclidean')
            **kwargs: Additional parameters passed to DBSCAN
        """
        algorithm = SKLearnDBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            **kwargs
        )
        super().__init__(algorithm, algorithm.fit_predict)


class AgglomerativeClustering(ClusteringAlgorithm):
    """Agglomerative Hierarchical Clustering.
    
    Builds a hierarchy of clusters using a bottom-up approach.
    Useful for understanding hierarchical relationships in biased language.
    """

    def __init__(self, n_clusters=2, linkage='ward', metric='euclidean', **kwargs):
        """
        Initialize Agglomerative clusterer.
        
        Args:
            n_clusters: Number of clusters to find (default: 2)
            linkage: Linkage criterion to use (default: 'ward')
            metric: Distance metric (default: 'euclidean')
            **kwargs: Additional parameters passed to AgglomerativeClustering
        """
        algorithm = SKLearnAgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
            **kwargs
        )
        super().__init__(algorithm, algorithm.fit_predict)


class SpectralClustering(ClusteringAlgorithm):
    """Spectral Clustering.
    
    Uses eigenvalues of similarity matrix for dimensionality reduction
    before clustering. Effective for data with non-convex cluster shapes.
    """

    def __init__(self, n_clusters=8, affinity='rbf', random_state=42, **kwargs):
        """
        Initialize Spectral clusterer.
        
        Args:
            n_clusters: Number of clusters to find (default: 8)
            affinity: Kernel to use for affinity matrix (default: 'rbf')
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to SpectralClustering
        """
        algorithm = SKLearnSpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=random_state,
            **kwargs
        )
        super().__init__(algorithm, algorithm.fit_predict)


class GaussianMixtureClustering(ClusteringAlgorithm):
    """Gaussian Mixture Model clustering.
    
    Assumes data is generated from a mixture of Gaussian distributions.
    Provides soft clustering (probabilities) and can model complex distributions
    in biased text embeddings.
    """

    def __init__(self, n_components=2, covariance_type='full', random_state=42, **kwargs):
        """
        Initialize Gaussian Mixture clusterer.
        
        Args:
            n_components: Number of mixture components (default: 2)
            covariance_type: Type of covariance parameters (default: 'full')
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to GaussianMixture
        """
        algorithm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            **kwargs
        )
        # GaussianMixture uses predict instead of fit_predict
        def fit_predict_wrapper(X, y=None):
            algorithm.fit(X)
            return algorithm.predict(X)
        super().__init__(algorithm, fit_predict_wrapper)


class HDBSCANClustering(ClusteringAlgorithm):
    """Hierarchical Density-Based Spatial Clustering.
    
    Extension of DBSCAN that builds a hierarchy and extracts clusters
    of varying densities. Excellent for finding clusters of biased language
    at multiple scales without requiring a distance threshold.
    """

    def __init__(self, min_cluster_size=5, min_samples=None, metric='euclidean', 
                 cluster_selection_method='eom', **kwargs):
        """
        Initialize HDBSCAN clusterer.
        
        Args:
            min_cluster_size: Minimum size of clusters (default: 5)
            min_samples: Conservative cluster membership (default: None)
            metric: Distance metric to use (default: 'euclidean')
            cluster_selection_method: Method for selecting clusters (default: 'eom')
            **kwargs: Additional parameters passed to HDBSCAN
        """
        algorithm = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            **kwargs
        )
        super().__init__(algorithm, algorithm.fit_predict)