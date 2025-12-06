import umap
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.manifold import TSNE

from .extractor import FeatureExtractor


class DimensionalityReducer(FeatureExtractor):
    """Class for dimensionality reduction methods."""

    def __init__(self, reducer) -> None:
        """
        Initialize the dimensionality reducer.

        Args:
            reducer: An instance of the reduction method. Must have
                        a `fit_transform` method.
        """
        fit_transform = getattr(reducer, "fit_transform", None)
        if not callable(fit_transform):
            raise ValueError("Reducer must have a callable 'fit_transform' method.")
        super().__init__(reducer, fit_transform)


class PCAReducer(DimensionalityReducer):
    """Principal Component Analysis for dimensionality reduction.
    
    PCA is a linear dimensionality reduction technique that projects data
    onto orthogonal components that capture maximum variance.
    """

    def __init__(self, n_components=2, **kwargs):
        """
        Initialize PCA reducer.
        
        Args:
            n_components: Number of components to keep (default: 2)
            **kwargs: Additional parameters passed to PCA
        """
        reducer = PCA(n_components=n_components, **kwargs)
        super().__init__(reducer)


class TSNEReducer(DimensionalityReducer):
    """t-Distributed Stochastic Neighbor Embedding for dimensionality reduction.
    
    t-SNE is a non-linear technique well-suited for visualizing high-dimensional
    data by preserving local structure. Good for finding clusters in text embeddings.
    """

    def __init__(self, n_components=2, perplexity=30.0, learning_rate='auto', 
                 max_iter=1000, random_state=42, **kwargs):
        """
        Initialize t-SNE reducer.
        
        Args:
            n_components: Number of dimensions for output (default: 2)
            perplexity: Balance between local and global aspects (default: 30.0)
            learning_rate: Learning rate for optimization (default: 'auto')
            n_iter: Number of iterations for optimization (default: 1000)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to TSNE
        """
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        super().__init__(reducer)


class UMAPReducer(DimensionalityReducer):
    """Uniform Manifold Approximation and Projection for dimensionality reduction.
    
    UMAP is a non-linear technique that preserves both local and global structure.
    Often faster than t-SNE and better at preserving global structure. Excellent
    for text embeddings and bias analysis.
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, 
                 metric='cosine', random_state=42, **kwargs):
        """
        Initialize UMAP reducer.
        
        Args:
            n_components: Number of dimensions for output (default: 2)
            n_neighbors: Size of local neighborhood (default: 15)
            min_dist: Minimum distance between points in low-dim space (default: 0.1)
            metric: Distance metric to use (default: 'cosine' for text)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to UMAP
        """
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            **kwargs
        )
        super().__init__(reducer)


class TruncatedSVDReducer(DimensionalityReducer):
    """Truncated Singular Value Decomposition for dimensionality reduction.
    
    Similar to PCA but works with sparse matrices, making it ideal for
    bag-of-words or TF-IDF representations of text data.
    """

    def __init__(self, n_components=2, random_state=42, **kwargs):
        """
        Initialize Truncated SVD reducer.
        
        Args:
            n_components: Number of components to keep (default: 2)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to TruncatedSVD
        """
        reducer = TruncatedSVD(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
        super().__init__(reducer)


class NMFReducer(DimensionalityReducer):
    """Non-negative Matrix Factorization for dimensionality reduction.
    
    NMF decomposes data into non-negative components, useful for topic modeling
    and finding interpretable latent features in text data.
    """

    def __init__(self, n_components=2, init='nndsvda', random_state=42, **kwargs):
        """
        Initialize NMF reducer.
        
        Args:
            n_components: Number of components to extract (default: 2)
            init: Initialization method (default: 'nndsvda')
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters passed to NMF
        """
        reducer = NMF(
            n_components=n_components,
            init=init,
            random_state=random_state,
            **kwargs
        )
        super().__init__(reducer)

