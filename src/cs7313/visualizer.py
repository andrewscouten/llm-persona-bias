import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    """Class for visualizing embeddings in 2D and 3D using Plotly."""

    def __init__(
        self,
        params_2d: dict = None,
        params_3d: dict = None
    ) -> None:
        """
        Initialize the Visualizer.

        Args:
            params_2d: Default parameters for 2D visualization
            params_3d: Default parameters for 3D visualization
        """
        self.params_2d = params_2d if params_2d is not None else {}
        self.params_3d = params_3d if params_3d is not None else {}

    def _visualize_3d(
        self, 
        embeddings: np.ndarray, 
        **kwargs
    ) -> go.Figure:
        """
        Visualize embeddings in 3D using Plotly.

        Args:
            embeddings: 3D embeddings for visualization
            **kwargs: Additional parameters for Plotly scatter_3d
        """
        if embeddings.shape[1] != 3:
            raise ValueError("Embeddings must be 3-dimensional for 3D visualization.")

        fig = px.scatter_3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            **self.params_3d,
            **kwargs
        )
        return fig

    def _visualize_2d(
        self, 
        embeddings: np.ndarray, 
        **kwargs
    ) -> go.Figure:
        """
        Visualize embeddings in 2D using Plotly.

        Args:
            embeddings: 2D embeddings for visualization
            **kwargs: Additional parameters for Plotly scatter
        """
        if embeddings.shape[1] != 2:
            raise ValueError("Embeddings must be 2-dimensional for 2D visualization.")

        fig = go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            **self.params_2d,
            **kwargs
        )
        return fig

    def visualize(
        self,
        embeddings: np.ndarray, 
        show: bool = False,
        **kwargs
    ) -> go.Figure:
        """
        Visualize embeddings based on embedding dimensionality. Supports 2D and 3D.

        Args:
            embeddings: Embeddings for visualization
            show: Whether to display the figure immediately
            **kwargs: Additional parameters for Plotly scatter/scatter_3d
        """
        if embeddings.shape[1] == 3:
            fig = self._visualize_3d(embeddings, **kwargs)
        elif embeddings.shape[1] == 2:
            fig = self._visualize_2d(embeddings, **kwargs)
        else:
            raise ValueError("Embeddings must be either 2D or 3D for visualization.")
        
        if show:
            fig.show()

        return fig
    
    def visualize_multiple(
            self,
            embeddings: dict[str, np.ndarray], 
            n_cols: int = 3,
    ) -> go.Figure:
        """
        Visualize multiple embeddings in a single figure using subplots.

        Args:
            embeddings: Dictionary mapping names to embeddings for visualization
        """
        titles = list(embeddings.keys())
        n = len(embeddings)
        n_rows = (n + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols, 
            subplot_titles=titles,
        )
        for i, (name, emb) in enumerate(embeddings.items()):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            fig_i = self.visualize(
                emb,
                legendgroup=name,  # Group legends by dataset
                showlegend=(i == 0),  # Only show legend for first subplot
            )
            fig_i.name = name
            fig.add_trace(fig_i, row=row, col=col)

        return fig