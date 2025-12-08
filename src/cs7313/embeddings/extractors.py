import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
from typing import Optional, List, Any
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

from .preprocessing import PreprocessingStrategy, DefaultPreprocessing


class EmbeddingExtractor():
    """
    Main class for loading HuggingFace models and extracting embeddings.

    Supports multiple model types, provides preprocessing strategies, and
    enables embedding extraction and visualization with various clustering methods.

    Example:
        >>> extractor = EmbeddingExtractor("bert-base-uncased")
        >>> embeddings = extractor(data, text_column="text")
        >>> fig = extractor.visualize(embeddings)
        >>> fig.show()
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        preprocessing_strategy: Optional[PreprocessingStrategy] = None,
        **model_kwargs,
    ):
        """
        Initialize the embedding extractor with a HuggingFace model.

        Args:
            model_name: Name or path of the HuggingFace model
            device: Device to use ("cuda", "cpu", or None for auto-detection)
            preprocessing_strategy: Custom preprocessing strategy. If None, infers from model_name
            **model_kwargs: Additional keyword arguments for model loading
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        logger.info("Loading model and tokenizer...")
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Model and tokenizer loaded.")
        self.model.to(self.device)
        self.model.eval()

        # Set preprocessing strategy
        self.preprocessing_strategy = preprocessing_strategy
        if self.preprocessing_strategy is None:
            if "uncased" in model_name.lower():
                self.preprocessing_strategy = DefaultPreprocessing(cased=False)
            else:
                self.preprocessing_strategy = DefaultPreprocessing(cased=True)

        # Store extracted embeddings and metadata
        self._texts: Optional[List[str]] = None
        self._embeddings: Optional[np.ndarray] = None

    def _mean_pooling(
        self,
        model_output: Any,
        attention_mask: torch.Tensor
    ) -> np.ndarray:
        """
        Perform mean pooling on token embeddings.

        Args:
            model_output: Output from model (last_hidden_state)
            attention_mask: Attention mask tensor

        Returns:
            Pooled embeddings as numpy array
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).detach().cpu().numpy()

    def _extract_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        **tokenizer_kwargs
    ) -> np.ndarray:
        """
        Extract embeddings for texts with batching.

        Args:
            texts: List of texts to embed
            batch_size: Size of batches for processing
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        embeddings = []
        t_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            "max_length": 512,
            **tokenizer_kwargs
        }
        iterator = tqdm(range(0, len(texts), batch_size), disable=not show_progress)
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + batch_size]

                # Preprocess texts
                processed_texts = self.preprocessing_strategy(batch_texts)

                # Tokenize
                encoded_input = self.tokenizer(
                    processed_texts,
                    **t_kwargs
                )
                encoded_input = {
                    k: v.to(self.device) for k, v in encoded_input.items()
                }

                # Get embeddings
                model_output = self.model(**encoded_input)
                batch_embeddings = self._mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def __call__(
        self,
        data: np.ndarray,
        batch_size: int = 32,
        show_progress: bool = True,
        **tokenizer_args
    ) -> np.ndarray:
        """
        Extract embeddings from a dataframe column.

        Args:
            data: Input data as a numpy array
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings with shape (n_rows, embedding_dim)
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy ndarray.")
        
        texts = data.astype(str).tolist()
        self._texts = texts

        self._embeddings = self._extract_embeddings_batch(
            texts, 
            batch_size=batch_size,
            show_progress=show_progress,
            **tokenizer_args
        )

        return self._embeddings

    def get_embeddings(self) -> Optional[np.ndarray]:
        """Get last extracted embeddings."""
        return self._embeddings

    def get_texts(self) -> Optional[List[str]]:
        """Get texts corresponding to last extracted embeddings."""
        return self._texts