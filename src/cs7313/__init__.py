import logging
LOGGER = logging.getLogger(__name__)

from .embeddings import EmbeddingExtractor

__all__ = [
    "EmbeddingExtractor"
]