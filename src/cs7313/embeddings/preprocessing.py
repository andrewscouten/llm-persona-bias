from abc import ABC, abstractmethod
from typing import Union, List


class PreprocessingStrategy(ABC):
    """Abstract base class for model-specific preprocessing strategies."""

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """
        Preprocess text according to model requirements.

        Args:
            text: Single text to preprocess

        Returns:
            Preprocessed text
        """
        pass

    def __call__(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Apply preprocessing to single text or list of texts.
        
        Args:
            texts: Single text or list of texts to preprocess

        Returns:
            Preprocessed text or list of preprocessed texts
        """
        if isinstance(texts, list):
            return [self.preprocess(text) for text in texts]
        else:
            return self.preprocess(texts)


class DefaultPreprocessing(PreprocessingStrategy):
    """Default preprocessing strategy for generic models."""

    def __init__(self, cased : bool = False):
        """
        Initialize default preprocessing strategy.

        Args:
            cased: Whether the model contains cased or uncased text
        """
        self.cased = cased

    def preprocess(self, text: str) -> str:
        if not self.cased:
            text = text.lower()
        return text.strip()