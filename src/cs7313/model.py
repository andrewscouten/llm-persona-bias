"""Module for NLP model loading and inference."""
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from . import LOGGER


class NLPModel(ABC):
    """
    Main class for loading HuggingFace models for NLP tasks.
    """

    _DEFAULT_TOKENIZER_ARGS = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        LOGGER.debug("Loading model and tokenizer...")
        self.model: AutoModel = AutoModel.from_pretrained(model_name, **(model_kwargs or {}))
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name, **(tokenizer_kwargs or {}))
        LOGGER.debug("Model and tokenizer loaded.")

        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def preprocess(self, texts: List[str]) -> Any:
        """
        Preprocess a list of texts for model input.

        Args:
            texts: List of input texts
        """

    @abstractmethod
    def postprocess(self, model_output: Any) -> Any:
        """Postprocess model outputs to desired format."""

    def __call__(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        **tokenizer_args: Any
    ) -> Any:
        """
        Process a list of texts and return model outputs.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Whether to show a progress bar
            **tokenizer_args: Additional arguments for the tokenizer

        Returns:
            Model outputs for the input texts
        """
        model_out = []

        token_args: Dict[str, Any] = {**self._DEFAULT_TOKENIZER_ARGS, **tokenizer_args}
        iterator = tqdm(range(0, len(texts), batch_size), disable=not show_progress)
        with torch.no_grad():
            for i in iterator:
                batch = texts[i: i + batch_size]
                batch = self.preprocess(batch)
                # Tokenize
                encoded_input = self.tokenizer(batch, **token_args)
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                # Pass through model
                out = self.model(**encoded_input)
                out = self.postprocess(out)
                model_out.append(out)

        return np.vstack(model_out)
