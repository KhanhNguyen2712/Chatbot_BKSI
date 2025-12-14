"""Embedding model using sentence-transformers."""

from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import get_settings


class EmbeddingModel:
    """Embedding model wrapper using sentence-transformers."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.normalize = normalize
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            device = settings.embedding_device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer(
            self.model_name, 
            device=self.device,
            trust_remote_code=True,
        )
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def embed_texts(
        self,
        texts: Sequence[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query text.

        Args:
            query: Query text

        Returns:
            Numpy array of embedding with shape (embedding_dim,)
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embed documents (LangChain compatible interface).

        Args:
            documents: List of document texts

        Returns:
            List of embeddings as lists of floats
        """
        embeddings = self.embed_texts(documents, show_progress=True)
        return embeddings.tolist()

    def __call__(self, texts: str | list[str]) -> np.ndarray:
        """Make the model callable."""
        if isinstance(texts, str):
            return self.embed_query(texts)
        return self.embed_texts(texts)


@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    """Get cached embedding model instance."""
    return EmbeddingModel()


class LangChainEmbeddings:
    """LangChain-compatible embedding wrapper."""

    def __init__(self, model: EmbeddingModel | None = None):
        self.model = model or get_embedding_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents."""
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed query."""
        return self.model.embed_query(text).tolist()
