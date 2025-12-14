"""FastAPI dependencies."""

from functools import lru_cache

from src.cache import ResponseCache
from src.config import get_settings
from src.embeddings import EmbeddingModel
from src.rag import RAGChain
from src.vectorstore import LanceDBVectorStore


@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    """Get singleton embedding model."""
    return EmbeddingModel()


@lru_cache()
def get_vector_store() -> LanceDBVectorStore:
    """Get singleton vector store."""
    return LanceDBVectorStore(embedding_model=get_embedding_model())


@lru_cache()
def get_rag_chain() -> RAGChain:
    """Get singleton RAG chain."""
    return RAGChain(vector_store=get_vector_store())


@lru_cache()
def get_response_cache() -> ResponseCache:
    """Get singleton response cache."""
    return ResponseCache()
