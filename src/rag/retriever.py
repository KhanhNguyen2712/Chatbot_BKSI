"""Retriever for RAG pipeline."""

from loguru import logger

from src.models import SearchResult
from src.vectorstore import LanceDBVectorStore


class Retriever:
    """Retriever component for RAG pipeline."""

    def __init__(
        self,
        vector_store: LanceDBVectorStore,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            top_k: Number of results (overrides default)

        Returns:
            List of SearchResult objects
        """
        k = top_k or self.top_k

        logger.debug(f"Retrieving top {k} results for query: {query[:50]}...")

        results = self.vector_store.search(query, top_k=k)

        logger.debug(f"Retrieved {len(results)} results")
        return results

    def get_context(
        self,
        query: str,
        top_k: int | None = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Get formatted context string for RAG.

        Args:
            query: Query text
            top_k: Number of results
            separator: Separator between chunks

        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k)

        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get("source", "Unknown")
            context_parts.append(f"[Nguồn {i}: {source}]\n{result.content}")

        return separator.join(context_parts)
