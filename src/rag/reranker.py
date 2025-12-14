"""Reranker for improving retrieval results using CrossEncoder."""

from loguru import logger

from src.config import get_settings
from src.models import SearchResult


class Reranker:
    """Rerank retrieved results using CrossEncoder (BGE-Reranker-v2-m3).

    This reranker uses BAAI/bge-reranker-v2-m3 which supports multilingual
    including Vietnamese, making it ideal for Vietnamese RAG applications.
    """

    def __init__(
        self,
        model_name: str | None = None,
        top_n: int | None = None,
        device: str | None = None,
    ):
        """Initialize the reranker.

        Args:
            model_name: CrossEncoder model name. Default: cross-encoder/ms-marco-MiniLM-L-6-v2
            top_n: Number of top results to return after reranking.
            device: Device to run model on ('cuda' or 'cpu').
        """
        import torch

        settings = get_settings()
        self.top_n = top_n or settings.rag_rerank_top_n
        # Use smaller model (~80MB) instead of BGE-reranker-v2-m3 (2.27GB)
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # Determine device - fallback to CPU if CUDA not available
        if device is None:
            device = settings.embedding_device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available for reranker, falling back to CPU")
            device = "cpu"
        self.device = device

        # Lazy load CrossEncoder
        self._model = None

    def _get_model(self):
        """Lazy load the CrossEncoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    max_length=512,
                    device=self.device,
                )
                logger.info(
                    f"Loaded CrossEncoder model: {self.model_name} on {self.device}"
                )
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder: {e}. Reranking disabled.")
                return None
        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """
        Rerank search results using CrossEncoder.

        Args:
            query: Original query
            results: List of SearchResult to rerank
            top_n: Number of top results to return

        Returns:
            Reranked list of SearchResult sorted by relevance
        """
        if not results:
            return []

        model = self._get_model()
        if model is None:
            # Return original results if model not available
            return results[: top_n or self.top_n]

        n = top_n or self.top_n

        try:
            # Prepare query-document pairs for CrossEncoder
            # Handle both SearchResult objects and dict
            pairs = []
            for r in results:
                if isinstance(r, dict):
                    content = r.get("content", "")
                else:
                    content = r.content
                pairs.append((query, content))

            # Get relevance scores
            scores = model.predict(pairs, show_progress_bar=False)

            # Create list of (index, score) and sort by score descending
            scored_results = list(enumerate(scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Map back to results with updated scores
            reranked_results = []
            for idx, rerank_score in scored_results[:n]:
                original_result = results[idx]

                # Handle both dict and SearchResult
                if isinstance(original_result, dict):
                    reranked_result = {
                        **original_result,
                        "score": float(rerank_score),
                        "metadata": {
                            **original_result.get("metadata", {}),
                            "original_score": original_result.get("score", 0),
                            "rerank_score": float(rerank_score),
                        },
                    }
                else:
                    reranked_result = SearchResult(
                        content=original_result.content,
                        score=float(rerank_score),
                        document_id=original_result.document_id,
                        chunk_index=original_result.chunk_index,
                        metadata={
                            **original_result.metadata,
                            "original_score": original_result.score,
                            "rerank_score": float(rerank_score),
                        },
                    )
                reranked_results.append(reranked_result)

            logger.debug(
                f"Reranked {len(results)} results to top {len(reranked_results)}"
            )
            return reranked_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:n]

    def predict_score(self, query: str, document: str) -> float:
        """
        Get relevance score for a single query-document pair.

        Args:
            query: Query string
            document: Document content

        Returns:
            Relevance score (higher = more relevant)
        """
        model = self._get_model()
        if model is None:
            return 0.0

        try:
            score = model.predict([(query, document)], show_progress_bar=False)[0]
            return float(score)
        except Exception as e:
            logger.error(f"Score prediction failed: {e}")
            return 0.0
