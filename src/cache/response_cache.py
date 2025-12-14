"""Response caching using LangChain SQLiteCache."""

import hashlib
from pathlib import Path
from typing import Any

from langchain_community.cache import SQLiteCache
from langchain_core.globals import get_llm_cache, set_llm_cache
from loguru import logger

from src.config import get_settings
from src.models import ChatResponse


class ResponseCache:
    """Cache for RAG responses using LangChain SQLiteCache.

    This uses LangChain's built-in caching mechanism which integrates
    seamlessly with the LangChain ecosystem.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        enabled: bool | None = None,
    ):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache database.
            enabled: Whether caching is enabled.
        """
        settings = get_settings()
        self.enabled = enabled if enabled is not None else settings.cache_enabled
        cache_path = cache_dir or settings.cache_dir

        self._custom_cache: dict[str, Any] = {}  # For non-LLM responses

        if self.enabled:
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            db_path = Path(cache_path) / "langchain_cache.db"

            # Set up LangChain's global LLM cache
            self._sqlite_cache = SQLiteCache(database_path=str(db_path))
            set_llm_cache(self._sqlite_cache)

            logger.info(f"LangChain SQLiteCache initialized at: {db_path}")
        else:
            self._sqlite_cache = None
            logger.info("Response cache disabled")

    def _generate_key(self, query: str, **kwargs) -> str:
        """Generate cache key from query and parameters."""
        key_parts = [query]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, query: str, **kwargs) -> ChatResponse | None:
        """
        Get cached response.

        Args:
            query: Query string
            **kwargs: Additional parameters for cache key

        Returns:
            ChatResponse if cached, None otherwise
        """
        if not self.enabled:
            return None

        key = self._generate_key(query, **kwargs)
        cached_data = self._custom_cache.get(key)

        if cached_data is not None:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            response = ChatResponse(**cached_data)
            response.cached = True
            return response

        return None

    def set(
        self,
        query: str,
        response: ChatResponse,
        **kwargs,
    ) -> None:
        """
        Cache a response.

        Args:
            query: Query string
            response: ChatResponse to cache
            **kwargs: Additional parameters for cache key
        """
        if not self.enabled:
            return

        key = self._generate_key(query, **kwargs)

        # Store response as dict
        cache_data = {
            "answer": response.answer,
            "sources": response.sources,
            "session_id": response.session_id,
            "cached": True,
        }

        self._custom_cache[key] = cache_data
        logger.debug(f"Cached response for query: {query[:50]}...")

    def invalidate(self, query: str, **kwargs) -> bool:
        """
        Invalidate a cached response.

        Args:
            query: Query string
            **kwargs: Additional parameters for cache key

        Returns:
            True if cache entry was deleted
        """
        if not self.enabled:
            return False

        key = self._generate_key(query, **kwargs)
        if key in self._custom_cache:
            del self._custom_cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached responses."""
        self._custom_cache.clear()

        # Clear LangChain cache if available
        if self._sqlite_cache is not None:
            try:
                self._sqlite_cache.clear()
            except Exception as e:
                logger.warning(f"Failed to clear LangChain cache: {e}")

        logger.info("Response cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "custom_cache_size": len(self._custom_cache),
            "langchain_cache": self._sqlite_cache is not None,
        }

    @property
    def llm_cache(self):
        """Get the LangChain LLM cache instance."""
        return get_llm_cache()
