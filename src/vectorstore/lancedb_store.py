"""LanceDB vector store implementation."""

import json
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from loguru import logger

from src.embeddings import EmbeddingModel
from src.models import Chunk


class LanceDBVectorStore:
    """Vector store using LanceDB."""

    def __init__(
        self,
        persist_dir: str | Path = "lancedb_data",
        embedding_model: EmbeddingModel | None = None,
        table_name: str = "chunks",
    ):
        """Initialize LanceDB vector store.

        Args:
            persist_dir: Directory to persist the database.
            embedding_model: Embedding model for vectorization.
            table_name: Name of the table to store chunks.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.table_name = table_name

        # Connect to LanceDB
        self.db = lancedb.connect(str(self.persist_dir))
        self._table = None

        # Try to open existing table
        if self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)
            logger.info(
                f"Opened existing table '{self.table_name}' with {self._table.count_rows()} rows"
            )

    @property
    def is_empty(self) -> bool:
        """Check if the vector store is empty."""
        if self._table is None:
            return True
        return self._table.count_rows() == 0

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension from model."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not set")
        # Get dimension by embedding a sample text
        sample_embedding = self.embedding_model.embed_query("test")
        return len(sample_embedding)

    def build_index(self, chunks: list[Chunk], force_rebuild: bool = False) -> None:
        """Build vector index from chunks.

        Args:
            chunks: List of chunks to index.
            force_rebuild: If True, drop existing table and rebuild.
        """
        if not chunks:
            logger.warning("No chunks to index")
            return

        if self.embedding_model is None:
            raise ValueError("Embedding model required for building index")

        # Drop existing table if force rebuild
        if force_rebuild and self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
            self._table = None
            logger.info(f"Dropped existing table '{self.table_name}'")

        # Prepare data
        logger.info(f"Building index for {len(chunks)} chunks...")

        # Get embeddings in batches
        contents = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(contents)

        # Create records
        records = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append(
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata": json.dumps(chunk.metadata),
                    "vector": embedding,
                }
            )

        # Create or add to table
        if self._table is None:
            self._table = self.db.create_table(self.table_name, records)
            logger.info(f"Created table '{self.table_name}' with {len(records)} rows")
        else:
            self._table.add(records)
            logger.info(f"Added {len(records)} rows to table '{self.table_name}'")

        # Create vector index for faster search
        try:
            self._table.create_index(
                metric="cosine",
                num_partitions=min(256, len(chunks) // 10 + 1),
                num_sub_vectors=min(96, self._get_embedding_dim() // 8),
            )
            logger.info("Created vector index")
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to existing index.

        Args:
            chunks: List of chunks to add.
        """
        if self._table is None:
            self.build_index(chunks)
            return

        if self.embedding_model is None:
            raise ValueError("Embedding model required for adding chunks")

        contents = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(contents)

        records = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append(
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata": json.dumps(chunk.metadata),
                    "vector": embedding,
                }
            )

        self._table.add(records)
        logger.info(f"Added {len(records)} chunks to index")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks.

        Args:
            query: Search query.
            top_k: Number of results to return.
            filter_expr: Optional SQL filter expression.

        Returns:
            List of search results with content, score, and metadata.
        """
        if self._table is None or self.is_empty:
            logger.warning("Vector store is empty")
            return []

        if self.embedding_model is None:
            raise ValueError("Embedding model required for search")

        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Build search query
        search_query = self._table.search(query_embedding).limit(top_k)

        if filter_expr:
            search_query = search_query.where(filter_expr)

        # Execute search
        results = search_query.to_list()

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "id": result.get("id"),
                    "content": result.get("content"),
                    "score": 1
                    - result.get("_distance", 0),  # Convert distance to similarity
                    "document_id": result.get("document_id"),
                    "chunk_index": result.get("chunk_index"),
                    "metadata": json.loads(result.get("metadata", "{}")),
                }
            )

        return formatted_results

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: ID of document to delete.

        Returns:
            Number of chunks deleted.
        """
        if self._table is None:
            return 0

        # Count before delete
        count_before = self._table.count_rows()

        # Delete chunks with matching document_id
        self._table.delete(f"document_id = '{document_id}'")

        # Count after delete
        count_after = self._table.count_rows()
        deleted = count_before - count_after

        logger.info(f"Deleted {deleted} chunks for document '{document_id}'")
        return deleted

    def get_all_document_ids(self) -> list[str]:
        """Get all unique document IDs in the store.

        Returns:
            List of document IDs.
        """
        if self._table is None:
            return []

        # Query unique document IDs
        df = self._table.to_pandas()
        return df["document_id"].unique().tolist()

    def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics.

        Returns:
            Dictionary with stats.
        """
        if self._table is None:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "table_name": self.table_name,
                "persist_dir": str(self.persist_dir),
            }

        df = self._table.to_pandas()
        return {
            "total_chunks": len(df),
            "total_documents": df["document_id"].nunique(),
            "table_name": self.table_name,
            "persist_dir": str(self.persist_dir),
        }

    def clear(self) -> None:
        """Clear all data from the vector store."""
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
            self._table = None
            logger.info(f"Cleared table '{self.table_name}'")
