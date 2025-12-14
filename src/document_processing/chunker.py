"""Text chunking utilities."""

import hashlib
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import get_settings
from src.models import Chunk, Document


class TextChunker:
    """Split documents into chunks for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.rag_chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag_chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def _generate_chunk_id(
        self, document_id: str, chunk_index: int, content: str
    ) -> str:
        """Generate unique chunk ID."""
        hash_input = f"{document_id}_{chunk_index}_{content[:50]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def chunk_text(
        self,
        text: str,
        document_id: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Text content to split
            document_id: Parent document ID
            metadata: Additional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for document {document_id}")
            return []

        # Split text
        texts = self.splitter.split_text(text)

        chunks = []
        base_metadata = metadata or {}

        for idx, chunk_text in enumerate(texts):
            chunk_id = self._generate_chunk_id(document_id, idx, chunk_text)

            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                content=chunk_text,
                chunk_index=idx,
                metadata={
                    **base_metadata,
                    "chunk_size": len(chunk_text),
                    "total_chunks": len(texts),
                },
            )
            chunks.append(chunk)

        logger.debug(f"Split document {document_id} into {len(chunks)} chunks")
        return chunks

    def chunk_document(
        self, document: Document, content: str | None = None
    ) -> list[Chunk]:
        """
        Split a document into chunks.

        Args:
            document: Document object
            content: Optional content override (otherwise uses document metadata)

        Returns:
            List of Chunk objects
        """
        if content is None:
            content = document.metadata.get("content", "")

        if not content:
            logger.warning(f"No content found for document {document.id}")
            return []

        metadata = {
            "source": document.filename,
            "doc_type": document.doc_type.value,
            "source_path": document.source_path,
        }

        chunks = self.chunk_text(
            text=content,
            document_id=document.id,
            metadata=metadata,
        )

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of all Chunk objects
        """
        all_chunks = []

        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking document {doc.id}: {e}")

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
