"""Pydantic models for Chatbot BKSI."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    PPTX = "pptx"
    MARKDOWN = "md"
    TEXT = "txt"


class Document(BaseModel):
    """Document metadata model."""

    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    doc_type: DocumentType = Field(..., description="Document type")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    source_path: str = Field(..., description="Path to source file")
    processed_path: str | None = Field(None, description="Path to processed markdown")
    chunk_count: int = Field(default=0, description="Number of chunks")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Text chunk model."""

    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Position in document")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="Message role: user/assistant/system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Chat request model for API."""

    message: str = Field(..., description="User message", min_length=1)
    session_id: str | None = Field(
        None, description="Session ID for conversation memory"
    )
    top_k: int = Field(
        default=5, description="Number of documents to retrieve", ge=1, le=20
    )
    use_rerank: bool = Field(default=True, description="Whether to use reranking")


class ChatResponse(BaseModel):
    """Chat response model for API."""

    answer: str = Field(..., description="Assistant response")
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="Source documents"
    )
    session_id: str = Field(..., description="Session ID")
    cached: bool = Field(default=False, description="Whether response was cached")


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    use_rerank: bool = Field(default=True)


class SearchResult(BaseModel):
    """Search result model."""

    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Similarity score")
    document_id: str = Field(..., description="Source document ID")
    chunk_index: int = Field(..., description="Chunk position")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""

    document_id: str = Field(..., description="Assigned document ID")
    filename: str = Field(..., description="Original filename")
    status: DocumentStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")


class DocumentListResponse(BaseModel):
    """Response for listing documents."""

    documents: list[Document] = Field(default_factory=list)
    total: int = Field(default=0)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    version: str
    components: dict[str, bool] = Field(default_factory=dict)
