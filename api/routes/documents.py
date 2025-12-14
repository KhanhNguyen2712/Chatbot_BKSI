"""Document management API routes."""

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.config import get_settings
from src.document_processing import DocumentParser, TextChunker
from src.models import (
    Document,
    DocumentListResponse,
    DocumentStatus,
    DocumentUploadResponse,
)
from src.vectorstore import LanceDBVectorStore

router = APIRouter(prefix="/documents", tags=["documents"])

# Global instances (will be properly initialized via dependencies in production)
_parser = DocumentParser()
_chunker = TextChunker()


def get_vector_store() -> LanceDBVectorStore:
    """Get vector store instance."""
    from api.dependencies import get_vector_store as _get_vs

    return _get_vs()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    process: bool = True,
) -> DocumentUploadResponse:
    """
    Upload a document file.

    Supported formats: PDF, DOCX, DOC, PPTX, MD, TXT

    Args:
        file: The file to upload
        process: Whether to process and index the document immediately
    """
    settings = get_settings()

    # Validate file extension
    filename = file.filename or "unnamed"
    suffix = Path(filename).suffix.lower()
    supported = [".pdf", ".docx", ".doc", ".pptx", ".md", ".txt"]

    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {supported}",
        )

    # Generate document ID
    doc_id = str(uuid.uuid4())[:12]

    # Save to raw directory
    raw_dir = Path(settings.leann_index_path).parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_path = raw_dir / f"{doc_id}_{filename}"

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    status = DocumentStatus.PENDING
    message = "Document uploaded successfully"

    # Process if requested
    if process:
        try:
            # Parse document
            processed_dir = (
                Path(settings.leann_index_path).parent / "data" / "processed"
            )
            doc = _parser.parse_file(file_path, processed_dir)

            if doc.status == DocumentStatus.COMPLETED:
                # Chunk and index
                content = _parser.get_content(doc)
                chunks = _chunker.chunk_text(
                    content,
                    document_id=doc_id,
                    metadata={"source": filename},
                )

                # Add to vector store
                vector_store = get_vector_store()
                vector_store.add_chunks(chunks)

                status = DocumentStatus.COMPLETED
                message = f"Document processed and indexed ({len(chunks)} chunks)"
            else:
                status = doc.status
                message = (
                    f"Processing failed: {doc.metadata.get('error', 'Unknown error')}"
                )

        except Exception as e:
            status = DocumentStatus.FAILED
            message = f"Processing error: {str(e)}"

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=filename,
        status=status,
        message=message,
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """
    List all indexed documents.
    """
    vector_store = get_vector_store()
    stats = vector_store.get_stats()

    # Get unique documents from metadata
    documents = []
    seen_docs = set()

    for chunk_id, data in vector_store._metadata.items():
        doc_id = data.get("document_id", "unknown")
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            documents.append(
                Document(
                    id=doc_id,
                    filename=data.get("metadata", {}).get("source", "Unknown"),
                    doc_type=data.get("metadata", {}).get("doc_type", "txt"),
                    status=DocumentStatus.COMPLETED,
                    source_path=data.get("metadata", {}).get("source_path", ""),
                )
            )

    return DocumentListResponse(
        documents=documents,
        total=len(documents),
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> dict[str, Any]:
    """
    Delete a document and its chunks from the index.

    Args:
        document_id: The document ID to delete
    """
    vector_store = get_vector_store()

    deleted_count = vector_store.delete_document(document_id)

    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}",
        )

    return {
        "status": "success",
        "message": f"Deleted {deleted_count} chunks for document {document_id}",
        "deleted_chunks": deleted_count,
    }


@router.get("/stats")
async def get_stats() -> dict[str, Any]:
    """
    Get vector store statistics.
    """
    vector_store = get_vector_store()
    return vector_store.get_stats()


@router.post("/reindex")
async def reindex_documents() -> dict[str, Any]:
    """
    Reindex all documents in the data directories.

    This will:
    1. Parse all documents in data/raw/
    2. Read all markdown files in data/processed/
    3. Chunk all content
    4. Rebuild the vector index
    """
    settings = get_settings()
    base_dir = Path(settings.leann_index_path).parent

    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"

    all_chunks = []

    # Process raw documents
    if raw_dir.exists():
        docs = _parser.parse_directory(raw_dir, processed_dir)
        for doc in docs:
            if doc.status == DocumentStatus.COMPLETED:
                content = _parser.get_content(doc)
                chunks = _chunker.chunk_text(
                    content,
                    document_id=doc.id,
                    metadata={"source": doc.filename},
                )
                all_chunks.extend(chunks)

    # Process already-processed markdown files
    if processed_dir.exists():
        for md_file in processed_dir.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                doc_id = md_file.stem
                chunks = _chunker.chunk_text(
                    content,
                    document_id=doc_id,
                    metadata={"source": md_file.name},
                )
                all_chunks.extend(chunks)
            except Exception:
                pass

    # Rebuild index
    if all_chunks:
        vector_store = get_vector_store()
        vector_store.build_index(all_chunks)

    return {
        "status": "success",
        "message": f"Reindexed {len(all_chunks)} chunks",
        "total_chunks": len(all_chunks),
    }
