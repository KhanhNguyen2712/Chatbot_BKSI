"""Chat API routes."""

from fastapi import APIRouter, Depends

from api.dependencies import get_rag_chain, get_response_cache
from src.cache import ResponseCache
from src.models import ChatRequest, ChatResponse, SearchRequest, SearchResult
from src.rag import RAGChain

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_chain: RAGChain = Depends(get_rag_chain),
    cache: ResponseCache = Depends(get_response_cache),
) -> ChatResponse:
    """
    Send a chat message and get a response.

    The response includes:
    - answer: The AI-generated response
    - sources: List of source documents used
    - session_id: Session identifier for conversation continuity
    - cached: Whether the response was from cache
    """
    # Check cache first
    cached_response = cache.get(
        request.message,
        top_k=request.top_k,
        use_rerank=request.use_rerank,
    )
    if cached_response:
        cached_response.session_id = request.session_id or cached_response.session_id
        return cached_response

    # Generate response
    response = rag_chain.chat(
        message=request.message,
        session_id=request.session_id,
        top_k=request.top_k,
        use_rerank=request.use_rerank,
    )

    # Cache the response
    cache.set(
        request.message,
        response,
        top_k=request.top_k,
        use_rerank=request.use_rerank,
    )

    return response


@router.post("/search", response_model=list[SearchResult])
async def search(
    request: SearchRequest,
    rag_chain: RAGChain = Depends(get_rag_chain),
) -> list[SearchResult]:
    """
    Search for relevant documents without generating a response.

    Returns a list of relevant document chunks with scores.
    """
    results = rag_chain.search(
        query=request.query,
        top_k=request.top_k,
        use_rerank=request.use_rerank,
    )
    return results


@router.post("/clear-memory")
async def clear_memory(
    session_id: str | None = None,
    rag_chain: RAGChain = Depends(get_rag_chain),
) -> dict[str, str]:
    """
    Clear conversation memory.

    If session_id is provided, clears only that session.
    Otherwise, clears all sessions.
    """
    rag_chain.clear_memory(session_id)
    return {
        "status": "success",
        "message": f"Memory cleared for {'session ' + session_id if session_id else 'all sessions'}",
    }


@router.post("/clear-cache")
async def clear_cache(
    cache: ResponseCache = Depends(get_response_cache),
) -> dict[str, str]:
    """Clear response cache."""
    cache.clear()
    return {"status": "success", "message": "Cache cleared"}
