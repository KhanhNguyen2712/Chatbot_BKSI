"""RAG Chain using LangChain with OpenRouter."""

import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config import get_prompt, get_settings
from src.models import ChatResponse, SearchResult
from src.vectorstore import LanceDBVectorStore

from .memory import ConversationMemory
from .reranker import Reranker
from .retriever import Retriever


class RAGChain:
    """RAG Chain for question answering with retrieval."""

    def __init__(
        self,
        vector_store: LanceDBVectorStore,
        use_rerank: bool | None = None,
        use_memory: bool | None = None,
    ):
        self.settings = get_settings()
        self.vector_store = vector_store

        # Initialize components
        self.retriever = Retriever(vector_store, top_k=self.settings.rag_top_k)

        # Reranker
        self.use_rerank = (
            use_rerank if use_rerank is not None else self.settings.rag_rerank_enabled
        )
        self.reranker = Reranker() if self.use_rerank else None

        # Memory
        self.use_memory = (
            use_memory if use_memory is not None else self.settings.memory_enabled
        )
        self.memory = ConversationMemory() if self.use_memory else None

        # Initialize LLM with OpenRouter
        self.llm = ChatOpenAI(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            openai_api_key=self.settings.openrouter_api_key,
            openai_api_base=self.settings.openrouter_base_url,
        )

        # Load prompts
        self.system_prompt = get_prompt("system_prompt")
        self.rag_prompt_template = get_prompt("rag_prompt")

        logger.info(
            f"RAGChain initialized with model: {self.settings.llm_model}, "
            f"rerank: {self.use_rerank}, memory: {self.use_memory}"
        )

    def _retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Retrieve and optionally rerank results."""
        # Retrieve more if reranking
        retrieve_k = top_k * 3 if self.use_rerank else top_k
        results = self.retriever.retrieve(query, top_k=retrieve_k)

        # Rerank if enabled
        if self.use_rerank and self.reranker and results:
            results = self.reranker.rerank(query, results, top_n=top_k)

        return results

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results into context string."""
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            # Handle both SearchResult objects and dict
            if isinstance(result, dict):
                metadata = result.get("metadata", {})
                content = result.get("content", "")
            else:
                metadata = result.metadata
                content = result.content
            source = (
                metadata.get("source", "Unknown")
                if isinstance(metadata, dict)
                else "Unknown"
            )
            context_parts.append(f"[Nguồn {i}: {source}]\n{content}")

        return "\n\n---\n\n".join(context_parts)

    def _format_sources(self, results: list[SearchResult]) -> list[dict[str, Any]]:
        """Format sources for response."""
        sources = []
        for result in results:
            # Handle both SearchResult objects and dict
            if isinstance(result, dict):
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                score = result.get("score", 0)
                document_id = result.get("document_id", "")
            else:
                content = result.content
                metadata = result.metadata
                score = result.score
                document_id = result.document_id

            sources.append(
                {
                    "content": (
                        content[:200] + "..." if len(content) > 200 else content
                    ),
                    "source": (
                        metadata.get("source", "Unknown")
                        if isinstance(metadata, dict)
                        else "Unknown"
                    ),
                    "score": round(score, 4),
                    "document_id": document_id,
                }
            )
        return sources

    def chat(
        self,
        message: str,
        session_id: str | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
    ) -> ChatResponse:
        """
        Process a chat message and return response.

        Args:
            message: User message
            session_id: Session ID for conversation memory
            top_k: Number of documents to retrieve
            use_rerank: Whether to use reranking

        Returns:
            ChatResponse with answer and sources
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        k = top_k or self.settings.rag_top_k
        should_rerank = use_rerank if use_rerank is not None else self.use_rerank

        logger.info(f"Processing chat message for session {session_id}")

        # Retrieve relevant documents
        if should_rerank:
            results = self._retrieve_and_rerank(message, top_k=k)
        else:
            results = self.retriever.retrieve(message, top_k=k)

        # Format context
        context = self._format_context(results)

        # Build messages
        messages = [SystemMessage(content=self.system_prompt)]

        # Add conversation history if memory enabled
        if self.use_memory and self.memory:
            history = self.memory.get_langchain_messages(session_id)
            for role, content in history:
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(SystemMessage(content=f"Assistant: {content}"))

        # Format the RAG prompt
        if context:
            user_content = self.rag_prompt_template.format(
                context=context, question=message
            )
        else:
            user_content = (
                f"{message}\n\n(Không tìm thấy thông tin liên quan trong hệ thống)"
            )

        messages.append(HumanMessage(content=user_content))

        # Generate response
        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = f"Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu: {str(e)}"

        # Update memory
        if self.use_memory and self.memory:
            self.memory.add_message(session_id, "user", message)
            self.memory.add_message(session_id, "assistant", answer)

        # Format sources
        sources = self._format_sources(results)

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            cached=False,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool | None = None,
    ) -> list[SearchResult]:
        """
        Search for relevant documents without generating response.

        Args:
            query: Search query
            top_k: Number of results
            use_rerank: Whether to use reranking

        Returns:
            List of SearchResult
        """
        should_rerank = use_rerank if use_rerank is not None else self.use_rerank

        if should_rerank:
            return self._retrieve_and_rerank(query, top_k=top_k)
        return self.retriever.retrieve(query, top_k=top_k)

    def clear_memory(self, session_id: str | None = None) -> None:
        """Clear conversation memory."""
        if self.memory:
            if session_id:
                self.memory.clear_session(session_id)
            else:
                self.memory.clear_all()
