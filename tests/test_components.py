"""Tests for Chatbot BKSI."""

from pathlib import Path

import pytest


class TestDocumentParser:
    """Tests for DocumentParser."""

    def test_parse_markdown(self, tmp_path):
        """Test parsing markdown file."""
        from src.document_processing import DocumentParser

        # Create test markdown file
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\nThis is a test document.")

        parser = DocumentParser(output_dir=tmp_path / "output")
        doc = parser.parse(md_file)

        assert doc is not None
        assert doc.id is not None
        assert doc.filename == "test.md"
        assert "Test" in doc.metadata.get("content", "")

    def test_parse_text(self, tmp_path):
        """Test parsing text file."""
        from src.document_processing import DocumentParser

        # Create test text file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is plain text content.")

        parser = DocumentParser(output_dir=tmp_path / "output")
        doc = parser.parse(txt_file)

        assert doc is not None
        assert doc.filename == "test.txt"


class TestTextChunker:
    """Tests for TextChunker."""

    def test_chunk_document(self):
        """Test chunking a document."""
        from src.document_processing import TextChunker
        from src.models import Document, DocumentStatus, DocumentType

        chunker = TextChunker(chunk_size=100, chunk_overlap=20)

        doc = Document(
            id="test-doc",
            filename="test.md",
            doc_type=DocumentType.MARKDOWN,
            status=DocumentStatus.COMPLETED,
            source_path="/path/to/test.md",
            metadata={"content": "A" * 300},  # 300 characters
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        assert all(chunk.document_id == "test-doc" for chunk in chunks)
        assert all(len(chunk.content) <= 100 for chunk in chunks)

    def test_chunk_short_content(self):
        """Test chunking short content."""
        from src.document_processing import TextChunker
        from src.models import Document, DocumentStatus, DocumentType

        chunker = TextChunker(chunk_size=100, chunk_overlap=20)

        doc = Document(
            id="test-doc",
            filename="test.md",
            doc_type=DocumentType.MARKDOWN,
            status=DocumentStatus.COMPLETED,
            source_path="/path/to/test.md",
            metadata={"content": "Short content"},
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Short content"


class TestEmbeddingModel:
    """Tests for EmbeddingModel."""

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model for testing."""
        from src.embeddings import EmbeddingModel

        return EmbeddingModel(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Small model for tests
            device="cpu",
        )

    def test_embed_single(self, embedding_model):
        """Test embedding single text."""
        embedding = embedding_model.embed_query("Hello world")

        assert embedding is not None
        assert len(embedding) > 0

    def test_embed_batch(self, embedding_model):
        """Test embedding batch of texts."""
        texts = ["Hello", "World", "Test"]
        embeddings = embedding_model.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(e) == len(embeddings[0]) for e in embeddings)


class TestConversationMemory:
    """Tests for ConversationMemory."""

    def test_add_and_get_history(self):
        """Test adding and retrieving history."""
        from src.rag import ConversationMemory

        memory = ConversationMemory(max_messages=10)
        session_id = "test-session"

        memory.add_message(session_id, "user", "Hello")
        memory.add_message(session_id, "assistant", "Hi there!")

        history = memory.get_history(session_id)

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_max_history_limit(self):
        """Test history limit."""
        from src.rag import ConversationMemory

        memory = ConversationMemory(max_messages=2)
        session_id = "test-session"

        for i in range(5):
            memory.add_message(session_id, "user", f"Message {i}")

        history = memory.get_history(session_id)

        # Should only keep last 2 messages
        assert len(history) == 2

    def test_clear_history(self):
        """Test clearing history."""
        from src.rag import ConversationMemory

        memory = ConversationMemory()
        session_id = "test-session"

        memory.add_message(session_id, "user", "Hello")
        memory.clear(session_id)

        history = memory.get_history(session_id)
        assert len(history) == 0


class TestResponseCache:
    """Tests for ResponseCache (LangChain SQLiteCache)."""

    def test_cache_set_get(self, tmp_path):
        """Test cache set and get."""
        from src.cache import ResponseCache
        from src.models import ChatResponse

        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)

        query = "test query"
        response = ChatResponse(
            answer="Test answer",
            sources=[],
            session_id="test-session",
        )

        cache.set(query, response)
        result = cache.get(query)

        assert result is not None
        assert result.answer == "Test answer"
        assert result.cached is True

    def test_cache_disabled(self, tmp_path):
        """Test disabled cache."""
        from src.cache import ResponseCache
        from src.models import ChatResponse

        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=False)

        query = "test query"
        response = ChatResponse(
            answer="Test answer",
            sources=[],
            session_id="test-session",
        )

        cache.set(query, response)
        result = cache.get(query)

        assert result is None

    def test_cache_clear(self, tmp_path):
        """Test cache clearing."""
        from src.cache import ResponseCache
        from src.models import ChatResponse

        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=True)

        response = ChatResponse(answer="Test", sources=[], session_id="s1")
        cache.set("key1", response)
        cache.set("key2", response)
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestRetriever:
    """Tests for Retriever."""

    @pytest.fixture
    def mock_vector_store(self, mocker):
        """Create mock vector store."""
        mock = mocker.MagicMock()
        mock.search.return_value = [
            {
                "id": "chunk-1",
                "content": "Test content",
                "score": 0.9,
                "metadata": {"document_id": "doc-1"},
            }
        ]
        return mock

    def test_retrieve(self, mock_vector_store):
        """Test retrieval."""
        from src.rag import Retriever

        retriever = Retriever(vector_store=mock_vector_store, top_k=5)
        results = retriever.retrieve("test query")

        assert len(results) == 1
        assert results[0]["content"] == "Test content"
        mock_vector_store.search.assert_called_once()


class TestReranker:
    """Tests for Reranker (CrossEncoder)."""

    def test_rerank(self):
        """Test reranking with CrossEncoder."""
        from src.models import SearchResult
        from src.rag import Reranker

        # Use a smaller model for testing
        reranker = Reranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=2,
            device="cpu",
        )

        results = [
            SearchResult(
                content="Python programming language guide",
                score=0.5,
                document_id="doc-1",
                chunk_index=0,
            ),
            SearchResult(
                content="Java programming tutorial",
                score=0.6,
                document_id="doc-2",
                chunk_index=0,
            ),
            SearchResult(
                content="The weather is nice today",
                score=0.7,
                document_id="doc-3",
                chunk_index=0,
            ),
        ]

        reranked = reranker.rerank("Python tutorial", results)

        # Should return top_n results
        assert len(reranked) == 2
        # Scores should be updated
        assert all(hasattr(r, "score") for r in reranked)
        # First result should be more relevant to Python
        assert "Python" in reranked[0].content or "programming" in reranked[0].content
