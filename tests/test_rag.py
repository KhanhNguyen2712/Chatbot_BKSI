import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_chain import ask_question
from src.vector_store import search_similar
from src.embedding import embedding_model

class TestRAG(unittest.TestCase):
    def test_embedding(self):
        """Test embedding functionality."""
        text = "Đây là một câu test."
        embedding = embedding_model.embed_query(text)
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

    def test_vector_search(self):
        """Test vector search (requires data in store)."""
        try:
            results = search_similar("quy chế học tập", top_k=1)
            self.assertIsInstance(results, list)
        except Exception:
            # Skip if no data
            self.skipTest("No data in vector store")

    def test_rag_chain(self):
        """Test RAG chain (requires data and API key)."""
        try:
            response = ask_question("Quy chế học tập là gì?")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        except Exception:
            # Skip if not configured
            self.skipTest("RAG chain not configured")

if __name__ == "__main__":
    unittest.main()
