from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from typing import List
from src.config import Config

class VietnameseEmbeddings(Embeddings):
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

# Global embedding instance
embedding_model = VietnameseEmbeddings()
