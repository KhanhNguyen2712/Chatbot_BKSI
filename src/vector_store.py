import chromadb
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from src.config import Config
from src.embedding import embedding_model

def create_vector_store():
    """Create or load ChromaDB vector store."""
    client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
    vector_store = Chroma(
        client=client,
        collection_name="university_docs",
        embedding_function=embedding_model
    )
    return vector_store

def add_documents_to_store(documents):
    """Add documents to vector store."""
    vector_store = create_vector_store()
    vector_store.add_documents(documents)
    print(f"Added {len(documents)} documents to vector store.")

def search_similar(query: str, top_k: int = Config.TOP_K):
    """Search for similar documents."""
    vector_store = create_vector_store()
    results = vector_store.similarity_search(query, k=top_k)
    return results
