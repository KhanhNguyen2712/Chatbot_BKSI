import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # OpenRouter API
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    MODEL_NAME = "deepseek/deepseek-chat-v3.1:free"  # Free model

    # Embedding model
    EMBEDDING_MODEL_NAME = "dangvantuan/vietnamese-document-embedding"  # Free Vietnamese embedding

    # ChromaDB
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

    # Data paths
    DATA_RAW_PATH = "./data/raw"
    DATA_PROCESSED_PATH = "./data/processed"

    # Chunk settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Retrieval settings
    TOP_K = 5
