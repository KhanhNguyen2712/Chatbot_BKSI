from langchain_openai import ChatOpenAI
from src.config import Config

def create_llm():
    """Create LLM instance using OpenRouter."""
    llm = ChatOpenAI(
        model=Config.MODEL_NAME,
        openai_api_key=Config.OPENROUTER_API_KEY,
        openai_api_base=Config.OPENROUTER_BASE_URL,
        temperature=0.7,
        max_tokens=1000
    )
    return llm
