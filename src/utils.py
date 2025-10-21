import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration."""
    return logger

def validate_env_vars():
    """Validate required environment variables."""
    import os
    from src.config import Config

    required_vars = ["OPENROUTER_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

    logger.info("Environment variables validated successfully.")
