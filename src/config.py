"""Configuration management for Chatbot BKSI."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Groq API Configuration
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL"
    )
    
    # LLM Configuration
    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")
    llm_model: str = Field(default="openai/gpt-oss-120b", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")

    # Embedding Configuration
    embedding_model: str = Field(
        default="dangvantuan/vietnamese-document-embedding", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cuda", alias="EMBEDDING_DEVICE")

    # LanceDB Configuration
    lancedb_persist_dir: str = Field(
        default="./lancedb_data", alias="LANCEDB_PERSIST_DIR"
    )

    # RAG Configuration
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_chunk_size: int = Field(default=512, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=50, alias="RAG_CHUNK_OVERLAP")
    rag_rerank_enabled: bool = Field(default=True, alias="RAG_RERANK_ENABLED")
    rag_rerank_top_n: int = Field(default=3, alias="RAG_RERANK_TOP_N")

    # Cache Configuration
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_dir: str = Field(default="./.cache", alias="CACHE_DIR")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")

    # Memory Configuration
    memory_enabled: bool = Field(default=True, alias="MEMORY_ENABLED")
    memory_max_messages: int = Field(default=20, alias="MEMORY_MAX_MESSAGES")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # UI Configuration
    gradio_port: int = Field(default=7860, alias="GRADIO_SERVER_PORT")
    gradio_share: bool = Field(default=False, alias="GRADIO_SHARE")
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_SERVER_PORT")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", alias="LOG_FILE")


class ConfigManager:
    """Manages configuration from YAML files and environment variables."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._settings_cache: dict[str, Any] = {}
        self._prompts_cache: dict[str, str] = {}

    def load_settings(self) -> dict[str, Any]:
        """Load settings from YAML file."""
        if not self._settings_cache:
            settings_path = self.config_dir / "settings.yaml"
            if settings_path.exists():
                with open(settings_path, "r", encoding="utf-8") as f:
                    self._settings_cache = yaml.safe_load(f) or {}
        return self._settings_cache

    def load_prompts(self) -> dict[str, str]:
        """Load prompt templates from YAML file."""
        if not self._prompts_cache:
            prompts_path = self.config_dir / "prompts.yaml"
            if prompts_path.exists():
                with open(prompts_path, "r", encoding="utf-8") as f:
                    self._prompts_cache = yaml.safe_load(f) or {}
        return self._prompts_cache

    def get_prompt(self, key: str, **kwargs) -> str:
        """Get a prompt template and format it with provided variables."""
        prompts = self.load_prompts()
        template = prompts.get(key, "")
        if kwargs:
            return template.format(**kwargs)
        return template

    def get_setting(self, *keys: str, default: Any = None) -> Any:
        """Get a nested setting value using dot notation."""
        settings = self.load_settings()
        value = settings
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache()
def get_config_manager() -> ConfigManager:
    """Get cached config manager instance."""
    return ConfigManager()


# Convenience functions
def get_prompt(key: str, **kwargs) -> str:
    """Get a formatted prompt template."""
    return get_config_manager().get_prompt(key, **kwargs)


def get_yaml_setting(*keys: str, default: Any = None) -> Any:
    """Get a setting from YAML config."""
    return get_config_manager().get_setting(*keys, default=default)
