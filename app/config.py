from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Multi-Agent Research Assistant"
    app_env: Literal["development", "test", "production"] = "development"
    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    database_path: Path = Path("data/research_assistant.db")
    max_upload_file_bytes: int = 20 * 1024 * 1024
    max_upload_total_bytes: int = 50 * 1024 * 1024

    chat_provider: Literal["dashscope", "openai", "fake"] = "dashscope"
    embedding_provider: Literal["dashscope", "openai", "fake"] = "dashscope"
    dashscope_api_key: SecretStr | None = Field(default=None, repr=False)
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_chat_model: str = "qwen3.7-flash"
    dashscope_embedding_model: str = "text-embedding-v4"
    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    provider_timeout_seconds: float = 60.0
    provider_max_retries: int = 2
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embedding_batch_size: int = 32
    retrieval_top_k: int = 6
    retrieval_candidate_k: int = 20
    retrieval_rrf_k: int = 60
    retrieval_min_similarity: float = 0.15
    max_query_expansions: int = 4
    max_revision_iterations: int = 2
    max_workflow_steps: int = 24
    worker_count: int = 2


def get_settings(env_file: Path | None = None) -> Settings:
    return Settings(_env_file=env_file if env_file is not None else ".env")
