from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="CONTEXT_", extra="ignore"
    )

    source_dir: Path = Path("Context-fetcher")
    data_dir: Path = Path("data")
    min_chunk_tokens: int = Field(default=350, ge=20)
    max_chunk_tokens: int = Field(default=600, ge=50)
    stable_check_seconds: float = Field(default=1.0, ge=0)
    stable_checks: int = Field(default=2, ge=1)
    retry_attempts: int = Field(default=3, ge=1)
    retry_base_seconds: float = Field(default=1.0, ge=0)
    log_level: str = "INFO"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-20b"
    retrieval_limit: int = Field(default=6, ge=1, le=20)
    dense_threshold: float = Field(default=0.35, ge=0, le=1)
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    max_upload_mb: int = Field(default=100, ge=1, le=2048)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "context.db"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    def prepare(self) -> None:
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

