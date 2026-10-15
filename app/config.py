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

    @property
    def database_path(self) -> Path:
        return self.data_dir / "context.db"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "extracted"

    def prepare(self) -> None:
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

