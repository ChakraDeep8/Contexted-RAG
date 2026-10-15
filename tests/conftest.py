from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        source_dir=tmp_path / "Context-fetcher",
        data_dir=tmp_path / "data",
        min_chunk_tokens=20,
        max_chunk_tokens=50,
        stable_check_seconds=0,
        stable_checks=1,
        retry_attempts=1,
        retry_base_seconds=0,
    )

