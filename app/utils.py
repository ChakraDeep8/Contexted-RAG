from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def normalize_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_chunk_id(file_hash: str, sequence: int, structural_key: str) -> str:
    value = f"{file_hash}:{sequence}:{structural_key}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def approximate_tokens(text: str) -> int:
    # Conservative tokenizer-free estimate used only for chunk sizing.
    return max(1, (len(text) + 3) // 4)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def wait_until_stable(path: Path, checks: int, delay: float) -> None:
    previous: tuple[int, int] | None = None
    stable = 0
    while stable < checks:
        stat = path.stat()
        current = (stat.st_size, stat.st_mtime_ns)
        stable = stable + 1 if current == previous else 0
        previous = current
        if stable < checks:
            time.sleep(delay)

