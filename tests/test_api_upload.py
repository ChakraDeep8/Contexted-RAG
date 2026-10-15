from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.repository import SQLiteRepository


class FakeServices:
    def __init__(self, settings: Settings):
        self.repository = SQLiteRepository(settings.database_path)
        self.reconciliations = 0

    def reconcile(self) -> tuple[int, int]:
        self.reconciliations += 1
        return 1, 3

    def close(self) -> None:
        pass


def test_upload_saves_to_source_folder_and_avoids_overwrite(tmp_path: Path) -> None:
    settings = Settings(source_dir=tmp_path / "Context-fetcher", data_dir=tmp_path / "data")
    settings.prepare()
    services = FakeServices(settings)

    with TestClient(create_app(settings, services)) as client:
        first = client.post("/api/sources/upload", files={"files": ("report.pdf", b"first")})
        second = client.post("/api/sources/upload", files={"files": ("report.pdf", b"second")})

    assert first.status_code == 200
    assert first.json()["files"] == ["report.pdf"]
    assert first.json()["embedded_chunks"] == 3
    assert second.json()["files"] == ["report (2).pdf"]
    assert (settings.source_dir / "report.pdf").read_bytes() == b"first"
    assert (settings.source_dir / "report (2).pdf").read_bytes() == b"second"


def test_upload_rejects_unsupported_files(tmp_path: Path) -> None:
    settings = Settings(source_dir=tmp_path / "Context-fetcher", data_dir=tmp_path / "data")
    settings.prepare()

    with TestClient(create_app(settings, FakeServices(settings))) as client:
        response = client.post("/api/sources/upload", files={"files": ("notes.txt", b"no")})

    assert response.status_code == 400
    assert not (settings.source_dir / "notes.txt").exists()
