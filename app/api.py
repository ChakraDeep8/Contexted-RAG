from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from watchfiles import awatch

from .chat import ChatService, LLMError, OpenAICompatibleClient, REFUSAL
from .config import Settings
from .services import ApplicationServices
from .utils import SUPPORTED_EXTENSIONS

LOGGER = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


def create_app(settings: Settings | None = None, services: ApplicationServices | None = None) -> FastAPI:
    settings = settings or Settings()
    services = services or ApplicationServices(settings)

    async def monitor() -> None:
        try:
            await asyncio.to_thread(services.reconcile)
            async for _changes in awatch(settings.source_dir, debounce=500, step=100, recursive=True):
                await asyncio.to_thread(services.reconcile)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Background indexing stopped unexpectedly")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(monitor())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            services.close()

    app = FastAPI(title="Contexted RAG", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.services = services

    @app.get("/api/health")
    def health():
        return {"status": "ok", "llm_configured": bool(settings.llm_api_key)}

    @app.get("/api/sources")
    def sources():
        return [asdict(record) for record in services.repository.list_files()]

    @app.get("/api/sources/{file_id}/chunks")
    def source_chunks(file_id: int):
        record = next((item for item in services.repository.list_files() if item.id == file_id), None)
        if not record:
            raise HTTPException(404, "Source not found")
        return [asdict(chunk) for chunk in services.repository.chunks_for_file(record.path)]

    @app.get("/api/index/status")
    def index_status():
        records = services.repository.list_files()
        return {
            "total": len(records),
            "ready": sum(record.state.value == "ready" for record in records),
            "failed": sum(record.state.value == "failed" for record in records),
            "processing": sum(record.state.value in {"pending", "processing"} for record in records),
            "pending_embeddings": sum(1 for _ in services.repository.chunks_requiring_embeddings()),
        }

    @app.post("/api/index/reconcile")
    async def reconcile():
        changed, embedded = await asyncio.to_thread(services.reconcile)
        return {"changed_files": changed, "embedded_chunks": embedded}

    @app.post("/api/sources/upload")
    async def upload_sources(files: list[UploadFile] = File(...)):
        if not files:
            raise HTTPException(400, "Choose at least one document")
        prepared: list[tuple[UploadFile, str]] = []
        for upload in files:
            name = Path(upload.filename or "").name.strip()
            if not name or name in {".", ".."} or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise HTTPException(400, f"Unsupported file '{name or 'unnamed'}'. Allowed: {allowed}")
            prepared.append((upload, name))

        saved: list[str] = []
        limit = settings.max_upload_mb * 1024 * 1024
        for upload, original_name in prepared:
            destination = _available_destination(settings.source_dir, original_name)
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.uploading")
            size = 0
            try:
                with temporary.open("wb") as output:
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise HTTPException(
                                413, f"'{original_name}' exceeds the {settings.max_upload_mb} MB limit"
                            )
                        output.write(chunk)
                os.replace(temporary, destination)
                saved.append(destination.name)
            finally:
                await upload.close()
                temporary.unlink(missing_ok=True)

        changed, embedded = await asyncio.to_thread(services.reconcile)
        return {"files": saved, "changed_files": changed, "embedded_chunks": embedded}

    @app.post("/api/index/retry/{file_id}")
    async def retry(file_id: int):
        record = next((item for item in services.repository.list_files() if item.id == file_id), None)
        if not record:
            raise HTTPException(404, "Source not found")
        path = settings.source_dir / Path(record.path)
        if not path.exists():
            raise HTTPException(404, "Source file no longer exists")
        result = await asyncio.to_thread(services.ingestion.ingest_file, path)
        embedded = await asyncio.to_thread(services.retriever.index_pending)
        return {"result": asdict(result), "embedded_chunks": embedded}

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        if not services.repository.list_active_chunks():
            conversation_id = request.conversation_id or str(uuid.uuid4())
            services.repository.save_message(conversation_id, "user", request.question.strip())
            services.repository.save_message(conversation_id, "assistant", REFUSAL)
            return {"conversation_id": conversation_id, "answer": REFUSAL, "citations": [], "refused": True}
        try:
            service = ChatService(settings, services.retriever, OpenAICompatibleClient(settings))
            result = await service.ask(request.question.strip(), request.conversation_id)
            return asdict(result)
        except LLMError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest):
        result = await chat(request)

        async def events():
            yield f"event: meta\ndata: {json.dumps({'conversation_id': result['conversation_id'], 'refused': result['refused']})}\n\n"
            for word in result["answer"].split(" "):
                yield f"event: delta\ndata: {json.dumps({'text': word + ' '})}\n\n"
                await asyncio.sleep(0)
            yield f"event: citations\ndata: {json.dumps(result['citations'])}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")
    return app


app = create_app()


def _available_destination(directory: Path, filename: str) -> Path:
    destination = directory / filename
    if not destination.exists():
        return destination
    stem, suffix = destination.stem, destination.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1

