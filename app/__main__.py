from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import Settings
from .pipeline import IngestionPipeline

CONSOLE = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app", description="Context ingestion pipeline")
    parser.add_argument("--source-dir", type=Path, help="Override Context-fetcher directory")
    parser.add_argument("--data-dir", type=Path, help="Override data directory")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("ingest", help="Reconcile the source folder once")
    subcommands.add_parser("watch", help="Reconcile, then continuously monitor the folder")
    subcommands.add_parser("status", help="Show indexed and failed files")
    subcommands.add_parser("index", help="Create embeddings for pending chunks")
    search_parser = subcommands.add_parser("search", help="Test hybrid retrieval")
    search_parser.add_argument("query")
    subcommands.add_parser("serve", help="Run the local web application")
    inspect_parser = subcommands.add_parser("inspect", help="Show chunks and citations for a file")
    inspect_parser.add_argument("file", help="Relative path or unique filename")
    return parser


def make_settings(args: argparse.Namespace) -> Settings:
    overrides = {}
    if args.source_dir:
        overrides["source_dir"] = args.source_dir
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    return Settings(**overrides)


def show_status(pipeline: IngestionPipeline) -> None:
    table = Table(title="Context ingestion status")
    for column in ("Path", "State", "Format", "Chunks", "Indexed", "Error"):
        table.add_column(column)
    records = pipeline.repository.list_files()
    for record in records:
        table.add_row(
            record.path, record.state.value, record.format, str(record.chunk_count),
            record.indexed_at or "-", record.last_error or "-",
        )
    CONSOLE.print(table if records else "No files have been discovered yet.")


def inspect_file(pipeline: IngestionPipeline, query: str) -> int:
    records = pipeline.repository.list_files()
    normalized = Path(query).as_posix()
    matches = [record for record in records if record.path == normalized or Path(record.path).name == query]
    if not matches:
        CONSOLE.print(f"[red]No indexed file matches:[/red] {query}")
        return 1
    if len(matches) > 1:
        CONSOLE.print("[red]Filename is ambiguous. Use one of these relative paths:[/red]")
        for record in matches:
            CONSOLE.print(f"  {record.path}")
        return 1
    record = matches[0]
    chunks = pipeline.repository.chunks_for_file(record.path)
    CONSOLE.print(f"[bold]{record.path}[/bold] — {record.state.value}, {len(chunks)} chunks")
    if record.last_error:
        CONSOLE.print(f"[yellow]Last error:[/yellow] {record.last_error}")
    for chunk in chunks:
        loc = chunk.location
        citation = ", ".join(
            part for part in (
                f"page {loc.page}" if loc.page else "",
                f"sheet {loc.sheet}" if loc.sheet else "",
                f"rows {loc.row_start}-{loc.row_end}" if loc.row_start else "",
                f"cells {loc.cell_range}" if loc.cell_range else "",
                f"section {loc.section}" if loc.section else "",
            ) if part
        ) or "document level"
        CONSOLE.rule(f"Chunk {chunk.sequence} · {chunk.id[:12]} · {citation}")
        CONSOLE.print(chunk.text)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = make_settings(args)
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    pipeline = IngestionPipeline(settings)
    if args.command == "ingest":
        results = pipeline.reconcile()
        for result in results:
            color = "green" if result.state.value == "ready" else "red"
            CONSOLE.print(f"[{color}]{result.action:>11}[/{color}] {result.path} ({result.chunk_count} chunks)")
            if result.error:
                CONSOLE.print(f"             [red]{result.error}[/red]")
        if not results:
            CONSOLE.print("Everything is up to date.")
        if any(True for _ in pipeline.repository.chunks_requiring_embeddings(1)):
            from .services import ApplicationServices

            embedded = ApplicationServices(settings).retriever.index_pending()
            CONSOLE.print(f"Embedded {embedded} chunks for retrieval.")
        return 1 if any(result.state.value == "failed" for result in results) else 0
    if args.command == "index":
        from .services import ApplicationServices

        services = ApplicationServices(settings)
        count = services.retriever.index_pending()
        CONSOLE.print(f"Indexed {count} pending chunks.")
        return 0
    if args.command == "search":
        from .services import ApplicationServices

        services = ApplicationServices(settings)
        services.retriever.index_pending()
        hits = services.retriever.search(args.query, settings.retrieval_limit)
        for index, hit in enumerate(hits, 1):
            loc = hit.chunk.location
            CONSOLE.rule(f"{index}. {hit.file.path} · page {loc.page or '-'} · score {hit.score:.4f}")
            CONSOLE.print(hit.chunk.text)
        return 0
    if args.command == "serve":
        import uvicorn

        uvicorn.run("app.api:app", host=settings.host, port=settings.port, reload=False)
        return 0
    if args.command == "watch":
        try:
            asyncio.run(pipeline.watch())
        except KeyboardInterrupt:
            CONSOLE.print("\nWatcher stopped cleanly.")
        return 0
    if args.command == "status":
        show_status(pipeline)
        return 0
    if args.command == "inspect":
        return inspect_file(pipeline, args.file)
    return 2


if __name__ == "__main__":
    sys.exit(main())

