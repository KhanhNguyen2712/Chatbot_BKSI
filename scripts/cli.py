"""CLI for Chatbot BKSI."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src import __version__
from src.config import get_settings
from src.utils import setup_logging

app = typer.Typer(
    name="bksi",
    help="Chatbot BKSI - Trợ lý sinh viên BKSI",
    add_completion=False,
)
console = Console()


@app.callback()
def callback():
    """Chatbot BKSI CLI."""
    setup_logging()


@app.command()
def version():
    """Show version information."""
    console.print(
        f"[bold green]Chatbot BKSI[/bold green] version [cyan]{__version__}[/cyan]"
    )


@app.command()
def ingest(
    data_dir: Path = typer.Option(
        Path("data/raw"),
        "--data-dir",
        "-d",
        help="Directory containing documents to ingest",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        "-r",
        help="Rebuild vector index from scratch",
    ),
):
    """Ingest documents into vector store."""
    from src.document_processing import DocumentParser, TextChunker
    from src.embeddings import EmbeddingModel
    from src.vectorstore import LanceDBVectorStore

    settings = get_settings()

    if not data_dir.exists():
        console.print(f"[red]Error:[/red] Directory {data_dir} does not exist")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Initialize components
        progress.add_task("Initializing components...", total=None)

        embedding_model = EmbeddingModel(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )

        vector_store = LanceDBVectorStore(
            persist_dir=settings.lancedb_persist_dir,
            embedding_model=embedding_model,
        )

        parser = DocumentParser()
        chunker = TextChunker(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

        # Find documents
        extensions = ["*.pdf", "*.docx", "*.doc", "*.pptx", "*.md", "*.txt"]
        files = []
        for ext in extensions:
            files.extend(data_dir.glob(f"**/{ext}"))

        if not files:
            console.print(f"[yellow]No documents found in {data_dir}[/yellow]")
            raise typer.Exit(0)

        console.print(f"[green]Found {len(files)} documents[/green]")

        # Create output directory for processed files
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Process documents
        all_chunks = []
        for file in files:
            task = progress.add_task(f"Processing {file.name}...", total=None)

            try:
                # Parse document and save markdown to data/processed
                doc = parser.parse_file(file, output_dir=processed_dir)

                # Chunk content
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)

                console.print(f"  [OK] {file.name}: {len(chunks)} chunks")
            except Exception as e:
                console.print(f"  [ERROR] {file.name}: {e}[/red]")

            progress.remove_task(task)

        # Build index
        if all_chunks:
            task = progress.add_task("Building vector index...", total=None)

            if rebuild:
                vector_store.build_index(all_chunks, force_rebuild=True)
            else:
                vector_store.add_chunks(all_chunks)

            progress.remove_task(task)
            console.print(
                f"\n[bold green][OK] Indexed {len(all_chunks)} chunks[/bold green]"
            )
        else:
            console.print("[yellow]No chunks to index[/yellow]")


@app.command()
def api(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(
        True, "--reload/--no-reload", help="Enable auto-reload"
    ),
):
    """Run FastAPI server."""
    import uvicorn

    console.print(
        f"[bold green]Starting API server at http://{host}:{port}[/bold green]"
    )
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def gradio(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(7860, "--port", "-p", help="Port to bind"),
    share: bool = typer.Option(False, "--share/--no-share", help="Create public link"),
):
    """Run Gradio UI."""
    console.print(
        f"[bold green]Starting Gradio UI at http://{host}:{port}[/bold green]"
    )

    from ui.gradio_app import demo

    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
    )


@app.command()
def streamlit():
    """Run Streamlit UI."""
    import subprocess
    import sys

    console.print("[bold green]Starting Streamlit UI...[/bold green]")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/streamlit_app.py",
            "--server.headless",
            "true",
        ]
    )


@app.command()
def chat(
    session_id: Optional[str] = typer.Option(
        None, "--session", "-s", help="Session ID"
    ),
):
    """Interactive chat in terminal."""
    from src.embeddings import EmbeddingModel
    from src.rag import ConversationMemory, RAGChain, Reranker, Retriever
    from src.vectorstore import LanceDBVectorStore

    settings = get_settings()

    console.print("[bold cyan]Chatbot BKSI - Interactive Chat[/bold cyan]")
    console.print("Type 'quit' or 'exit' to end the session\n")

    # Initialize components
    with console.status("Initializing..."):
        embedding_model = EmbeddingModel(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )

        vector_store = LanceDBVectorStore(
            persist_dir=settings.lancedb_persist_dir,
            embedding_model=embedding_model,
        )

        rag_chain = RAGChain(
            vector_store=vector_store,
            use_rerank=settings.rag_rerank_enabled,
            use_memory=settings.memory_enabled,
        )

    # Generate session ID if not provided
    if not session_id:
        import uuid

        session_id = str(uuid.uuid4())[:8]

    console.print(f"Session: [cyan]{session_id}[/cyan]\n")

    # Chat loop
    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ")

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("\n[yellow]Goodbye![/yellow]")
                break

            if not user_input.strip():
                continue

            # Get response
            with console.status("Thinking..."):
                result = rag_chain.query(
                    question=user_input,
                    session_id=session_id,
                )

            # Display response
            console.print(
                f"\n[bold blue]Assistant:[/bold blue] {result.get('answer', 'No response')}\n"
            )

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
            break


@app.command()
def clear_cache():
    """Clear response cache."""
    from src.cache import ResponseCache

    settings = get_settings()
    cache = ResponseCache(
        cache_dir=Path(settings.cache_dir),
        enabled=settings.cache_enabled,
    )
    cache.clear()
    console.print("[green]Cache cleared successfully[/green]")


if __name__ == "__main__":
    app()
