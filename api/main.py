"""FastAPI main application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, documents
from src import __version__
from src.config import get_settings
from src.models import HealthResponse
from src.utils import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    setup_logging()
    yield
    # Shutdown (cleanup if needed)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Chatbot BKSI API",
        description="RAG-based Q&A API for BKSI documents",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat.router)
    app.include_router(documents.router)

    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint."""
        return {
            "name": "Chatbot BKSI API",
            "version": __version__,
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check():
        """Health check endpoint."""
        from api.dependencies import get_rag_chain, get_response_cache, get_vector_store

        components = {
            "vector_store": not get_vector_store().is_empty,
            "rag_chain": True,
            "cache": get_response_cache().enabled,
        }

        return HealthResponse(
            status="healthy" if all(components.values()) else "degraded",
            version=__version__,
            components=components,
        )

    return app


app = create_app()


def run_server():
    """Run the API server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )


if __name__ == "__main__":
    run_server()
