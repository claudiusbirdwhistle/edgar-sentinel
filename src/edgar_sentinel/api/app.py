"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from edgar_sentinel.api.deps import AppState, api_key_middleware
from edgar_sentinel.api.routes import router
from edgar_sentinel.core.config import SentinelConfig, load_config
from edgar_sentinel.core.exceptions import ConfigError, EdgarSentinelError, StorageError
from edgar_sentinel.ingestion.store import create_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    config = app.state._pending_config or load_config()
    store = await create_store(config.storage)

    app.state.app_state = AppState(config=config, store=store, jobs={})

    yield

    await store.close()


def create_app(config: SentinelConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    import edgar_sentinel

    app = FastAPI(
        title="EDGAR Sentinel API",
        description="SEC filing sentiment signal generator",
        version=edgar_sentinel.__version__,
        lifespan=lifespan,
    )

    # Stash config so lifespan can retrieve it
    app.state._pending_config = config

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional API key middleware
    if config and config.api.api_key:
        app.middleware("http")(api_key_middleware)

    app.include_router(router, prefix="/api")

    # Exception handlers
    @app.exception_handler(EdgarSentinelError)
    async def sentinel_exception_handler(request: Request, exc: EdgarSentinelError):
        status_map = {
            ConfigError: 400,
            StorageError: 500,
        }
        status = status_map.get(type(exc), 500)
        return JSONResponse(
            status_code=status,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    return app
