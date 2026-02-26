"""Dependency injection for FastAPI routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from fastapi import Request
from fastapi.responses import JSONResponse

from edgar_sentinel.core.config import SentinelConfig
from edgar_sentinel.ingestion.store import SqliteStore


@dataclass
class JobStatus:
    """Tracks a background pipeline job."""

    job_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    created_at: str  # ISO-8601
    completed_at: str | None = None
    result: dict | None = None
    error: str | None = None


@dataclass
class AppState:
    """Shared application state, attached to app.state during lifespan."""

    config: SentinelConfig
    store: SqliteStore
    jobs: dict[str, JobStatus] = field(default_factory=dict)


def get_app_state(request: Request) -> AppState:
    """Dependency: retrieve AppState from the request."""
    return request.app.state.app_state


def get_config(request: Request) -> SentinelConfig:
    """Dependency: retrieve config."""
    return request.app.state.app_state.config


def get_store(request: Request) -> SqliteStore:
    """Dependency: retrieve storage backend."""
    return request.app.state.app_state.store


EXEMPT_PATHS = {"/api/health"}


async def api_key_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Middleware: validate X-API-Key header when authentication is enabled."""
    if request.url.path in EXEMPT_PATHS:
        return await call_next(request)

    config = request.app.state.app_state.config
    if config.api.api_key:
        api_key = request.headers.get("X-API-Key")
        if api_key != config.api.api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Invalid or missing API key"},
            )
    return await call_next(request)
