"""Integration tests for the FastAPI REST API.

Uses FastAPI TestClient with real SQLite storage — no mocks.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient

from edgar_sentinel.api.app import create_app
from edgar_sentinel.core.config import (
    APIConfig,
    EdgarConfig,
    SentinelConfig,
    StorageConfig,
)
from edgar_sentinel.core.models import (
    CompositeSignal,
    Filing,
    FilingMetadata,
    FilingSection,
    FormType,
    SentimentResult,
    StorageBackend,
)


pytestmark = pytest.mark.integration


@pytest.fixture
def api_app(api_config):
    return create_app(config=api_config)


@pytest.fixture
def client(api_app):
    with TestClient(api_app) as c:
        yield c


class TestHealthEndpoint:
    """API health check."""

    def test_health_returns_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_version(self, client):
        response = client.get("/api/health")
        data = response.json()
        assert "version" in data


class TestSignalEndpoints:
    """API signal operations."""

    def test_list_signals_empty(self, client):
        response = client.get("/api/signals")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_signals_with_query_params(self, client):
        response = client.get(
            "/api/signals",
            params={"ticker": "AAPL", "start_date": "2023-01-01"},
        )
        assert response.status_code == 200

    def test_get_ticker_signals_not_found(self, client):
        response = client.get("/api/signals/NONEXISTENT")
        assert response.status_code == 404


class TestFilingEndpoints:
    """API filing operations."""

    def test_list_filings_empty(self, client):
        response = client.get("/api/filings")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []

    def test_get_ticker_filings_not_found(self, client):
        response = client.get("/api/filings/NONEXISTENT")
        assert response.status_code == 404

    def test_get_filing_detail_not_found(self, client):
        response = client.get("/api/filings/detail/0000000000-00-000000")
        assert response.status_code == 404


class TestPipelineEndpoints:
    """API pipeline trigger operations."""

    def test_trigger_ingest_returns_job(self, client):
        response = client.post(
            "/api/pipeline/ingest",
            json={
                "tickers": ["AAPL"],
                "form_types": ["10-K"],
                "start_year": 2023,
                "end_year": 2023,
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_trigger_analyze_returns_job(self, client):
        response = client.post(
            "/api/pipeline/analyze",
            json={"analyzers": ["dictionary"]},
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data

    def test_trigger_backtest_returns_job(self, client):
        response = client.post(
            "/api/pipeline/backtest",
            json={
                "start_date": "2020-01-01",
                "end_date": "2025-12-31",
            },
        )
        assert response.status_code == 202

    def test_get_job_not_found(self, client):
        response = client.get("/api/jobs/nonexistent-job-id")
        assert response.status_code == 404


class TestAPIAuthentication:
    """API key authentication when enabled."""

    def test_unauthenticated_when_key_required(self, tmp_path):
        config = SentinelConfig(
            edgar=EdgarConfig(user_agent="Test test@test.com"),
            storage=StorageConfig(
                backend=StorageBackend.SQLITE,
                sqlite_path=str(tmp_path / "auth-test.db"),
            ),
            api=APIConfig(api_key="test-secret-key"),
        )
        app = create_app(config=config)
        with TestClient(app) as client:
            response = client.get("/api/signals")
            assert response.status_code == 401

    def test_authenticated_with_valid_key(self, tmp_path):
        config = SentinelConfig(
            edgar=EdgarConfig(user_agent="Test test@test.com"),
            storage=StorageConfig(
                backend=StorageBackend.SQLITE,
                sqlite_path=str(tmp_path / "auth-test.db"),
            ),
            api=APIConfig(api_key="test-secret-key"),
        )
        app = create_app(config=config)
        with TestClient(app) as client:
            response = client.get(
                "/api/signals",
                headers={"X-API-Key": "test-secret-key"},
            )
            assert response.status_code == 200
