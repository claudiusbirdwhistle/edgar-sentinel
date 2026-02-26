"""Tests for the FastAPI REST API module."""

from __future__ import annotations

import json
from datetime import date, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from edgar_sentinel.api.app import create_app
from edgar_sentinel.api.deps import AppState, JobStatus
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
    SimilarityResult,
    StorageBackend,
)


# -- Fixtures --


def _make_config(tmp_path, api_key=None):
    """Create a test config."""
    return SentinelConfig(
        edgar=EdgarConfig(user_agent="Test test@test.com"),
        storage=StorageConfig(
            backend=StorageBackend.SQLITE,
            sqlite_path=str(tmp_path / "test.db"),
        ),
        api=APIConfig(api_key=api_key),
    )


@pytest.fixture
def config(tmp_path):
    return _make_config(tmp_path)


@pytest.fixture
def app(config):
    return create_app(config=config)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def authed_config(tmp_path):
    return _make_config(tmp_path, api_key="test-secret-key")


@pytest.fixture
def authed_app(authed_config):
    return create_app(config=authed_config)


@pytest.fixture
def authed_client(authed_app):
    with TestClient(authed_app) as c:
        yield c


def _make_filing_metadata(ticker="AAPL", accession="0000320193-23-000106"):
    return FilingMetadata(
        cik="320193",
        ticker=ticker,
        company_name=f"{ticker} Inc.",
        form_type=FormType.FORM_10K,
        filed_date=date(2023, 11, 3),
        accession_number=accession,
        url="https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/idx.htm",
    )


def _make_filing(ticker="AAPL", accession="0000320193-23-000106"):
    meta = _make_filing_metadata(ticker, accession)
    text = "The Company designs manufactures and markets smartphones."
    section = FilingSection(
        filing_id=accession,
        section_name="mda",
        raw_text=text,
        word_count=len(text.split()),
        extracted_at=datetime(2024, 1, 15, 10, 30, 0),
    )
    return Filing(metadata=meta, sections={"mda": section})


def _make_composite(ticker="AAPL", score=0.72, signal_date=None, rank=1):
    return CompositeSignal(
        ticker=ticker,
        signal_date=signal_date or date(2023, 11, 5),
        composite_score=score,
        components={"dictionary_mda": 0.5, "similarity_mda": 0.5},
        rank=rank,
    )


def _make_sentiment(filing_id="0000320193-23-000106"):
    return SentimentResult(
        filing_id=filing_id,
        section_name="mda",
        analyzer_name="dictionary",
        sentiment_score=0.15,
        confidence=0.85,
        metadata={},
        analyzed_at=datetime(2024, 1, 15, 11, 0, 0),
    )


def _make_similarity(filing_id="0000320193-23-000106"):
    return SimilarityResult(
        filing_id=filing_id,
        prior_filing_id="0000320193-22-000108",
        section_name="mda",
        similarity_score=0.82,
        change_score=0.18,
        analyzed_at=datetime(2024, 1, 15, 11, 0, 0),
    )


# -- Health Endpoint --


@pytest.mark.unit
class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["storage_backend"] == "sqlite"
        assert "total_filings" in data
        assert "total_signals" in data

    def test_health_no_auth_required(self, authed_client):
        """Health endpoint should be accessible without API key."""
        resp = authed_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# -- Signals Endpoints --


@pytest.mark.unit
class TestSignalsEndpoints:
    def test_list_signals_empty(self, client):
        resp = client.get("/api/signals")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []
        assert data["offset"] == 0
        assert data["limit"] == 50

    def test_list_signals_with_data(self, app, client):
        """Insert composites via store, then query via API."""
        store = app.state.app_state.store

        async def seed():
            for i, ticker in enumerate(["AAPL", "MSFT", "GOOGL"]):
                c = _make_composite(ticker=ticker, score=0.5 + i * 0.1, rank=i + 1)
                await store.save_composite(c)

        import asyncio
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.get("/api/signals")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3
        # Default sort is composite_score desc
        scores = [item["composite_score"] for item in data["items"]]
        assert scores == sorted(scores, reverse=True)

    def test_list_signals_filter_ticker(self, app, client):
        store = app.state.app_state.store

        async def seed():
            await store.save_composite(_make_composite(ticker="AAPL", score=0.7))
            await store.save_composite(_make_composite(ticker="MSFT", score=0.5, signal_date=date(2023, 11, 6)))

        import asyncio
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.get("/api/signals?ticker=AAPL")
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["ticker"] == "AAPL"

    def test_list_signals_filter_score(self, app, client):
        store = app.state.app_state.store

        async def seed():
            await store.save_composite(_make_composite(ticker="AAPL", score=0.8))
            await store.save_composite(_make_composite(ticker="MSFT", score=0.3, signal_date=date(2023, 11, 6)))

        import asyncio
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.get("/api/signals?min_score=0.5")
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["ticker"] == "AAPL"

    def test_list_signals_pagination(self, app, client):
        store = app.state.app_state.store

        async def seed():
            for i in range(5):
                c = _make_composite(
                    ticker=f"T{i:03d}",
                    score=0.1 * (i + 1),
                    signal_date=date(2023, 11, i + 1),
                    rank=i + 1,
                )
                await store.save_composite(c)

        import asyncio
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.get("/api/signals?offset=2&limit=2")
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["offset"] == 2

    def test_list_signals_sort_by_date(self, app, client):
        store = app.state.app_state.store

        async def seed():
            await store.save_composite(_make_composite(ticker="AAPL", signal_date=date(2023, 11, 1)))
            await store.save_composite(_make_composite(ticker="MSFT", signal_date=date(2023, 12, 1)))

        import asyncio
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.get("/api/signals?sort_by=signal_date&sort_desc=true")
        data = resp.json()
        dates = [item["signal_date"] for item in data["items"]]
        assert dates == sorted(dates, reverse=True)

    def test_get_ticker_signals(self, app, client):
        store = app.state.app_state.store

        async def seed():
            await store.save_composite(_make_composite(ticker="AAPL", score=0.7))

        import asyncio
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.get("/api/signals/AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["ticker"] == "AAPL"

    def test_get_ticker_signals_case_insensitive(self, app, client):
        store = app.state.app_state.store

        async def seed():
            await store.save_composite(_make_composite(ticker="AAPL"))

        import asyncio
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.get("/api/signals/aapl")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_ticker_signals_404(self, client):
        resp = client.get("/api/signals/ZZZZ")
        assert resp.status_code == 404
        assert "ZZZZ" in resp.json()["detail"]


# -- Filings Endpoints --


@pytest.mark.unit
class TestFilingsEndpoints:
    def test_list_filings_empty(self, client):
        resp = client.get("/api/filings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_filings_with_data(self, app, client):
        store = app.state.app_state.store

        async def seed():
            await store.save_filing(_make_filing("AAPL"))

        import asyncio
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.get("/api/filings")
        data = resp.json()
        assert data["total"] == 1
        item = data["items"][0]
        assert item["ticker"] == "AAPL"
        assert item["form_type"] == "10-K"
        assert "mda" in item["sections"]
        assert "dictionary" in item["analysis_status"]

    def test_get_ticker_filings(self, app, client):
        store = app.state.app_state.store

        async def seed():
            await store.save_filing(_make_filing("AAPL"))

        import asyncio
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.get("/api/filings/AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["ticker"] == "AAPL"

    def test_get_ticker_filings_404(self, client):
        resp = client.get("/api/filings/ZZZZ")
        assert resp.status_code == 404

    def test_get_filing_detail(self, app, client):
        store = app.state.app_state.store

        async def seed():
            await store.save_filing(_make_filing("AAPL"))

        import asyncio
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.get("/api/filings/detail/0000320193-23-000106")
        assert resp.status_code == 200
        data = resp.json()
        assert data["accession_number"] == "0000320193-23-000106"
        assert "section_texts" in data
        assert "mda" in data["section_texts"]
        assert "analysis_results" in data

    def test_get_filing_detail_404(self, client):
        resp = client.get("/api/filings/detail/9999999999-99-999999")
        assert resp.status_code == 404

    def test_filing_detail_with_analysis(self, app, client):
        """Filing detail should include sentiment and similarity results."""
        store = app.state.app_state.store
        accession = "0000320193-23-000106"
        prior_accession = "0000320193-22-000108"

        async def seed():
            await store.save_filing(_make_filing("AAPL", accession))
            # Prior filing needed for FK on similarity result
            await store.save_filing(_make_filing("AAPL", prior_accession))
            await store.save_sentiment(_make_sentiment(accession))
            await store.save_similarity(_make_similarity(accession))

        import asyncio
        asyncio.get_event_loop().run_until_complete(seed())

        resp = client.get(f"/api/filings/detail/{accession}")
        data = resp.json()
        assert len(data["analysis_results"]) == 2
        analyzer_names = {r["analyzer_name"] for r in data["analysis_results"]}
        assert "dictionary" in analyzer_names
        assert "similarity" in analyzer_names


# -- Pipeline Endpoints --


@pytest.mark.unit
class TestPipelineEndpoints:
    def test_trigger_ingest(self, client):
        resp = client.post(
            "/api/pipeline/ingest",
            json={
                "tickers": ["AAPL", "MSFT"],
                "start_year": 2023,
                "end_year": 2024,
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "pending"
        assert data["job_id"].startswith("ingest-")
        assert "2 tickers" in data["message"]

    def test_trigger_analyze(self, client):
        resp = client.post(
            "/api/pipeline/analyze",
            json={"analyzers": ["dictionary"]},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"].startswith("analyze-")
        assert "dictionary" in data["message"]

    def test_trigger_backtest(self, client):
        resp = client.post(
            "/api/pipeline/backtest",
            json={
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"].startswith("backtest-")

    def test_ingest_validation_error(self, client):
        """Missing required fields should return 422."""
        resp = client.post("/api/pipeline/ingest", json={})
        assert resp.status_code == 422

    def test_ingest_empty_tickers(self, client):
        resp = client.post(
            "/api/pipeline/ingest",
            json={"tickers": [], "start_year": 2023, "end_year": 2024},
        )
        assert resp.status_code == 422

    def test_backtest_validation(self, client):
        """Invalid quantile count should return 422."""
        resp = client.post(
            "/api/pipeline/backtest",
            json={
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
                "num_quantiles": 1,  # below min of 2
            },
        )
        assert resp.status_code == 422


# -- Job Status --


@pytest.mark.unit
class TestJobStatus:
    def test_job_status_after_trigger(self, client):
        """After triggering a job, polling should return its status."""
        resp = client.post(
            "/api/pipeline/ingest",
            json={"tickers": ["AAPL"], "start_year": 2023, "end_year": 2024},
        )
        job_id = resp.json()["job_id"]

        status_resp = client.get(f"/api/jobs/{job_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("pending", "running", "completed", "failed")

    def test_job_not_found(self, client):
        resp = client.get("/api/jobs/nonexistent-job-id")
        assert resp.status_code == 404


# -- Authentication --


@pytest.mark.unit
class TestAuthentication:
    def test_api_key_required(self, authed_client):
        """Without API key, non-health endpoints should return 401."""
        resp = authed_client.get("/api/signals")
        assert resp.status_code == 401

    def test_api_key_valid(self, authed_client):
        """With correct API key, endpoint should work."""
        resp = authed_client.get(
            "/api/signals",
            headers={"X-API-Key": "test-secret-key"},
        )
        assert resp.status_code == 200

    def test_api_key_invalid(self, authed_client):
        """With wrong API key, should get 401."""
        resp = authed_client.get(
            "/api/signals",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_health_exempt_from_auth(self, authed_client):
        """Health endpoint should not require API key."""
        resp = authed_client.get("/api/health")
        assert resp.status_code == 200


# -- App Factory --


@pytest.mark.unit
class TestAppFactory:
    def test_create_app_returns_fastapi(self, config):
        app = create_app(config=config)
        assert app.title == "EDGAR Sentinel API"
        assert app.version == "0.1.0"

    def test_create_app_includes_routes(self, config):
        app = create_app(config=config)
        route_paths = [r.path for r in app.routes]
        assert "/api/health" in route_paths
        assert "/api/signals" in route_paths
        assert "/api/filings" in route_paths

    def test_create_app_cors_enabled(self, config):
        app = create_app(config=config)
        # CORS middleware is added via add_middleware
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes


# -- Schema Validation --


@pytest.mark.unit
class TestSchemaValidation:
    def test_signal_response_serialization(self):
        from edgar_sentinel.api.schemas import SignalResponse

        s = SignalResponse(
            ticker="AAPL",
            signal_date=date(2023, 11, 5),
            composite_score=0.72,
            rank=1,
            components={"dictionary": 0.5, "similarity": 0.5},
        )
        data = s.model_dump(mode="json")
        assert data["ticker"] == "AAPL"
        assert data["signal_date"] == "2023-11-05"
        assert data["composite_score"] == 0.72

    def test_health_response_defaults(self):
        from edgar_sentinel.api.schemas import HealthResponse

        h = HealthResponse(
            version="0.1.0",
            storage_backend="sqlite",
            total_filings=10,
            total_signals=5,
        )
        assert h.status == "ok"

    def test_ingest_request_validation(self):
        from edgar_sentinel.api.schemas import IngestRequest

        req = IngestRequest(
            tickers=["AAPL"],
            start_year=2023,
            end_year=2024,
        )
        assert req.form_types == ["10-K", "10-Q"]
        assert req.force is False

    def test_paginated_response(self):
        from edgar_sentinel.api.schemas import SignalListResponse, SignalResponse

        resp = SignalListResponse(
            total=100,
            offset=10,
            limit=20,
            items=[],
        )
        assert resp.total == 100

    def test_job_status_response(self):
        from edgar_sentinel.api.schemas import JobStatusResponse

        resp = JobStatusResponse(
            job_id="ingest-abc123",
            status="completed",
            created_at=datetime(2024, 1, 1),
            completed_at=datetime(2024, 1, 1, 0, 5, 0),
            result={"filings_ingested": 10},
        )
        assert resp.result["filings_ingested"] == 10


# -- Module Imports --


@pytest.mark.unit
class TestModuleImports:
    def test_api_package_importable(self):
        import edgar_sentinel.api
        assert hasattr(edgar_sentinel.api, "create_app")

    def test_schemas_importable(self):
        from edgar_sentinel.api.schemas import (
            HealthResponse,
            SignalListResponse,
            SignalResponse,
            FilingResponse,
            IngestRequest,
            JobResponse,
        )
        assert HealthResponse is not None

    def test_deps_importable(self):
        from edgar_sentinel.api.deps import (
            AppState,
            JobStatus,
            get_app_state,
            get_config,
            get_store,
        )
        assert AppState is not None
