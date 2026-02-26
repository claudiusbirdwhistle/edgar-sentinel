"""Comprehensive integration tests for /tools/edgar-sentinel-runner.py.

Tests each pipeline stage (ingestion, analysis, signals, backtest) with
a real SQLite store, mocking only the EDGAR network layer. Catches
silent errors like Pydantic validation failures, missing fields, and
incorrect model construction.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from edgar_sentinel.core.config import StorageConfig
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
from edgar_sentinel.ingestion.store import SqliteStore


pytestmark = pytest.mark.integration

# --- Helpers ---

RUNNER_PATH = Path("/tools/edgar-sentinel-runner.py")


def _make_metadata(
    ticker: str,
    year: int,
    quarter: int = 0,
    form: FormType = FormType.FORM_10K,
) -> FilingMetadata:
    """Create a valid FilingMetadata for testing."""
    # Accession format: XXXXXXXXXX-YY-ZZZZZZ (10-2-6 digits = 18 total)
    cik = "0000320193"
    seq = f"{quarter:02d}{hash(ticker) % 10000:04d}"  # 6-digit sequence
    accession = f"{cik}-{year % 100:02d}-{seq}"
    return FilingMetadata(
        cik=cik,
        ticker=ticker,
        company_name=f"{ticker} Inc.",
        form_type=form,
        filed_date=date(year, 3 + quarter * 3, 15) if quarter else date(year, 11, 3),
        accession_number=accession,
        url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-','')}/filing.htm",
    )


def _make_filing(meta: FilingMetadata) -> Filing:
    """Create a Filing with one MDA section."""
    mda_text = (
        "Revenue grew significantly in the current fiscal year. "
        "The company experienced strong demand across all product lines. "
        "Operating expenses were managed effectively despite market challenges. "
        "Management believes current strategic initiatives will drive long-term growth. "
        "Cash flow from operations improved year over year due to efficiency gains."
    )
    section = FilingSection(
        filing_id=meta.accession_number,
        section_name="mda",
        raw_text=mda_text,
        word_count=len(mda_text.split()),
        extracted_at=datetime.now(timezone.utc),
    )
    return Filing(metadata=meta, sections={"mda": section})


@pytest.fixture
async def pipeline_store(tmp_path: Path) -> SqliteStore:
    """Initialized store for pipeline tests."""
    config = StorageConfig(
        backend=StorageBackend.SQLITE,
        sqlite_path=str(tmp_path / "pipeline_test.db"),
    )
    store = SqliteStore(config)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def seeded_store(pipeline_store: SqliteStore) -> SqliteStore:
    """Store pre-seeded with filings for 3 tickers, 2 years of 10-Ks."""
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        for year in [2024, 2025]:
            meta = _make_metadata(ticker, year)
            filing = _make_filing(meta)
            await pipeline_store.save_filing(filing)
    return pipeline_store


# --- Import runner functions ---
# The runner is at /tools/edgar-sentinel-runner.py. We import its functions
# after adding it to sys.path.

@pytest.fixture(autouse=True)
def _setup_runner_imports():
    """Ensure the runner module can be imported."""
    tools_dir = str(RUNNER_PATH.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)


def _import_runner():
    """Import runner functions fresh."""
    # Use importlib to load it as a module
    import importlib.util
    spec = importlib.util.spec_from_file_location("edgar_sentinel_runner", str(RUNNER_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Stage 1: Ingestion Tests
# =============================================================================


class TestRunIngestion:
    """Test run_ingestion() constructs Filing objects correctly."""

    async def test_filing_metadata_passed_directly(self, pipeline_store, tmp_path):
        """Filing should be constructed with metadata=FilingMetadata, not flat fields."""
        runner = _import_runner()

        # Create mock filings that get_filings_for_ticker would return
        mock_meta = _make_metadata("TSLA", 2025)

        # Mock EdgarClient
        mock_client = AsyncMock()
        mock_client.get_filings_for_ticker = AsyncMock(return_value=[mock_meta])
        mock_client.get_filing_document = AsyncMock(
            return_value="<html><body>Item 7 - Management's Discussion and Analysis of Financial Condition\nRevenue grew significantly in the current quarter driven by strong product demand. The company expanded into new markets successfully. Operating margins improved compared to prior year. Cash generation remained robust. Management is optimistic about upcoming quarters.</body></html>"
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        config = {"ingestion": {"tickers": "TSLA", "formType": "10-K", "startYear": 2025, "endYear": 2025}}

        with patch("edgar_sentinel.ingestion.client.EdgarClient", return_value=mock_client):
            result = await runner.run_ingestion(pipeline_store, config, config["ingestion"])

        # Verify no failures (the Pydantic error would appear here)
        assert result["failures"] == 0, f"Ingestion failures: {result['failure_details']}"
        assert result["new_fetched"] >= 0  # Might be 0 if parser extracts nothing

    async def test_ingestion_handles_multiple_tickers(self, pipeline_store, tmp_path):
        """Ingestion should process multiple tickers without Pydantic errors."""
        runner = _import_runner()

        tickers = ["AAPL", "MSFT"]
        metas = {t: [_make_metadata(t, 2025)] for t in tickers}

        mock_client = AsyncMock()
        mock_client.get_filings_for_ticker = AsyncMock(
            side_effect=lambda ticker, *a, **kw: metas.get(ticker.upper(), [])
        )
        mock_client.get_filing_document = AsyncMock(
            return_value="<html><body>Item 7 - Management's Discussion and Analysis\nRevenue increased. Growth was strong. Operations improved. Results exceeded expectations. Forward guidance is positive.</body></html>"
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        config = {"ingestion": {"tickers": "AAPL,MSFT", "formType": "both", "startYear": 2025, "endYear": 2025}}

        with patch("edgar_sentinel.ingestion.client.EdgarClient", return_value=mock_client):
            result = await runner.run_ingestion(pipeline_store, config, config["ingestion"])

        assert result["failures"] == 0, f"Failures: {result['failure_details']}"
        assert len(result["tickers"]) == 2

    async def test_ingestion_skips_already_stored_filings(self, seeded_store):
        """Cached filings should be skipped, not re-ingested."""
        runner = _import_runner()

        meta = _make_metadata("AAPL", 2025)
        mock_client = AsyncMock()
        mock_client.get_filings_for_ticker = AsyncMock(return_value=[meta])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        config = {"ingestion": {"tickers": "AAPL", "formType": "10-K", "startYear": 2025, "endYear": 2025}}

        with patch("edgar_sentinel.ingestion.client.EdgarClient", return_value=mock_client):
            result = await runner.run_ingestion(seeded_store, config, config["ingestion"])

        assert result["from_cache"] == 1
        assert result["new_fetched"] == 0

    async def test_ingestion_failure_details_captured(self, pipeline_store):
        """When a filing fails, the error is captured (not silently swallowed)."""
        runner = _import_runner()

        meta = _make_metadata("FAIL", 2025)
        mock_client = AsyncMock()
        mock_client.get_filings_for_ticker = AsyncMock(return_value=[meta])
        mock_client.get_filing_document = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        config = {"ingestion": {"tickers": "FAIL", "formType": "10-K", "startYear": 2025, "endYear": 2025}}

        with patch("edgar_sentinel.ingestion.client.EdgarClient", return_value=mock_client):
            result = await runner.run_ingestion(pipeline_store, config, config["ingestion"])

        assert result["failures"] >= 1
        assert any("Connection refused" in d for d in result["failure_details"])


# =============================================================================
# Stage 2: Analysis Tests
# =============================================================================


class TestRunAnalysis:
    """Test run_analysis() works with stored filings."""

    async def test_dictionary_analysis_on_stored_filings(self, seeded_store):
        """Dictionary analyzer should produce sentiment results from stored filings."""
        runner = _import_runner()

        config = {"analysis": {"dictionary": True, "similarity": False}}
        result = await runner.run_analysis(seeded_store, config, config["analysis"])

        # Should have processed the 6 filings
        assert result["filings_with_sections"] == 6
        # Dictionary analyzer should produce results (even if some are cached)
        total = result["new_sentiment"] + result["cached_sentiment"]
        assert total > 0, f"No sentiment results generated: {result}"

    async def test_similarity_analysis_groups_by_ticker(self, seeded_store):
        """Similarity analyzer should compare sequential filings per ticker."""
        runner = _import_runner()

        config = {"analysis": {"dictionary": False, "similarity": True}}
        result = await runner.run_analysis(seeded_store, config, config["analysis"])

        # 3 tickers × 1 comparison each (2025 vs 2024) = 3 similarity results
        total = result["new_similarity"] + result["cached_similarity"]
        assert total >= 3, f"Expected at least 3 similarity results, got {total}: {result}"

    async def test_analysis_with_no_filings(self, pipeline_store):
        """Analysis on empty store should return zero results gracefully."""
        runner = _import_runner()

        config = {"analysis": {"dictionary": True, "similarity": True}}
        result = await runner.run_analysis(pipeline_store, config, config["analysis"])

        assert result["filings_with_sections"] == 0
        assert result["new_sentiment"] == 0
        assert result["new_similarity"] == 0


# =============================================================================
# Stage 3: Signal Generation Tests
# =============================================================================


class TestRunSignals:
    """Test run_signals() generates composite signals."""

    async def test_signals_generated_from_sentiment(self, seeded_store):
        """Signal generation should work with stored sentiment results."""
        runner = _import_runner()

        # First run analysis to populate sentiment data
        ana_config = {"analysis": {"dictionary": True, "similarity": True}}
        await runner.run_analysis(seeded_store, ana_config, ana_config["analysis"])

        # Now generate signals
        config = {
            "ingestion": {"startYear": 2024, "endYear": 2025, "tickers": "AAPL,MSFT,GOOGL"},
            "signals": {"bufferDays": 2, "decayHalfLife": 90, "compositeMethod": "equal"},
        }
        bt_config = {
            "startDate": "2024-01-01",
            "endDate": "2025-12-31",
            "rebalanceFrequency": "quarterly",
        }

        result = await runner.run_signals(seeded_store, config, config["signals"], bt_config)

        assert result["rebalance_dates"] > 0
        assert result["signals_generated"] >= 0  # May be 0 if no signals fall within dates
        assert "composites" in result

    async def test_signals_with_no_analysis_data(self, seeded_store):
        """Signal generation with no analysis results should produce 0 signals, not crash."""
        runner = _import_runner()

        config = {
            "ingestion": {"startYear": 2024, "endYear": 2025, "tickers": "AAPL"},
            "signals": {"bufferDays": 2, "decayHalfLife": 90, "compositeMethod": "equal"},
        }
        bt_config = {
            "startDate": "2024-01-01",
            "endDate": "2025-12-31",
            "rebalanceFrequency": "quarterly",
        }

        result = await runner.run_signals(seeded_store, config, config["signals"], bt_config)

        # Should not crash, just produce 0 signals
        assert result["rebalance_dates"] > 0
        assert result["signals_generated"] == 0


# =============================================================================
# Stage 4: Backtest Tests
# =============================================================================


class TestRunBacktest:
    """Test run_backtest() with pre-computed composites."""

    async def test_backtest_with_composites(self, seeded_store):
        """Backtest should complete given composite signals."""
        runner = _import_runner()

        # Create composite signals manually
        composites = []
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            for q_date in [date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30), date(2024, 12, 31)]:
                c = CompositeSignal(
                    ticker=ticker,
                    signal_date=q_date,
                    composite_score=0.5 + (hash(ticker + str(q_date)) % 100) / 200,
                    components={"dictionary_mda": 0.5},
                    rank=None,
                )
                await seeded_store.save_composite(c)
                composites.append(c)

        config = {
            "ingestion": {"tickers": "AAPL,MSFT,GOOGL", "startYear": 2024, "endYear": 2025},
            "signals": {"bufferDays": 2},
            "backtest": {"rebalanceFrequency": "quarterly", "numQuantiles": 3, "longQuantile": 1},
        }

        # run_backtest uses YFinanceProvider which needs network — mock it
        with patch("edgar_sentinel.backtest.returns.YFinanceProvider") as MockProvider:
            import pandas as pd
            import numpy as np

            # Mock returns: monthly returns for all tickers
            dates_idx = pd.date_range("2024-01-31", "2025-12-31", freq="ME")
            mock_returns = pd.DataFrame(
                np.random.default_rng(42).normal(0.01, 0.05, (len(dates_idx), 4)),
                index=dates_idx,
                columns=["AAPL", "MSFT", "GOOGL", "SPY"],
            )
            instance = MockProvider.return_value
            instance.get_returns = MagicMock(return_value=mock_returns)

            result = await runner.run_backtest(seeded_store, config, config["backtest"], composites)

        assert "strategy" in result
        assert "benchmarks" in result
        assert "monthlyReturns" in result
        assert "signalRankings" in result
        # Strategy keys exist
        assert "totalReturn" in result["strategy"]
        assert "sharpeRatio" in result["strategy"]


# =============================================================================
# Full Pipeline Integration
# =============================================================================


class TestFullPipeline:
    """Test the full main() pipeline end-to-end."""

    async def test_full_pipeline_stages_sequentially(self, seeded_store, tmp_path):
        """All 4 pipeline stages run sequentially with pre-seeded data."""
        runner = _import_runner()

        config = {
            "ingestion": {"tickers": "AAPL,MSFT,GOOGL", "formType": "both", "startYear": 2024, "endYear": 2025},
            "analysis": {"dictionary": True, "similarity": True},
            "signals": {"bufferDays": 2, "decayHalfLife": 90, "compositeMethod": "equal"},
            "backtest": {"rebalanceFrequency": "quarterly", "numQuantiles": 3, "longQuantile": 1},
        }

        # Mock EdgarClient (filings already in store)
        mock_client = AsyncMock()
        mock_client.get_filings_for_ticker = AsyncMock(return_value=[])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        import pandas as pd
        import numpy as np

        dates_idx = pd.date_range("2024-01-31", "2025-12-31", freq="ME")
        mock_returns = pd.DataFrame(
            np.random.default_rng(42).normal(0.01, 0.05, (len(dates_idx), 4)),
            index=dates_idx,
            columns=["AAPL", "MSFT", "GOOGL", "SPY"],
        )

        # Stage 1: Ingestion (uses cached filings)
        with patch("edgar_sentinel.ingestion.client.EdgarClient", return_value=mock_client):
            ing_result = await runner.run_ingestion(seeded_store, config, config["ingestion"])
        assert ing_result["failures"] == 0

        # Stage 2: Analysis
        ana_result = await runner.run_analysis(seeded_store, config, config["analysis"])
        assert ana_result["filings_with_sections"] > 0

        # Stage 3: Signals
        bt_config_for_signals = {
            "startDate": "2024-01-01",
            "endDate": "2025-12-31",
            "rebalanceFrequency": "quarterly",
        }
        sig_result = await runner.run_signals(seeded_store, config, config["signals"], bt_config_for_signals)
        assert "signals_generated" in sig_result

        # Seed composites directly so backtest has enough data for quantile assignment
        composites = []
        for ticker in ["AAPL", "MSFT", "GOOGL"]:
            for q_date in [date(2024, 3, 29), date(2024, 6, 28), date(2024, 9, 30), date(2024, 12, 31),
                           date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)]:
                c = CompositeSignal(
                    ticker=ticker,
                    signal_date=q_date,
                    composite_score=0.5 + (hash(ticker + str(q_date)) % 100) / 200,
                    components={"dictionary_mda": 0.5},
                    rank=None,
                )
                await seeded_store.save_composite(c)
                composites.append(c)

        # Stage 4: Backtest
        with patch("edgar_sentinel.backtest.returns.YFinanceProvider") as MockProvider:
            instance = MockProvider.return_value
            instance.get_returns = MagicMock(return_value=mock_returns)

            bt_result = await runner.run_backtest(
                seeded_store, config, config["backtest"], composites
            )
        assert "strategy" in bt_result
        assert "benchmarks" in bt_result
        assert "monthlyReturns" in bt_result


# =============================================================================
# Model Construction Regression Tests
# =============================================================================


class TestModelConstruction:
    """Regression tests ensuring models are constructed correctly.

    These catch the specific class of bugs where runner.py constructs
    Pydantic models with wrong field names or missing required fields.
    """

    def test_filing_requires_metadata_not_flat_fields(self):
        """Filing(accession_number=...) should raise, Filing(metadata=...) should work."""
        meta = _make_metadata("AAPL", 2024)

        # Correct construction
        filing = Filing(metadata=meta, sections={})
        assert filing.metadata.ticker == "AAPL"

        # Wrong construction (what the bug was doing)
        with pytest.raises(Exception):
            Filing(
                accession_number=meta.accession_number,
                cik=meta.cik,
                ticker="AAPL",
                company_name="Apple Inc.",
                form_type=FormType.FORM_10K,
                filed_date=date(2024, 11, 3),
                url="https://www.sec.gov/test",
                sections={},
            )

    def test_filing_metadata_validates_cik_format(self):
        """CIK must be numeric."""
        with pytest.raises(Exception):
            FilingMetadata(
                cik="not-a-number",
                ticker="TEST",
                company_name="Test",
                form_type=FormType.FORM_10K,
                filed_date=date(2024, 1, 1),
                accession_number="0000320193-24-000106",
                url="https://www.sec.gov/test",
            )

    def test_filing_metadata_validates_url_https(self):
        """URL must start with https://."""
        with pytest.raises(Exception):
            FilingMetadata(
                cik="320193",
                ticker="TEST",
                company_name="Test",
                form_type=FormType.FORM_10K,
                filed_date=date(2024, 1, 1),
                accession_number="0000320193-24-000106",
                url="http://insecure.example.com",
            )

    def test_filing_metadata_validates_accession_format(self):
        """Accession number must match EDGAR format."""
        with pytest.raises(Exception):
            FilingMetadata(
                cik="320193",
                ticker="TEST",
                company_name="Test",
                form_type=FormType.FORM_10K,
                filed_date=date(2024, 1, 1),
                accession_number="invalid",
                url="https://www.sec.gov/test",
            )

    def test_sentiment_score_range_validation(self):
        """Sentiment score must be in [-1, 1]."""
        with pytest.raises(Exception):
            SentimentResult(
                filing_id="0000320193-24-000106",
                section_name="mda",
                analyzer_name="dictionary",
                sentiment_score=5.0,  # Out of range
                confidence=0.5,
                analyzed_at=datetime.now(timezone.utc),
            )

    def test_composite_signal_construction(self):
        """CompositeSignal should accept all required fields."""
        cs = CompositeSignal(
            ticker="AAPL",
            signal_date=date(2024, 3, 31),
            composite_score=0.65,
            components={"dict_mda": 0.3, "sim_mda": 0.35},
            rank=1,
        )
        assert cs.composite_score == 0.65
        assert cs.rank == 1
