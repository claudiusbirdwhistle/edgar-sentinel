"""Tests for the CLI module."""

from __future__ import annotations

import json
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from edgar_sentinel.cli import cli, _resolve_form_types, _resolve_tickers
from edgar_sentinel.core.models import (
    CompositeSignal,
    Filing,
    FilingMetadata,
    FilingSection,
    FormType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_config():
    """Minimal SentinelConfig mock for CLI tests."""
    config = MagicMock()
    config.storage.backend.value = "sqlite"
    config.storage.sqlite_path = ":memory:"
    config.analyzers.dictionary.enabled = True
    config.analyzers.similarity.enabled = True
    config.analyzers.llm.enabled = False
    return config


@pytest.fixture
def sample_filings():
    """Create sample Filing objects for testing."""
    meta1 = FilingMetadata(
        cik="320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type=FormType.FORM_10K,
        filed_date=date(2023, 11, 3),
        accession_number="0000320193-23-000106",
        url="https://www.sec.gov/test1",
    )
    section1 = FilingSection(
        filing_id="0000320193-23-000106",
        section_name="mda",
        raw_text="Test filing content for Apple",
        word_count=5,
        extracted_at=datetime(2024, 1, 15, 10, 0, 0),
    )
    meta2 = FilingMetadata(
        cik="789019",
        ticker="MSFT",
        company_name="Microsoft Corp",
        form_type=FormType.FORM_10K,
        filed_date=date(2023, 10, 24),
        accession_number="0000789019-23-000100",
        url="https://www.sec.gov/test2",
    )
    section2 = FilingSection(
        filing_id="0000789019-23-000100",
        section_name="mda",
        raw_text="Test filing content for Microsoft",
        word_count=5,
        extracted_at=datetime(2024, 1, 15, 10, 0, 0),
    )
    return [
        Filing(metadata=meta1, sections={"mda": section1}),
        Filing(metadata=meta2, sections={"mda": section2}),
    ]


@pytest.fixture
def sample_composites():
    """Sample CompositeSignal objects for testing."""
    return [
        CompositeSignal(
            ticker="AAPL",
            signal_date=date(2023, 11, 5),
            composite_score=0.72,
            components={"dictionary_mda": 0.8, "similarity_mda": 0.6},
            rank=1,
        ),
        CompositeSignal(
            ticker="MSFT",
            signal_date=date(2023, 11, 5),
            composite_score=0.55,
            components={"dictionary_mda": 0.6, "similarity_mda": 0.5},
            rank=2,
        ),
        CompositeSignal(
            ticker="GOOGL",
            signal_date=date(2023, 11, 5),
            composite_score=-0.23,
            components={"dictionary_mda": -0.3, "similarity_mda": -0.1},
            rank=3,
        ),
    ]


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestResolveFormTypes:
    def test_both(self):
        result = _resolve_form_types("both")
        assert FormType.FORM_10K in result
        assert FormType.FORM_10Q in result
        assert len(result) == 2

    def test_10k(self):
        result = _resolve_form_types("10-K")
        assert result == [FormType.FORM_10K]

    def test_10q(self):
        result = _resolve_form_types("10-Q")
        assert result == [FormType.FORM_10Q]

    def test_unknown_raises(self):
        from click import UsageError

        with pytest.raises(UsageError):
            _resolve_form_types("8-K")


class TestResolveTickers:
    def test_custom_with_tickers(self):
        result = _resolve_tickers("custom", "AAPL,MSFT,GOOGL")
        assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_custom_normalizes_case(self):
        result = _resolve_tickers("custom", "aapl, msft")
        assert result == ["AAPL", "MSFT"]

    def test_custom_no_tickers_raises(self):
        from click import UsageError

        with pytest.raises(UsageError, match="--tickers is required"):
            _resolve_tickers("custom", None)

    def test_custom_empty_string_raises(self):
        from click import UsageError

        with pytest.raises(UsageError, match="--tickers is required"):
            _resolve_tickers("custom", "")

    def test_unknown_universe_raises(self):
        from click import UsageError

        with pytest.raises(UsageError, match="not yet supported"):
            _resolve_tickers("sp500", None)


# ---------------------------------------------------------------------------
# CLI group tests
# ---------------------------------------------------------------------------


class TestCLIGroup:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "EDGAR Sentinel" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_unknown_command(self, runner):
        result = runner.invoke(cli, ["nonexistent"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# ingest command tests
# ---------------------------------------------------------------------------


class TestIngestCommand:
    def test_ingest_help(self, runner):
        result = runner.invoke(cli, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "Download and parse" in result.output
        assert "--tickers" in result.output
        assert "--start" in result.output
        assert "--end" in result.output

    def test_ingest_requires_start_end(self, runner):
        result = runner.invoke(cli, ["ingest", "--tickers", "AAPL"])
        assert result.exit_code != 0

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_ingest_runs(self, mock_store_fn, mock_load, runner, mock_config):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.filing_exists = AsyncMock(return_value=False)
        mock_store.save_filing = AsyncMock()
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        with patch("edgar_sentinel.ingestion.EdgarClient") as mc, \
             patch("edgar_sentinel.ingestion.FilingParser") as mp:
            mc_inst = AsyncMock()
            mc_inst.search_filings = AsyncMock(return_value=[])
            mc.return_value = mc_inst
            mp.return_value = MagicMock()

            result = runner.invoke(
                cli, ["ingest", "--tickers", "AAPL", "--start", "2023", "--end", "2023"]
            )
            assert result.exit_code == 0
            assert "Ingested" in result.output


# ---------------------------------------------------------------------------
# analyze command tests
# ---------------------------------------------------------------------------


class TestAnalyzeCommand:
    def test_analyze_help(self, runner):
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "Run sentiment analyzers" in result.output
        assert "--analyzer" in result.output

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    @patch("edgar_sentinel.cli._resolve_analyzers")
    def test_analyze_no_filings(
        self, mock_ra, mock_store_fn, mock_load, runner, mock_config
    ):
        mock_load.return_value = mock_config
        mock_az = MagicMock()
        mock_az.name = "dictionary"
        mock_ra.return_value = [mock_az]
        mock_store = AsyncMock()
        mock_store.list_filings = AsyncMock(return_value=[])
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        result = runner.invoke(cli, ["analyze"])
        assert result.exit_code == 1
        assert "No filings found" in result.output

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_analyze_with_filings(
        self, mock_store_fn, mock_load, runner, mock_config, sample_filings
    ):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.list_filings = AsyncMock(return_value=sample_filings)
        mock_store.save_sentiment = AsyncMock()
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        with patch("edgar_sentinel.cli._resolve_analyzers") as mock_ra:
            mock_az = MagicMock()
            mock_az.name = "dictionary"
            mock_az.analyze.return_value = [MagicMock()]
            mock_ra.return_value = [mock_az]

            result = runner.invoke(cli, ["analyze"])
            assert result.exit_code == 0
            assert "Generated" in result.output


# ---------------------------------------------------------------------------
# signals command tests
# ---------------------------------------------------------------------------


class TestSignalsCommand:
    def test_signals_help(self, runner):
        result = runner.invoke(cli, ["signals", "--help"])
        assert result.exit_code == 0
        assert "Build and display composite signals" in result.output

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_signals_no_data(self, mock_store_fn, mock_load, runner, mock_config):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.get_composites = AsyncMock(return_value=[])
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        result = runner.invoke(cli, ["signals"])
        assert result.exit_code == 1
        assert "No composite signals" in result.output

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_signals_table_output(
        self, mock_store_fn, mock_load, runner, mock_config, sample_composites
    ):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.get_composites = AsyncMock(return_value=sample_composites)
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        result = runner.invoke(cli, ["signals"])
        assert result.exit_code == 0
        assert "AAPL" in result.output
        assert "MSFT" in result.output

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_signals_json_output(
        self, mock_store_fn, mock_load, runner, mock_config, sample_composites
    ):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.get_composites = AsyncMock(return_value=sample_composites)
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        result = runner.invoke(cli, ["signals", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 3
        assert data[0]["ticker"] in ("AAPL", "MSFT", "GOOGL")

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_signals_csv_output(
        self, mock_store_fn, mock_load, runner, mock_config, sample_composites
    ):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.get_composites = AsyncMock(return_value=sample_composites)
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        result = runner.invoke(cli, ["signals", "--format", "csv"])
        assert result.exit_code == 0
        lines = result.output.strip().split("\n")
        assert len(lines) == 4  # header + 3 data rows
        assert "ticker" in lines[0]
        assert "composite_score" in lines[0]

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_signals_filter_by_ticker(
        self, mock_store_fn, mock_load, runner, mock_config, sample_composites
    ):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.get_composites = AsyncMock(return_value=sample_composites[:1])
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        result = runner.invoke(cli, ["signals", "--ticker", "AAPL", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["ticker"] == "AAPL"

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_signals_filter_by_date(
        self, mock_store_fn, mock_load, runner, mock_config, sample_composites
    ):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.get_composites = AsyncMock(return_value=sample_composites)
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        # Filter to a date that doesn't match any signals
        result = runner.invoke(
            cli, ["signals", "--date", "2024-01-01", "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 0


# ---------------------------------------------------------------------------
# backtest command tests
# ---------------------------------------------------------------------------


class TestBacktestCommand:
    def test_backtest_help(self, runner):
        result = runner.invoke(cli, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "Run a backtest" in result.output
        assert "--start" in result.output
        assert "--end" in result.output
        assert "--rebalance" in result.output

    def test_backtest_requires_start_end(self, runner):
        result = runner.invoke(cli, ["backtest"])
        assert result.exit_code != 0

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_backtest_no_signals(self, mock_store_fn, mock_load, runner, mock_config):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.get_composites = AsyncMock(return_value=[])
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        result = runner.invoke(
            cli, ["backtest", "--start", "2020-01-01", "--end", "2024-12-31"]
        )
        assert result.exit_code == 1
        assert "No composite signals" in result.output

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_backtest_table_output(
        self, mock_store_fn, mock_load, runner, mock_config, sample_composites
    ):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.get_composites = AsyncMock(return_value=sample_composites)
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        with patch("edgar_sentinel.backtest.run_backtest") as mbt, \
             patch("edgar_sentinel.backtest.MetricsCalculator") as mmc:
            mock_result = MagicMock()
            mock_result.total_return = 0.423
            mock_result.annualized_return = 0.073
            mock_result.sharpe_ratio = 0.82
            mock_result.max_drawdown = -0.187
            mock_result.turnover = 0.342
            mock_result.information_ratio = 0.61
            mock_result.factor_exposures = None
            mock_result.model_dump.return_value = {"total_return": 0.423}
            mbt.return_value = mock_result

            mc_inst = MagicMock()
            mc_inst.compute_all.return_value = {"sortino_ratio": 1.14}
            mmc.return_value = mc_inst

            result = runner.invoke(
                cli,
                ["backtest", "--start", "2020-01-01", "--end", "2024-12-31"],
            )
            assert result.exit_code == 0
            assert "42.3%" in result.output
            assert "Sharpe" in result.output

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_backtest_json_output(
        self, mock_store_fn, mock_load, runner, mock_config, sample_composites
    ):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.get_composites = AsyncMock(return_value=sample_composites)
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        with patch("edgar_sentinel.backtest.run_backtest") as mbt, \
             patch("edgar_sentinel.backtest.MetricsCalculator") as mmc:
            mock_result = MagicMock()
            mock_result.total_return = 0.1
            mock_result.annualized_return = 0.05
            mock_result.sharpe_ratio = 0.5
            mock_result.max_drawdown = -0.1
            mock_result.turnover = 0.2
            mock_result.information_ratio = None
            mock_result.factor_exposures = None
            mock_result.model_dump.return_value = {"total_return": 0.1}
            mbt.return_value = mock_result

            mc_inst = MagicMock()
            mc_inst.compute_all.return_value = {}
            mmc.return_value = mc_inst

            result = runner.invoke(
                cli,
                [
                    "backtest",
                    "--start",
                    "2020-01-01",
                    "--end",
                    "2024-12-31",
                    "--format",
                    "json",
                ],
            )
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "result" in data
            assert "metrics" in data


# ---------------------------------------------------------------------------
# serve command tests
# ---------------------------------------------------------------------------


class TestServeCommand:
    def test_serve_help(self, runner):
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the REST API" in result.output
        assert "--port" in result.output
        assert "--host" in result.output


# ---------------------------------------------------------------------------
# status command tests
# ---------------------------------------------------------------------------


class TestStatusCommand:
    def test_status_help(self, runner):
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0
        assert "Show system status" in result.output

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_status_empty_db(self, mock_store_fn, mock_load, runner, mock_config):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.list_filings = AsyncMock(return_value=[])
        mock_store.get_composites = AsyncMock(return_value=[])
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Status" in result.output
        assert "0" in result.output

    @patch("edgar_sentinel.core.load_config")
    @patch("edgar_sentinel.cli._create_store_async")
    def test_status_with_data(
        self, mock_store_fn, mock_load, runner, mock_config, sample_filings, sample_composites
    ):
        mock_load.return_value = mock_config
        mock_store = AsyncMock()
        mock_store.list_filings = AsyncMock(return_value=sample_filings)
        mock_store.get_composites = AsyncMock(return_value=sample_composites)
        mock_store.close = AsyncMock()
        mock_store_fn.return_value = mock_store

        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "2" in result.output


# ---------------------------------------------------------------------------
# Module-level tests
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_cli_importable(self):
        from edgar_sentinel.cli import cli, main

        assert cli is not None
        assert main is not None

    def test_entry_point_name(self):
        """Verify the CLI group is named correctly."""
        assert cli.name == "cli"
