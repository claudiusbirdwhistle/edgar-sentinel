"""Integration tests for the CLI.

Uses Click's CliRunner — no subprocesses.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from edgar_sentinel.cli import cli


pytestmark = pytest.mark.integration


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIHelp:
    """CLI help and version commands."""

    def test_help_output(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "EDGAR Sentinel" in result.output

    def test_version_output(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    def test_subcommand_help_ingest(self, runner):
        result = runner.invoke(cli, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--tickers" in result.output

    def test_subcommand_help_analyze(self, runner):
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--analyzer" in result.output

    def test_subcommand_help_signals(self, runner):
        result = runner.invoke(cli, ["signals", "--help"])
        assert result.exit_code == 0

    def test_subcommand_help_backtest(self, runner):
        result = runner.invoke(cli, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "--start" in result.output

    def test_subcommand_help_serve(self, runner):
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output

    def test_subcommand_help_status(self, runner):
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0


class TestCLIIngest:
    """CLI ingest command validation."""

    def test_ingest_missing_required_options(self, runner):
        """Ingest requires --start and --end."""
        result = runner.invoke(cli, ["ingest", "--tickers", "AAPL"])
        assert result.exit_code != 0

    def test_ingest_requires_tickers_for_custom_universe(self, runner):
        """Custom universe requires --tickers."""
        result = runner.invoke(cli, [
            "ingest", "--universe", "custom", "--start", "2023", "--end", "2023",
        ])
        assert result.exit_code != 0


class TestCLIAnalyze:
    """CLI analyze command validation."""

    def test_analyze_help_shows_analyzer_choices(self, runner):
        result = runner.invoke(cli, ["analyze", "--help"])
        assert "dictionary" in result.output
        assert "similarity" in result.output
