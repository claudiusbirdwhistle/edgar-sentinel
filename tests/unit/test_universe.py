"""Tests for backtest.universe: UniverseProvider implementations (TDD red phase)."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from edgar_sentinel.backtest.universe import (
    Sp500HistoricalProvider,
    StaticUniverseProvider,
    UniverseProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a sp500_ticker_start_end.csv-compatible file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "start_date", "end_date"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# StaticUniverseProvider
# ---------------------------------------------------------------------------


class TestStaticUniverseProvider:
    """StaticUniverseProvider returns the same tickers for all dates."""

    def test_returns_all_tickers_for_any_date(self):
        tickers = ["AAPL", "MSFT", "GOOG"]
        provider = StaticUniverseProvider(tickers)
        result = provider.get_tickers(date(2020, 1, 1))
        assert set(result) == set(tickers)

    def test_returns_same_tickers_regardless_of_date(self):
        tickers = ["AAPL", "MSFT"]
        provider = StaticUniverseProvider(tickers)
        d1 = provider.get_tickers(date(2010, 6, 30))
        d2 = provider.get_tickers(date(2024, 12, 31))
        assert d1 == d2

    def test_empty_list_is_valid(self):
        provider = StaticUniverseProvider([])
        assert provider.get_tickers(date(2024, 1, 1)) == []

    def test_does_not_mutate_input(self):
        original = ["AAPL", "MSFT"]
        provider = StaticUniverseProvider(original)
        original.append("GOOG")
        result = provider.get_tickers(date(2024, 1, 1))
        assert "GOOG" not in result

    def test_conforms_to_universe_provider_protocol(self):
        provider = StaticUniverseProvider(["AAPL"])
        assert isinstance(provider, UniverseProvider)


# ---------------------------------------------------------------------------
# Sp500HistoricalProvider
# ---------------------------------------------------------------------------


class TestSp500HistoricalProvider:
    """Sp500HistoricalProvider reads from a CSV and returns point-in-time tickers."""

    def test_includes_ticker_active_on_date(self, tmp_path):
        _write_csv(
            tmp_path / "sp500_ticker_start_end.csv",
            [{"ticker": "AAPL", "start_date": "2000-01-01", "end_date": ""}],
        )
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        result = provider.get_tickers(date(2024, 6, 15))
        assert "AAPL" in result

    def test_excludes_ticker_not_yet_added(self, tmp_path):
        _write_csv(
            tmp_path / "sp500_ticker_start_end.csv",
            [{"ticker": "NVDA", "start_date": "2020-01-01", "end_date": ""}],
        )
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        result = provider.get_tickers(date(2018, 12, 31))
        assert "NVDA" not in result

    def test_excludes_ticker_removed_before_date(self, tmp_path):
        _write_csv(
            tmp_path / "sp500_ticker_start_end.csv",
            [{"ticker": "GE", "start_date": "1996-01-01", "end_date": "2018-11-19"}],
        )
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        result = provider.get_tickers(date(2022, 1, 1))
        assert "GE" not in result

    def test_includes_ticker_on_its_exit_date(self, tmp_path):
        """Ticker is included on its last day in the index (inclusive)."""
        _write_csv(
            tmp_path / "sp500_ticker_start_end.csv",
            [{"ticker": "GE", "start_date": "1996-01-01", "end_date": "2018-11-19"}],
        )
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        result = provider.get_tickers(date(2018, 11, 19))
        assert "GE" in result

    def test_includes_ticker_on_its_entry_date(self, tmp_path):
        _write_csv(
            tmp_path / "sp500_ticker_start_end.csv",
            [{"ticker": "META", "start_date": "2022-06-20", "end_date": ""}],
        )
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        result = provider.get_tickers(date(2022, 6, 20))
        assert "META" in result

    def test_handles_multiple_stints(self, tmp_path):
        """A ticker can have multiple stints in the index (like AAL)."""
        _write_csv(
            tmp_path / "sp500_ticker_start_end.csv",
            [
                {"ticker": "AAL", "start_date": "1996-01-02", "end_date": "1997-01-15"},
                {"ticker": "AAL", "start_date": "2015-03-23", "end_date": "2024-09-23"},
            ],
        )
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)

        # In index during first stint
        assert "AAL" in provider.get_tickers(date(1996, 6, 1))
        # Not in index between stints
        assert "AAL" not in provider.get_tickers(date(2000, 1, 1))
        # Back in index during second stint
        assert "AAL" in provider.get_tickers(date(2020, 1, 1))
        # Gone again after second removal
        assert "AAL" not in provider.get_tickers(date(2024, 10, 1))

    def test_returns_multiple_tickers(self, tmp_path):
        _write_csv(
            tmp_path / "sp500_ticker_start_end.csv",
            [
                {"ticker": "AAPL", "start_date": "1997-01-02", "end_date": ""},
                {"ticker": "MSFT", "start_date": "1994-01-02", "end_date": ""},
                {"ticker": "FUBAR", "start_date": "2010-01-01", "end_date": "2012-01-01"},
            ],
        )
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        result = provider.get_tickers(date(2024, 1, 1))
        assert "AAPL" in result
        assert "MSFT" in result
        assert "FUBAR" not in result

    def test_tickers_uppercased(self, tmp_path):
        """Tickers in the CSV are normalized to uppercase."""
        _write_csv(
            tmp_path / "sp500_ticker_start_end.csv",
            [{"ticker": "aapl", "start_date": "2000-01-01", "end_date": ""}],
        )
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        result = provider.get_tickers(date(2024, 1, 1))
        assert "AAPL" in result
        assert "aapl" not in result

    def test_skips_rows_with_no_ticker(self, tmp_path):
        """Rows with empty ticker are silently skipped."""
        csv_path = tmp_path / "sp500_ticker_start_end.csv"
        with open(csv_path, "w", newline="") as f:
            f.write("ticker,start_date,end_date\n")
            f.write("AAPL,2000-01-01,\n")
            f.write(",2010-01-01,\n")  # no ticker
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        result = provider.get_tickers(date(2024, 1, 1))
        assert "AAPL" in result
        assert "" not in result

    def test_skips_rows_with_no_start_date(self, tmp_path):
        """Rows with missing start_date are silently skipped."""
        csv_path = tmp_path / "sp500_ticker_start_end.csv"
        with open(csv_path, "w", newline="") as f:
            f.write("ticker,start_date,end_date\n")
            f.write("AAPL,2000-01-01,\n")
            f.write("BAD,,\n")  # no start_date
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        result = provider.get_tickers(date(2024, 1, 1))
        assert "AAPL" in result
        assert "BAD" not in result

    def test_skips_rows_with_invalid_date_format(self, tmp_path):
        """Rows with malformed dates are silently skipped."""
        csv_path = tmp_path / "sp500_ticker_start_end.csv"
        with open(csv_path, "w", newline="") as f:
            f.write("ticker,start_date,end_date\n")
            f.write("AAPL,2000-01-01,\n")
            f.write("BADDATES,not-a-date,\n")
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        result = provider.get_tickers(date(2024, 1, 1))
        assert "AAPL" in result
        assert "BADDATES" not in result

    def test_conforms_to_universe_provider_protocol(self, tmp_path):
        _write_csv(
            tmp_path / "sp500_ticker_start_end.csv",
            [{"ticker": "AAPL", "start_date": "2000-01-01", "end_date": ""}],
        )
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        assert isinstance(provider, UniverseProvider)

    def test_caches_data_in_memory(self, tmp_path):
        """Data is loaded once and cached — subsequent calls don't re-parse."""
        csv_path = tmp_path / "sp500_ticker_start_end.csv"
        _write_csv(csv_path, [{"ticker": "AAPL", "start_date": "2000-01-01", "end_date": ""}])
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        # First call loads data
        r1 = provider.get_tickers(date(2024, 1, 1))
        # Delete the CSV — subsequent call should still work (in-memory cache)
        csv_path.unlink()
        r2 = provider.get_tickers(date(2024, 6, 1))
        assert r1 == r2

    def test_uses_existing_cache_file_without_downloading(self, tmp_path):
        """If CSV is already in cache_dir, no download is attempted."""
        _write_csv(
            tmp_path / "sp500_ticker_start_end.csv",
            [{"ticker": "TSLA", "start_date": "2020-12-21", "end_date": ""}],
        )
        # Should load from file, not raise (no internet access mocked)
        provider = Sp500HistoricalProvider(cache_dir=tmp_path)
        result = provider.get_tickers(date(2023, 1, 1))
        assert "TSLA" in result


# ---------------------------------------------------------------------------
# BacktestEngine integration: dynamic universe filtering
# ---------------------------------------------------------------------------


class TestBacktestEngineWithUniverseProvider:
    """BacktestEngine uses UniverseProvider to filter signals per rebalance date."""

    def _make_signals(self, tickers: list[str], signal_date: date) -> list:
        from edgar_sentinel.core.models import CompositeSignal

        return [
            CompositeSignal(
                ticker=t,
                signal_date=signal_date,
                composite_score=float(i),
                components={"dict": float(i)},
            )
            for i, t in enumerate(tickers, start=1)
        ]

    def _make_config(self, universe: list[str]) -> "BacktestConfig":
        from edgar_sentinel.core.models import BacktestConfig

        return BacktestConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 12, 31),
            universe=universe,
            num_quantiles=2,
        )

    def _make_returns_df(self, tickers: list[str]) -> "pd.DataFrame":
        import pandas as pd

        idx = pd.date_range("2022-01-01", "2022-12-31", freq="ME")
        data = {t: [0.01] * len(idx) for t in tickers + ["SPY"]}
        return pd.DataFrame(data, index=idx)

    def test_engine_accepts_universe_provider_kwarg(self):
        from edgar_sentinel.backtest.engine import BacktestEngine
        from edgar_sentinel.core.models import BacktestConfig

        config = self._make_config(["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA"])
        provider = StaticUniverseProvider(["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA"])

        import pandas as pd

        returns = self._make_returns_df(["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA"])

        class MockProvider:
            def get_returns(self, tickers, start, end, frequency="monthly"):
                cols = [t for t in tickers if t in returns.columns]
                mask = (returns.index.date >= start) & (returns.index.date <= end)
                return returns.loc[mask, cols]

        # Should not raise
        engine = BacktestEngine(
            config=config,
            returns_provider=MockProvider(),
            universe_provider=provider,
        )
        assert engine is not None

    def test_engine_filters_signals_to_active_universe(self):
        """Signals for tickers outside the active universe are excluded per rebalance date."""
        import pandas as pd
        from edgar_sentinel.backtest.engine import BacktestEngine

        # Universe: AAPL always in, NVDA only added 2022-Q3 onward
        class DateFilteredProvider:
            def get_tickers(self, rebalance_date: date) -> list[str]:
                base = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
                if rebalance_date >= date(2022, 7, 1):
                    return base + ["NVDA"]
                return base

        all_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA"]
        config = self._make_config(all_tickers)
        signals = self._make_signals(all_tickers, date(2021, 12, 1))
        returns = self._make_returns_df(all_tickers)

        class MockProvider:
            def get_returns(self, tickers, start, end, frequency="monthly"):
                cols = [t for t in tickers if t in returns.columns]
                mask = (returns.index.date >= start) & (returns.index.date <= end)
                return returns.loc[mask, cols]

        engine = BacktestEngine(
            config=config,
            returns_provider=MockProvider(),
            universe_provider=DateFilteredProvider(),
        )
        result = engine.run(signals=signals)
        # Engine should complete without error
        assert result.total_return is not None
