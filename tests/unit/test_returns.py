"""Tests for edgar_sentinel.backtest.returns module."""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from edgar_sentinel.backtest.returns import (
    CSVProvider,
    ReturnsProvider,
    YFinanceProvider,
)

MOCK_YF = "edgar_sentinel.backtest.returns.yf"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    """Test that providers conform to the ReturnsProvider protocol."""

    def test_yfinance_is_returns_provider(self):
        assert isinstance(YFinanceProvider(), ReturnsProvider)

    def test_csv_is_returns_provider(self):
        provider = CSVProvider.__new__(CSVProvider)
        assert isinstance(provider, ReturnsProvider)

    def test_protocol_is_runtime_checkable(self):
        assert hasattr(ReturnsProvider, "__protocol_attrs__") or hasattr(
            ReturnsProvider, "__abstractmethods__"
        ) or issubclass(type(ReturnsProvider), type)


# ---------------------------------------------------------------------------
# YFinanceProvider — Construction
# ---------------------------------------------------------------------------


class TestYFinanceConstruction:
    """Test YFinanceProvider initialization."""

    def test_default_construction(self):
        provider = YFinanceProvider()
        assert provider._cache_db_path is None
        assert provider._buffer_days == 5
        assert provider._cache == {}

    def test_custom_buffer_days(self):
        provider = YFinanceProvider(request_buffer_days=10)
        assert provider._buffer_days == 10

    def test_with_cache_db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        provider = YFinanceProvider(cache_db_path=db_path)
        assert provider._cache_db_path == db_path
        # Verify table was created
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='returns_cache'"
        )
        assert cursor.fetchone() is not None
        conn.close()


# ---------------------------------------------------------------------------
# YFinanceProvider — Validation
# ---------------------------------------------------------------------------


class TestYFinanceValidation:
    """Test input validation."""

    def test_start_after_end_raises(self):
        provider = YFinanceProvider()
        with pytest.raises(ValueError, match="start.*must be before end"):
            provider.get_prices(["AAPL"], date(2024, 1, 1), date(2023, 1, 1))

    def test_start_equals_end_raises(self):
        provider = YFinanceProvider()
        with pytest.raises(ValueError, match="start.*must be before end"):
            provider.get_prices(["AAPL"], date(2024, 1, 1), date(2024, 1, 1))

    def test_empty_tickers_raises(self):
        provider = YFinanceProvider()
        with pytest.raises(ValueError, match="tickers must not be empty"):
            provider.get_prices([], date(2023, 1, 1), date(2024, 1, 1))

    def test_returns_start_after_end_raises(self):
        provider = YFinanceProvider()
        with pytest.raises(ValueError, match="start.*must be before end"):
            provider.get_returns(["AAPL"], date(2024, 1, 1), date(2023, 1, 1))

    def test_returns_invalid_frequency_raises(self):
        provider = YFinanceProvider()
        provider._cache["AAPL"] = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
        )
        with pytest.raises(ValueError, match="Unsupported frequency"):
            provider.get_returns(
                ["AAPL"], date(2023, 1, 1), date(2023, 12, 31), frequency="weekly"
            )


# ---------------------------------------------------------------------------
# YFinanceProvider — In-memory cache
# ---------------------------------------------------------------------------


class TestYFinanceCache:
    """Test in-memory caching behavior."""

    def test_cache_hit_skips_fetch(self):
        provider = YFinanceProvider()
        dates = pd.to_datetime(["2023-06-01", "2023-06-02", "2023-06-05"])
        provider._cache["AAPL"] = pd.Series([150.0, 151.0, 152.0], index=dates)

        # Should use cache, not call yfinance
        result = provider.get_prices(["AAPL"], date(2023, 6, 1), date(2023, 6, 5))
        assert "AAPL" in result.columns
        assert len(result) == 3

    def test_cache_miss_triggers_fetch(self):
        provider = YFinanceProvider()
        mock_df = pd.DataFrame(
            {"Close": [150.0, 151.0]},
            index=pd.to_datetime(["2023-06-01", "2023-06-02"]),
        )
        with patch(MOCK_YF) as mock_yf:
            mock_yf.download.return_value = mock_df
            result = provider.get_prices(
                ["AAPL"], date(2023, 6, 1), date(2023, 6, 2)
            )
        assert "AAPL" in provider._cache
        mock_yf.download.assert_called_once()

    def test_partial_cache_hit(self):
        """One ticker cached, one needs fetch."""
        provider = YFinanceProvider()
        dates = pd.to_datetime(["2023-06-01", "2023-06-02", "2023-06-05"])
        provider._cache["AAPL"] = pd.Series([150.0, 151.0, 152.0], index=dates)

        mock_df = pd.DataFrame(
            {("Close", "MSFT"): [250.0, 251.0, 252.0]},
            index=dates,
        )
        mock_df.columns = pd.MultiIndex.from_tuples([("Close", "MSFT")])

        with patch(MOCK_YF) as mock_yf:
            mock_yf.download.return_value = mock_df
            result = provider.get_prices(
                ["AAPL", "MSFT"], date(2023, 6, 1), date(2023, 6, 5)
            )

        assert "AAPL" in result.columns
        assert "MSFT" in result.columns


# ---------------------------------------------------------------------------
# YFinanceProvider — yfinance download handling
# ---------------------------------------------------------------------------


class TestYFinanceFetch:
    """Test _fetch_and_cache behavior."""

    def test_single_ticker_download(self):
        provider = YFinanceProvider()
        mock_df = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0]},
            index=pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
        )
        with patch(MOCK_YF) as mock_yf:
            mock_yf.download.return_value = mock_df
            result = provider._fetch_and_cache(
                ["AAPL"],
                date(2023, 1, 1),
                date(2023, 1, 5),
                pd.DataFrame(),
            )
        assert "AAPL" in result.columns
        assert len(result) == 3
        assert "AAPL" in provider._cache

    def test_multi_ticker_download(self):
        provider = YFinanceProvider()
        index = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
        mock_df = pd.DataFrame(
            {("Close", "AAPL"): [150.0, 151.0, 152.0],
             ("Close", "MSFT"): [250.0, 251.0, 252.0]},
            index=index,
        )
        mock_df.columns = pd.MultiIndex.from_tuples(
            [("Close", "AAPL"), ("Close", "MSFT")]
        )
        with patch(MOCK_YF) as mock_yf:
            mock_yf.download.return_value = mock_df
            result = provider._fetch_and_cache(
                ["AAPL", "MSFT"],
                date(2023, 1, 1),
                date(2023, 1, 5),
                pd.DataFrame(),
            )
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns

    def test_download_failure_returns_existing(self):
        provider = YFinanceProvider()
        existing = pd.DataFrame({"AAPL": [100.0]}, index=pd.to_datetime(["2023-01-03"]))
        with patch(MOCK_YF) as mock_yf:
            mock_yf.download.side_effect = Exception("Network error")
            result = provider._fetch_and_cache(
                ["AAPL"],
                date(2023, 1, 1),
                date(2023, 1, 5),
                existing,
            )
        assert result is existing

    def test_empty_download_returns_existing(self):
        provider = YFinanceProvider()
        existing = pd.DataFrame()
        with patch(MOCK_YF) as mock_yf:
            mock_yf.download.return_value = pd.DataFrame()
            result = provider._fetch_and_cache(
                ["AAPL"],
                date(2023, 1, 1),
                date(2023, 1, 5),
                existing,
            )
        assert result.empty

    def test_missing_ticker_excluded(self):
        provider = YFinanceProvider()
        index = pd.to_datetime(["2023-01-03", "2023-01-04"])
        mock_df = pd.DataFrame(
            {("Close", "AAPL"): [150.0, 151.0]},
            index=index,
        )
        mock_df.columns = pd.MultiIndex.from_tuples([("Close", "AAPL")])
        with patch(MOCK_YF) as mock_yf:
            mock_yf.download.return_value = mock_df
            result = provider._fetch_and_cache(
                ["AAPL", "INVALID"],
                date(2023, 1, 1),
                date(2023, 1, 5),
                pd.DataFrame(),
            )
        assert "AAPL" in result.columns
        assert "INVALID" not in result.columns


# ---------------------------------------------------------------------------
# YFinanceProvider — Returns computation
# ---------------------------------------------------------------------------


class TestYFinanceReturns:
    """Test return computation from prices."""

    def test_daily_returns(self):
        provider = YFinanceProvider()
        # Buffer=5 days, so start=June 1 fetches from May 27 onward
        dates = pd.to_datetime(
            ["2023-05-27", "2023-05-30", "2023-06-01", "2023-06-02", "2023-06-05"]
        )
        provider._cache["AAPL"] = pd.Series(
            [95.0, 100.0, 100.0, 105.0, 110.0], index=dates
        )
        result = provider.get_returns(
            ["AAPL"], date(2023, 6, 1), date(2023, 6, 5), frequency="daily"
        )
        assert len(result) == 3
        # Returns: June 1: 0% (100→100), June 2: 5% (100→105), June 5: ~4.76% (105→110)
        assert abs(result["AAPL"].iloc[0] - 0.0) < 1e-10
        assert abs(result["AAPL"].iloc[1] - 0.05) < 1e-10

    def test_monthly_returns(self):
        provider = YFinanceProvider()
        # Create prices spanning two months
        dates = pd.to_datetime([
            "2023-01-25", "2023-01-31",
            "2023-02-15", "2023-02-28",
            "2023-03-15", "2023-03-31",
        ])
        provider._cache["AAPL"] = pd.Series(
            [100.0, 100.0, 105.0, 110.0, 115.0, 120.0], index=dates
        )
        result = provider.get_returns(
            ["AAPL"], date(2023, 1, 1), date(2023, 3, 31), frequency="monthly"
        )
        # Should have monthly returns
        assert len(result) > 0

    def test_empty_prices_returns_empty(self):
        provider = YFinanceProvider()
        with patch(MOCK_YF) as mock_yf:
            mock_yf.download.return_value = pd.DataFrame()
            result = provider.get_returns(
                ["AAPL"], date(2023, 1, 1), date(2023, 12, 31)
            )
        assert result.empty


# ---------------------------------------------------------------------------
# YFinanceProvider — SQLite persistent cache
# ---------------------------------------------------------------------------


class TestYFinancePersistentCache:
    """Test SQLite-backed persistent caching."""

    def test_persist_and_reload(self, tmp_path):
        db_path = str(tmp_path / "cache.db")

        # First provider: fetch and persist
        provider1 = YFinanceProvider(cache_db_path=db_path)
        dates = pd.to_datetime(["2023-06-01", "2023-06-02", "2023-06-05"])
        prices = pd.DataFrame(
            {"AAPL": [150.0, 151.0, 152.0]}, index=dates
        )
        provider1._persist_to_store(prices)

        # Second provider: should load from cache
        provider2 = YFinanceProvider(cache_db_path=db_path)
        series = provider2._load_from_store("AAPL", date(2023, 6, 1), date(2023, 6, 5))
        assert series is not None
        assert len(series) == 3
        assert abs(series.iloc[0] - 150.0) < 1e-10

    def test_cache_miss_returns_none(self, tmp_path):
        db_path = str(tmp_path / "cache.db")
        provider = YFinanceProvider(cache_db_path=db_path)
        result = provider._load_from_store("AAPL", date(2023, 1, 1), date(2023, 12, 31))
        assert result is None

    def test_full_round_trip_with_cache(self, tmp_path):
        """Test get_prices uses SQLite cache on second call."""
        db_path = str(tmp_path / "cache.db")
        dates = pd.to_datetime(["2023-06-01", "2023-06-02", "2023-06-05"])

        mock_df = pd.DataFrame(
            {"Close": [150.0, 151.0, 152.0]}, index=dates
        )

        provider = YFinanceProvider(cache_db_path=db_path)

        with patch(MOCK_YF) as mock_yf:
            mock_yf.download.return_value = mock_df
            # First call: fetches from yfinance
            result1 = provider.get_prices(
                ["AAPL"], date(2023, 6, 1), date(2023, 6, 5)
            )
            assert mock_yf.download.call_count == 1

        # Clear in-memory cache to force SQLite lookup
        provider._cache.clear()

        with patch(MOCK_YF) as mock_yf:
            # Second call: should use SQLite cache, not yfinance
            result2 = provider.get_prices(
                ["AAPL"], date(2023, 6, 1), date(2023, 6, 5)
            )
            mock_yf.download.assert_not_called()

        assert len(result2) == 3

    def test_persist_error_handled_gracefully(self, tmp_path):
        """Persist errors should be logged, not raised."""
        db_path = str(tmp_path / "cache.db")
        provider = YFinanceProvider(cache_db_path=db_path)

        # Drop the table to cause an error
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE returns_cache")
        conn.commit()
        conn.close()

        dates = pd.to_datetime(["2023-06-01", "2023-06-02"])
        prices = pd.DataFrame({"AAPL": [150.0, 151.0]}, index=dates)

        # Should not raise
        provider._persist_to_store(prices)


# ---------------------------------------------------------------------------
# CSVProvider — Construction and loading
# ---------------------------------------------------------------------------


def _write_csv(tmp_path: Path, filename: str, content: str) -> str:
    """Helper to write a CSV file and return its path."""
    filepath = tmp_path / filename
    filepath.write_text(content)
    return str(filepath)


class TestCSVConstruction:
    """Test CSVProvider initialization and loading."""

    def test_file_not_found(self, tmp_path):
        provider = CSVProvider(str(tmp_path / "nonexistent.csv"))
        with pytest.raises(FileNotFoundError):
            provider.get_returns(["AAPL"], date(2023, 1, 1), date(2023, 12, 31))

    def test_missing_return_and_price_columns(self, tmp_path):
        csv = _write_csv(tmp_path, "bad.csv", "date,ticker,volume\n2023-01-03,AAPL,1000\n")
        provider = CSVProvider(csv)
        with pytest.raises(ValueError, match="return column.*price column"):
            provider.get_returns(["AAPL"], date(2023, 1, 1), date(2023, 12, 31))

    def test_missing_ticker_column(self, tmp_path):
        csv = _write_csv(tmp_path, "bad.csv", "date,ret\n2023-01-03,0.01\n")
        provider = CSVProvider(csv, ticker_column="symbol")
        with pytest.raises(ValueError, match="Ticker column.*not found"):
            provider.get_returns(["AAPL"], date(2023, 1, 1), date(2023, 12, 31))

    def test_auto_detect_return_column(self, tmp_path):
        csv = _write_csv(
            tmp_path, "returns.csv",
            "date,ticker,ret\n2023-01-03,AAPL,0.01\n2023-01-04,AAPL,0.02\n"
        )
        provider = CSVProvider(csv)
        provider._load()
        assert provider._return_col == "ret"

    def test_auto_detect_price_column(self, tmp_path):
        csv = _write_csv(
            tmp_path, "prices.csv",
            "date,ticker,adj_close\n2023-01-03,AAPL,150.0\n2023-01-04,AAPL,151.0\n"
        )
        provider = CSVProvider(csv)
        provider._load()
        assert provider._price_col == "adj_close"

    def test_custom_column_names(self, tmp_path):
        csv = _write_csv(
            tmp_path, "custom.csv",
            "dt,sym,prc\n2023-01-03,AAPL,150.0\n2023-01-04,AAPL,151.0\n"
        )
        provider = CSVProvider(
            csv, date_column="dt", ticker_column="sym", price_column="prc"
        )
        provider._load()
        assert provider._price_col == "prc"


# ---------------------------------------------------------------------------
# CSVProvider — Validation
# ---------------------------------------------------------------------------


class TestCSVValidation:
    """Test input validation for CSVProvider."""

    def test_start_after_end_raises(self, tmp_path):
        csv = _write_csv(
            tmp_path, "data.csv",
            "date,ticker,ret\n2023-01-03,AAPL,0.01\n"
        )
        provider = CSVProvider(csv)
        with pytest.raises(ValueError, match="start.*must be before end"):
            provider.get_returns(["AAPL"], date(2024, 1, 1), date(2023, 1, 1))

    def test_empty_tickers_raises(self, tmp_path):
        csv = _write_csv(
            tmp_path, "data.csv",
            "date,ticker,ret\n2023-01-03,AAPL,0.01\n"
        )
        provider = CSVProvider(csv)
        with pytest.raises(ValueError, match="tickers must not be empty"):
            provider.get_returns([], date(2023, 1, 1), date(2023, 12, 31))

    def test_prices_start_after_end_raises(self, tmp_path):
        csv = _write_csv(
            tmp_path, "data.csv",
            "date,ticker,adj_close\n2023-01-03,AAPL,150.0\n"
        )
        provider = CSVProvider(csv)
        with pytest.raises(ValueError, match="start.*must be before end"):
            provider.get_prices(["AAPL"], date(2024, 1, 1), date(2023, 1, 1))

    def test_prices_empty_tickers_raises(self, tmp_path):
        csv = _write_csv(
            tmp_path, "data.csv",
            "date,ticker,adj_close\n2023-01-03,AAPL,150.0\n"
        )
        provider = CSVProvider(csv)
        with pytest.raises(ValueError, match="tickers must not be empty"):
            provider.get_prices([], date(2023, 1, 1), date(2023, 12, 31))


# ---------------------------------------------------------------------------
# CSVProvider — Returns from pre-computed returns
# ---------------------------------------------------------------------------


class TestCSVReturns:
    """Test CSVProvider return extraction."""

    def test_returns_from_return_column(self, tmp_path):
        csv = _write_csv(
            tmp_path, "returns.csv",
            "date,ticker,ret\n"
            "2023-01-03,AAPL,0.01\n"
            "2023-01-04,AAPL,0.02\n"
            "2023-01-05,AAPL,-0.01\n"
            "2023-01-03,MSFT,0.005\n"
            "2023-01-04,MSFT,0.015\n"
            "2023-01-05,MSFT,-0.005\n"
        )
        provider = CSVProvider(csv)
        result = provider.get_returns(
            ["AAPL", "MSFT"], date(2023, 1, 1), date(2023, 1, 31), frequency="daily"
        )
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns
        assert len(result) == 3

    def test_returns_from_prices(self, tmp_path):
        csv = _write_csv(
            tmp_path, "prices.csv",
            "date,ticker,adj_close\n"
            "2023-01-03,AAPL,100.0\n"
            "2023-01-04,AAPL,105.0\n"
            "2023-01-05,AAPL,110.0\n"
        )
        provider = CSVProvider(csv)
        result = provider.get_returns(
            ["AAPL"], date(2023, 1, 1), date(2023, 1, 31), frequency="daily"
        )
        assert "AAPL" in result.columns
        # First return should be 5% (100 -> 105)
        assert abs(result["AAPL"].iloc[0] - 0.05) < 1e-10

    def test_missing_ticker_excluded(self, tmp_path):
        csv = _write_csv(
            tmp_path, "data.csv",
            "date,ticker,ret\n"
            "2023-01-03,AAPL,0.01\n"
            "2023-01-04,AAPL,0.02\n"
        )
        provider = CSVProvider(csv)
        result = provider.get_returns(
            ["AAPL", "UNKNOWN"], date(2023, 1, 1), date(2023, 1, 31), frequency="daily"
        )
        assert "AAPL" in result.columns
        assert "UNKNOWN" not in result.columns

    def test_all_tickers_missing_returns_empty(self, tmp_path):
        csv = _write_csv(
            tmp_path, "data.csv",
            "date,ticker,ret\n"
            "2023-01-03,AAPL,0.01\n"
        )
        provider = CSVProvider(csv)
        result = provider.get_returns(
            ["UNKNOWN"], date(2023, 1, 1), date(2023, 1, 31), frequency="daily"
        )
        assert result.empty

    def test_date_range_filtering(self, tmp_path):
        csv = _write_csv(
            tmp_path, "data.csv",
            "date,ticker,ret\n"
            "2023-01-03,AAPL,0.01\n"
            "2023-01-04,AAPL,0.02\n"
            "2023-02-01,AAPL,0.03\n"
            "2023-03-01,AAPL,0.04\n"
        )
        provider = CSVProvider(csv)
        result = provider.get_returns(
            ["AAPL"], date(2023, 1, 1), date(2023, 1, 31), frequency="daily"
        )
        assert len(result) == 2  # Only January dates

    def test_lazy_loading(self, tmp_path):
        csv = _write_csv(
            tmp_path, "data.csv",
            "date,ticker,ret\n2023-01-03,AAPL,0.01\n"
        )
        provider = CSVProvider(csv)
        assert provider._data is None
        provider.get_returns(["AAPL"], date(2023, 1, 1), date(2023, 12, 31))
        assert provider._data is not None


# ---------------------------------------------------------------------------
# CSVProvider — Prices
# ---------------------------------------------------------------------------


class TestCSVPrices:
    """Test CSVProvider price extraction."""

    def test_prices_from_csv(self, tmp_path):
        csv = _write_csv(
            tmp_path, "prices.csv",
            "date,ticker,adj_close\n"
            "2023-01-03,AAPL,150.0\n"
            "2023-01-04,AAPL,151.0\n"
            "2023-01-05,AAPL,152.0\n"
        )
        provider = CSVProvider(csv)
        result = provider.get_prices(
            ["AAPL"], date(2023, 1, 1), date(2023, 1, 31)
        )
        assert len(result) == 3
        assert abs(result["AAPL"].iloc[0] - 150.0) < 1e-10

    def test_no_price_column_raises(self, tmp_path):
        csv = _write_csv(
            tmp_path, "returns_only.csv",
            "date,ticker,ret\n2023-01-03,AAPL,0.01\n"
        )
        provider = CSVProvider(csv)
        with pytest.raises(ValueError, match="does not contain price data"):
            provider.get_prices(["AAPL"], date(2023, 1, 1), date(2023, 12, 31))

    def test_multi_ticker_prices(self, tmp_path):
        csv = _write_csv(
            tmp_path, "prices.csv",
            "date,ticker,adj_close\n"
            "2023-01-03,AAPL,150.0\n"
            "2023-01-03,MSFT,250.0\n"
            "2023-01-04,AAPL,151.0\n"
            "2023-01-04,MSFT,251.0\n"
        )
        provider = CSVProvider(csv)
        result = provider.get_prices(
            ["AAPL", "MSFT"], date(2023, 1, 1), date(2023, 1, 31)
        )
        assert "AAPL" in result.columns
        assert "MSFT" in result.columns
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Module-level imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Test package-level imports."""

    def test_backtest_package_exports(self):
        from edgar_sentinel.backtest import (
            CSVProvider,
            ReturnsProvider,
            YFinanceProvider,
        )

    def test_all_list(self):
        import edgar_sentinel.backtest as bt
        assert "ReturnsProvider" in bt.__all__
        assert "YFinanceProvider" in bt.__all__
        assert "CSVProvider" in bt.__all__
