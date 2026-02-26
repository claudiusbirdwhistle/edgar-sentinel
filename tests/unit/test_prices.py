"""Unit tests for the prices module."""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from edgar_sentinel.prices.models import PriceBar, PriceInterval, PriceMeta, PriceQuery
from edgar_sentinel.prices.provider import PriceAdapter, PriceProvider
from edgar_sentinel.prices.yahoo import YahooFinanceAdapter
from edgar_sentinel.prices.csv_adapter import CSVPriceAdapter, load_csv_prices
from edgar_sentinel.prices.store import SqlitePriceStore


# --- PriceBar model tests ---


class TestPriceBar:
    def test_create_valid(self):
        bar = PriceBar(
            ticker="AAPL",
            date=date(2024, 1, 15),
            open=185.0,
            high=187.5,
            low=184.0,
            close=186.5,
            volume=50000000,
            adj_close=186.5,
            source="test",
        )
        assert bar.ticker == "AAPL"
        assert bar.close == 186.5
        assert bar.source == "test"

    def test_high_must_be_gte_low(self):
        with pytest.raises(ValueError, match="high.*must be >= low"):
            PriceBar(
                ticker="AAPL",
                date=date(2024, 1, 15),
                open=185.0,
                high=183.0,  # < low
                low=184.0,
                close=186.5,
                volume=50000000,
            )

    def test_negative_volume_rejected(self):
        with pytest.raises(ValueError, match="volume must be >= 0"):
            PriceBar(
                ticker="AAPL",
                date=date(2024, 1, 15),
                open=185.0,
                high=187.5,
                low=184.0,
                close=186.5,
                volume=-1,
            )

    def test_adjusted_close_fallback(self):
        bar = PriceBar(
            ticker="AAPL",
            date=date(2024, 1, 15),
            open=185.0,
            high=187.5,
            low=184.0,
            close=186.5,
            volume=50000000,
        )
        assert bar.adj_close is None
        assert bar.adjusted_close == 186.5

    def test_adjusted_close_explicit(self):
        bar = PriceBar(
            ticker="AAPL",
            date=date(2024, 1, 15),
            open=185.0,
            high=187.5,
            low=184.0,
            close=186.5,
            volume=50000000,
            adj_close=185.0,
        )
        assert bar.adjusted_close == 185.0

    def test_default_source(self):
        bar = PriceBar(
            ticker="AAPL",
            date=date(2024, 1, 15),
            open=185.0,
            high=187.5,
            low=184.0,
            close=186.5,
            volume=50000000,
        )
        assert bar.source == "unknown"

    def test_frozen(self):
        bar = PriceBar(
            ticker="AAPL",
            date=date(2024, 1, 15),
            open=185.0,
            high=187.5,
            low=184.0,
            close=186.5,
            volume=50000000,
        )
        with pytest.raises(Exception):
            bar.close = 999.0  # type: ignore[misc]


class TestPriceQuery:
    def test_create_valid(self):
        q = PriceQuery(
            tickers=["AAPL", "MSFT"],
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
        )
        assert q.interval == PriceInterval.DAILY
        assert len(q.tickers) == 2

    def test_empty_tickers_rejected(self):
        with pytest.raises(ValueError, match="tickers must not be empty"):
            PriceQuery(
                tickers=[],
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
            )

    def test_end_must_be_after_start(self):
        with pytest.raises(ValueError, match="end.*must be after start"):
            PriceQuery(
                tickers=["AAPL"],
                start=date(2024, 12, 31),
                end=date(2024, 1, 1),
            )


# --- YahooFinanceAdapter tests ---


SAMPLE_YAHOO_RESPONSE = {
    "timestamp": [1705276200, 1705362600, 1705449000, 1705535400, 1705621800],
    "indicators": {
        "quote": [
            {
                "open": [185.09, 182.16, 186.09, 185.60, 184.55],
                "high": [185.60, 186.10, 188.00, 186.50, 185.40],
                "low": [182.00, 181.90, 185.00, 183.50, 183.00],
                "close": [183.63, 185.92, 187.44, 184.40, 183.96],
                "volume": [40477800, 49128400, 39631600, 42355900, 45113100],
            }
        ],
        "adjclose": [
            {
                "adjclose": [183.63, 185.92, 187.44, 184.40, 183.96],
            }
        ],
    },
    "meta": {
        "currency": "USD",
        "symbol": "AAPL",
        "exchangeName": "NMS",
        "fullExchangeName": "NasdaqGS",
        "instrumentType": "EQUITY",
    },
}


class TestYahooFinanceAdapter:
    def test_adapt_valid_response(self):
        adapter = YahooFinanceAdapter()
        bars = adapter.adapt(SAMPLE_YAHOO_RESPONSE, "AAPL")
        assert len(bars) == 5
        assert all(b.ticker == "AAPL" for b in bars)
        assert all(b.source == "yahoo_finance" for b in bars)
        # Sorted by date
        dates = [b.date for b in bars]
        assert dates == sorted(dates)

    def test_adapt_preserves_ohlcv(self):
        adapter = YahooFinanceAdapter()
        bars = adapter.adapt(SAMPLE_YAHOO_RESPONSE, "AAPL")
        first = bars[0]
        assert first.open == 185.09
        assert first.high == 185.60
        assert first.low == 182.00
        assert first.close == 183.63
        assert first.volume == 40477800
        assert first.adj_close == 183.63

    def test_adapt_empty_timestamps(self):
        adapter = YahooFinanceAdapter()
        bars = adapter.adapt({"timestamp": []}, "AAPL")
        assert bars == []

    def test_adapt_missing_timestamps(self):
        adapter = YahooFinanceAdapter()
        bars = adapter.adapt({}, "AAPL")
        assert bars == []

    def test_adapt_null_ohlc_values_skipped(self):
        data = {
            "timestamp": [1705276200, 1705362600],
            "indicators": {
                "quote": [
                    {
                        "open": [185.0, None],
                        "high": [186.0, None],
                        "low": [184.0, None],
                        "close": [185.5, None],
                        "volume": [1000, None],
                    }
                ],
            },
        }
        adapter = YahooFinanceAdapter()
        bars = adapter.adapt(data, "AAPL")
        assert len(bars) == 1

    def test_adapt_no_adjclose(self):
        data = {
            "timestamp": [1705276200],
            "indicators": {
                "quote": [
                    {
                        "open": [185.0],
                        "high": [186.0],
                        "low": [184.0],
                        "close": [185.5],
                        "volume": [1000],
                    }
                ],
            },
        }
        adapter = YahooFinanceAdapter()
        bars = adapter.adapt(data, "AAPL")
        assert len(bars) == 1
        assert bars[0].adj_close is None


# --- CSVPriceAdapter tests ---


class TestCSVPriceAdapter:
    def test_adapt_standard_columns(self):
        rows = [
            {"date": "2024-01-15", "open": "185.0", "high": "187.5", "low": "184.0", "close": "186.5", "volume": "50000000"},
            {"date": "2024-01-16", "open": "186.5", "high": "188.0", "low": "185.0", "close": "187.0", "volume": "48000000"},
        ]
        adapter = CSVPriceAdapter()
        bars = adapter.adapt(rows, "AAPL")
        assert len(bars) == 2
        assert bars[0].date == date(2024, 1, 15)
        assert bars[1].date == date(2024, 1, 16)
        assert all(b.source == "csv" for b in bars)

    def test_adapt_yahoo_finance_export_columns(self):
        """Yahoo Finance CSV downloads use 'Date', 'Open', 'Close', etc."""
        rows = [
            {"Date": "2024-01-15", "Open": "185.0", "High": "187.5", "Low": "184.0", "Close": "186.5", "Volume": "50000000", "Adj Close": "186.0"},
        ]
        adapter = CSVPriceAdapter()
        bars = adapter.adapt(rows, "AAPL")
        assert len(bars) == 1
        assert bars[0].adj_close == 186.0

    def test_adapt_close_only(self):
        """If only close is available, OHLC all default to close."""
        rows = [
            {"date": "2024-01-15", "close": "186.5"},
        ]
        adapter = CSVPriceAdapter()
        bars = adapter.adapt(rows, "AAPL")
        assert len(bars) == 1
        assert bars[0].open == 186.5
        assert bars[0].high == 186.5
        assert bars[0].low == 186.5

    def test_adapt_empty_data(self):
        adapter = CSVPriceAdapter()
        bars = adapter.adapt([], "AAPL")
        assert bars == []

    def test_missing_date_column_raises(self):
        rows = [{"price": "100.0"}]
        adapter = CSVPriceAdapter()
        with pytest.raises(ValueError, match="Cannot find date column"):
            adapter.adapt(rows, "AAPL")

    def test_missing_close_column_raises(self):
        rows = [{"date": "2024-01-15", "price": "100.0"}]
        adapter = CSVPriceAdapter()
        with pytest.raises(ValueError, match="Cannot find close column"):
            adapter.adapt(rows, "AAPL")

    def test_sorted_by_date(self):
        rows = [
            {"date": "2024-01-16", "close": "187.0"},
            {"date": "2024-01-15", "close": "186.5"},
        ]
        adapter = CSVPriceAdapter()
        bars = adapter.adapt(rows, "AAPL")
        assert bars[0].date < bars[1].date


class TestLoadCSVPrices:
    def test_load_from_file(self, tmp_path: Path):
        csv_file = tmp_path / "prices.csv"
        csv_file.write_text(
            "date,open,high,low,close,volume\n"
            "2024-01-15,185.0,187.5,184.0,186.5,50000000\n"
            "2024-01-16,186.5,188.0,185.0,187.0,48000000\n"
        )
        bars = load_csv_prices(str(csv_file), "AAPL")
        assert len(bars) == 2

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv_prices("/nonexistent/path.csv", "AAPL")


# --- SqlitePriceStore tests ---


@pytest.mark.asyncio
class TestSqlitePriceStore:
    async def test_store_and_retrieve(self, tmp_path: Path):
        store = SqlitePriceStore(str(tmp_path / "prices.db"))
        bars = [
            PriceBar(
                ticker="AAPL",
                date=date(2024, 1, 15),
                open=185.0,
                high=187.5,
                low=184.0,
                close=186.5,
                volume=50000000,
                source="test",
            ),
            PriceBar(
                ticker="AAPL",
                date=date(2024, 1, 16),
                open=186.5,
                high=188.0,
                low=185.0,
                close=187.0,
                volume=48000000,
                source="test",
            ),
        ]
        count = await store.store_bars(bars)
        assert count == 2

        retrieved = await store.get_bars("AAPL", date(2024, 1, 15), date(2024, 1, 16))
        assert len(retrieved) == 2
        assert retrieved[0].close == 186.5
        assert retrieved[1].close == 187.0

    async def test_store_empty(self, tmp_path: Path):
        store = SqlitePriceStore(str(tmp_path / "prices.db"))
        count = await store.store_bars([])
        assert count == 0

    async def test_upsert_semantics(self, tmp_path: Path):
        store = SqlitePriceStore(str(tmp_path / "prices.db"))
        bar_v1 = PriceBar(
            ticker="AAPL",
            date=date(2024, 1, 15),
            open=185.0,
            high=187.5,
            low=184.0,
            close=186.5,
            volume=50000000,
            source="v1",
        )
        bar_v2 = PriceBar(
            ticker="AAPL",
            date=date(2024, 1, 15),
            open=185.0,
            high=187.5,
            low=184.0,
            close=999.0,  # Updated close
            volume=50000000,
            source="v2",
        )
        await store.store_bars([bar_v1])
        await store.store_bars([bar_v2])

        retrieved = await store.get_bars("AAPL", date(2024, 1, 15), date(2024, 1, 15))
        assert len(retrieved) == 1
        assert retrieved[0].close == 999.0
        assert retrieved[0].source == "v2"

    async def test_has_data(self, tmp_path: Path):
        store = SqlitePriceStore(str(tmp_path / "prices.db"))
        assert not await store.has_data("AAPL", date(2024, 1, 15), date(2024, 1, 16))

        await store.store_bars([
            PriceBar(
                ticker="AAPL",
                date=date(2024, 1, 15),
                open=185.0,
                high=187.5,
                low=184.0,
                close=186.5,
                volume=50000000,
            ),
        ])
        assert await store.has_data("AAPL", date(2024, 1, 15), date(2024, 1, 16))

    async def test_get_tickers(self, tmp_path: Path):
        store = SqlitePriceStore(str(tmp_path / "prices.db"))
        assert await store.get_tickers() == []

        await store.store_bars([
            PriceBar(
                ticker="AAPL",
                date=date(2024, 1, 15),
                open=185.0,
                high=187.5,
                low=184.0,
                close=186.5,
                volume=50000000,
            ),
            PriceBar(
                ticker="MSFT",
                date=date(2024, 1, 15),
                open=370.0,
                high=375.0,
                low=368.0,
                close=372.0,
                volume=30000000,
            ),
        ])
        tickers = await store.get_tickers()
        assert tickers == ["AAPL", "MSFT"]

    async def test_filter_by_date_range(self, tmp_path: Path):
        store = SqlitePriceStore(str(tmp_path / "prices.db"))
        bars = [
            PriceBar(ticker="AAPL", date=date(2024, 1, 14), open=184.0, high=186.0, low=183.0, close=185.0, volume=100),
            PriceBar(ticker="AAPL", date=date(2024, 1, 15), open=185.0, high=187.0, low=184.0, close=186.0, volume=100),
            PriceBar(ticker="AAPL", date=date(2024, 1, 16), open=186.0, high=188.0, low=185.0, close=187.0, volume=100),
        ]
        await store.store_bars(bars)

        retrieved = await store.get_bars("AAPL", date(2024, 1, 15), date(2024, 1, 15))
        assert len(retrieved) == 1
        assert retrieved[0].date == date(2024, 1, 15)

    async def test_multiple_tickers_isolated(self, tmp_path: Path):
        store = SqlitePriceStore(str(tmp_path / "prices.db"))
        await store.store_bars([
            PriceBar(ticker="AAPL", date=date(2024, 1, 15), open=185.0, high=187.0, low=184.0, close=186.0, volume=100),
            PriceBar(ticker="MSFT", date=date(2024, 1, 15), open=370.0, high=375.0, low=368.0, close=372.0, volume=200),
        ])

        aapl = await store.get_bars("AAPL", date(2024, 1, 15), date(2024, 1, 15))
        msft = await store.get_bars("MSFT", date(2024, 1, 15), date(2024, 1, 15))
        assert len(aapl) == 1
        assert len(msft) == 1
        assert aapl[0].close == 186.0
        assert msft[0].close == 372.0


# --- Protocol conformance tests ---


class TestProtocolConformance:
    def test_yahoo_adapter_is_price_adapter(self):
        adapter = YahooFinanceAdapter()
        assert isinstance(adapter, PriceAdapter)

    def test_csv_adapter_is_price_adapter(self):
        adapter = CSVPriceAdapter()
        assert isinstance(adapter, PriceAdapter)
