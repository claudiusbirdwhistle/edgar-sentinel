"""CSV price adapter — imports price data from CSV files.

Demonstrates the adapter pattern: any raw data format can be ingested
by writing an adapter that produces PriceBar records.
"""

from __future__ import annotations

import csv
import logging
from datetime import date
from pathlib import Path
from typing import Any

from edgar_sentinel.prices.models import PriceBar

logger = logging.getLogger(__name__)

# Common column name mappings for auto-detection
_DATE_ALIASES = {"date", "Date", "DATE", "timestamp", "Timestamp"}
_OPEN_ALIASES = {"open", "Open", "OPEN"}
_HIGH_ALIASES = {"high", "High", "HIGH"}
_LOW_ALIASES = {"low", "Low", "LOW"}
_CLOSE_ALIASES = {"close", "Close", "CLOSE"}
_VOLUME_ALIASES = {"volume", "Volume", "VOLUME", "vol", "Vol"}
_ADJ_CLOSE_ALIASES = {"adj_close", "Adj Close", "adjclose", "adjusted_close", "Adj_Close"}


def _find_column(headers: list[str], aliases: set[str]) -> str | None:
    """Find the first header that matches any alias."""
    for h in headers:
        if h in aliases:
            return h
    return None


class CSVPriceAdapter:
    """Transforms CSV file data into PriceBar records.

    Supports flexible column mapping. If column names aren't specified,
    auto-detects common naming conventions.

    Parameters
    ----------
    date_col : str | None
        Name of the date column. Auto-detected if None.
    open_col : str | None
        Name of the open price column. Auto-detected if None.
    high_col : str | None
        Name of the high price column. Auto-detected if None.
    low_col : str | None
        Name of the low price column. Auto-detected if None.
    close_col : str | None
        Name of the close price column. Auto-detected if None.
    volume_col : str | None
        Name of the volume column. Auto-detected if None.
    adj_close_col : str | None
        Name of the adjusted close column. Auto-detected if None.
    date_format : str
        strptime format for date parsing. Default: ISO-8601.
    """

    def __init__(
        self,
        date_col: str | None = None,
        open_col: str | None = None,
        high_col: str | None = None,
        low_col: str | None = None,
        close_col: str | None = None,
        volume_col: str | None = None,
        adj_close_col: str | None = None,
        date_format: str = "%Y-%m-%d",
    ) -> None:
        self._date_col = date_col
        self._open_col = open_col
        self._high_col = high_col
        self._low_col = low_col
        self._close_col = close_col
        self._volume_col = volume_col
        self._adj_close_col = adj_close_col
        self._date_format = date_format

    def _resolve_columns(self, headers: list[str]) -> dict[str, str | None]:
        """Resolve column names from headers, using aliases for auto-detection."""
        return {
            "date": self._date_col or _find_column(headers, _DATE_ALIASES),
            "open": self._open_col or _find_column(headers, _OPEN_ALIASES),
            "high": self._high_col or _find_column(headers, _HIGH_ALIASES),
            "low": self._low_col or _find_column(headers, _LOW_ALIASES),
            "close": self._close_col or _find_column(headers, _CLOSE_ALIASES),
            "volume": self._volume_col or _find_column(headers, _VOLUME_ALIASES),
            "adj_close": self._adj_close_col or _find_column(headers, _ADJ_CLOSE_ALIASES),
        }

    def adapt(self, raw_data: Any, ticker: str) -> list[PriceBar]:
        """Parse CSV rows (list of dicts) into PriceBar list.

        Parameters
        ----------
        raw_data : list[dict[str, str]]
            Rows from csv.DictReader. Each dict maps column name → value.
        ticker : str
            The ticker symbol for all bars.

        Returns
        -------
        list[PriceBar]
            Sorted by date ascending.
        """
        if not raw_data:
            return []

        headers = list(raw_data[0].keys())
        cols = self._resolve_columns(headers)

        if cols["date"] is None:
            raise ValueError(f"Cannot find date column in headers: {headers}")
        if cols["close"] is None:
            raise ValueError(f"Cannot find close column in headers: {headers}")

        bars: list[PriceBar] = []
        for row in raw_data:
            try:
                bar_date = date.fromisoformat(row[cols["date"]])
            except (ValueError, KeyError):
                try:
                    from datetime import datetime as dt
                    bar_date = dt.strptime(row[cols["date"]], self._date_format).date()
                except (ValueError, KeyError):
                    logger.warning("Skipping row with unparseable date: %s", row.get(cols["date"]))
                    continue

            close_val = float(row[cols["close"]])

            bars.append(
                PriceBar(
                    ticker=ticker,
                    date=bar_date,
                    open=float(row[cols["open"]]) if cols["open"] and row.get(cols["open"]) else close_val,
                    high=float(row[cols["high"]]) if cols["high"] and row.get(cols["high"]) else close_val,
                    low=float(row[cols["low"]]) if cols["low"] and row.get(cols["low"]) else close_val,
                    close=close_val,
                    volume=int(float(row[cols["volume"]])) if cols["volume"] and row.get(cols["volume"]) else 0,
                    adj_close=float(row[cols["adj_close"]]) if cols["adj_close"] and row.get(cols["adj_close"]) else None,
                    source="csv",
                )
            )

        return sorted(bars, key=lambda b: b.date)


def load_csv_prices(filepath: str, ticker: str, **adapter_kwargs: Any) -> list[PriceBar]:
    """Convenience function: load price bars from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    ticker : str
        Ticker symbol for the data.
    **adapter_kwargs
        Passed to CSVPriceAdapter constructor.

    Returns
    -------
    list[PriceBar]
        Parsed and sorted price bars.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    adapter = CSVPriceAdapter(**adapter_kwargs)
    return adapter.adapt(rows, ticker)
