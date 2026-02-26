"""Returns data providers: protocol, Yahoo Finance, and CSV import."""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@runtime_checkable
class ReturnsProvider(Protocol):
    """Interface for historical return data providers.

    Returns a DataFrame with:
    - Index: DatetimeIndex (trading days or period-ends)
    - Columns: ticker symbols
    - Values: simple returns (not log returns)

    The frequency parameter determines the return calculation period:
    - "daily": day-over-day returns
    - "monthly": month-end to month-end returns
    """

    def get_returns(
        self,
        tickers: list[str],
        start: date,
        end: date,
        frequency: str = "monthly",
    ) -> pd.DataFrame: ...

    def get_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame: ...


class YFinanceProvider:
    """Fetches historical prices from Yahoo Finance with local caching.

    Parameters
    ----------
    cache_db_path : str | None
        Path to SQLite database for persistent price caching.
        Uses the returns_cache table (created automatically).
        If None, only in-memory caching is used.
    request_buffer_days : int
        Extra days to fetch before ``start`` to ensure the first return
        can be computed. Default: 5 (covers weekends + holidays).
    """

    def __init__(
        self,
        cache_db_path: str | None = None,
        request_buffer_days: int = 5,
    ) -> None:
        self._cache_db_path = cache_db_path
        self._buffer_days = request_buffer_days
        self._cache: dict[str, pd.Series] = {}

        if self._cache_db_path is not None:
            self._ensure_cache_table()

    def _ensure_cache_table(self) -> None:
        """Create the returns_cache table if it doesn't exist."""
        conn = sqlite3.connect(self._cache_db_path)
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS returns_cache (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    adj_close REAL NOT NULL,
                    PRIMARY KEY(ticker, date)
                )"""
            )
            conn.commit()
        finally:
            conn.close()

    def get_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices for the given tickers and date range.

        Returns
        -------
        pd.DataFrame
            Index: DatetimeIndex (trading days)
            Columns: ticker symbols
            Values: adjusted close prices (float)
            Missing tickers are excluded with a warning log.

        Raises
        ------
        ValueError
            If ``start >= end`` or ``tickers`` is empty.
        """
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")
        if not tickers:
            raise ValueError("tickers must not be empty")

        buffered_start = start - timedelta(days=self._buffer_days)

        result = pd.DataFrame()
        uncached: list[str] = []

        for ticker in tickers:
            if ticker in self._cache:
                series = self._cache[ticker]
                mask = (series.index.date >= buffered_start) & (
                    series.index.date <= end
                )
                if mask.any() and series.index.date.max() >= end:
                    result[ticker] = series[mask]
                    continue
            uncached.append(ticker)

        # Try loading from persistent cache
        if uncached and self._cache_db_path is not None:
            still_uncached: list[str] = []
            for ticker in uncached:
                cached_series = self._load_from_store(
                    ticker, buffered_start, end
                )
                if cached_series is not None and not cached_series.empty:
                    if cached_series.index.date.max() >= end:
                        self._cache[ticker] = cached_series
                        result[ticker] = cached_series
                        continue
                still_uncached.append(ticker)
            uncached = still_uncached

        if uncached:
            result = self._fetch_and_cache(uncached, buffered_start, end, result)

        return result

    def get_returns(
        self,
        tickers: list[str],
        start: date,
        end: date,
        frequency: str = "monthly",
    ) -> pd.DataFrame:
        """Compute simple returns from adjusted close prices.

        Parameters
        ----------
        frequency : str
            "daily" for day-over-day, "monthly" for month-end returns.

        Returns
        -------
        pd.DataFrame
            Index: DatetimeIndex
            Columns: ticker symbols
            Values: simple returns (e.g., 0.05 for 5% gain)
        """
        prices = self.get_prices(tickers, start, end)

        if prices.empty:
            return prices

        if frequency == "monthly":
            monthly_prices = prices.resample("ME").last()
            returns = monthly_prices.pct_change().dropna(how="all")
        elif frequency == "daily":
            returns = prices.pct_change().dropna(how="all")
        else:
            raise ValueError(
                f"Unsupported frequency: {frequency}. Use 'daily' or 'monthly'."
            )

        # Trim to requested date range (buffer days excluded)
        returns = returns[returns.index.date >= start]
        return returns

    def _fetch_and_cache(
        self,
        tickers: list[str],
        start: date,
        end: date,
        existing: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fetch from yfinance, merge with existing, update cache."""
        logger.info("Fetching prices for %d tickers from yfinance", len(tickers))

        try:
            raw = yf.download(
                tickers=tickers,
                start=str(start),
                end=str(end),
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            logger.error("yfinance download failed: %s", e)
            return existing

        if raw.empty:
            logger.warning("yfinance returned empty DataFrame")
            return existing

        # Handle single-ticker vs multi-ticker DataFrame structure
        if len(tickers) == 1:
            if "Close" in raw.columns:
                series = raw["Close"].dropna()
                if not series.empty:
                    self._cache[tickers[0]] = series
                    existing[tickers[0]] = series
                else:
                    logger.warning("No data for %s, excluding", tickers[0])
        else:
            for ticker in tickers:
                try:
                    if ("Close", ticker) in raw.columns:
                        series = raw[("Close", ticker)].dropna()
                    elif "Close" in raw.columns and ticker in raw["Close"].columns:
                        series = raw["Close"][ticker].dropna()
                    else:
                        logger.warning(
                            "Ticker %s not found in yfinance response", ticker
                        )
                        continue

                    if series.empty:
                        logger.warning("No data for %s, excluding", ticker)
                        continue
                    self._cache[ticker] = series
                    existing[ticker] = series
                except KeyError:
                    logger.warning(
                        "Ticker %s not found in yfinance response", ticker
                    )

        # Persist to storage if available
        if self._cache_db_path is not None:
            self._persist_to_store(existing)

        return existing

    def _persist_to_store(self, prices: pd.DataFrame) -> None:
        """Write fetched prices to the SQLite cache."""
        conn = sqlite3.connect(self._cache_db_path)
        try:
            for ticker in prices.columns:
                series = prices[ticker].dropna()
                rows = [
                    (ticker, idx.strftime("%Y-%m-%d"), float(val))
                    for idx, val in series.items()
                ]
                if rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO returns_cache (ticker, date, adj_close) "
                        "VALUES (?, ?, ?)",
                        rows,
                    )
            conn.commit()
        except Exception as e:
            logger.error("Failed to persist prices to cache: %s", e)
        finally:
            conn.close()

    def _load_from_store(
        self, ticker: str, start: date, end: date
    ) -> pd.Series | None:
        """Load cached prices from SQLite."""
        try:
            conn = sqlite3.connect(self._cache_db_path)
            try:
                cursor = conn.execute(
                    "SELECT date, adj_close FROM returns_cache "
                    "WHERE ticker = ? AND date >= ? AND date <= ? "
                    "ORDER BY date",
                    (ticker, start.isoformat(), end.isoformat()),
                )
                rows = cursor.fetchall()
            finally:
                conn.close()

            if not rows:
                return None

            dates = pd.to_datetime([r[0] for r in rows])
            values = [r[1] for r in rows]
            return pd.Series(values, index=dates, name=ticker)
        except Exception as e:
            logger.warning("Failed to load cached prices for %s: %s", ticker, e)
            return None


class CSVProvider:
    """Returns provider for institutional data exports (CRSP, Compustat).

    Expected CSV format:
    - Column "date": YYYY-MM-DD
    - Column "ticker": stock symbol
    - Column "ret" or "return": simple return
    - Column "price" or "adj_close": adjusted close price (optional)

    At least one of (ret/return) or (price/adj_close) must be present.
    If only prices are provided, returns are computed via pct_change().

    Parameters
    ----------
    filepath : str
        Path to CSV file.
    date_column : str
        Name of the date column. Default: "date".
    ticker_column : str
        Name of the ticker column. Default: "ticker".
    return_column : str | None
        Name of the return column. Default: auto-detected.
    price_column : str | None
        Name of the price column. Default: auto-detected.
    """

    def __init__(
        self,
        filepath: str,
        date_column: str = "date",
        ticker_column: str = "ticker",
        return_column: str | None = None,
        price_column: str | None = None,
    ) -> None:
        self._filepath = filepath
        self._date_col = date_column
        self._ticker_col = ticker_column
        self._return_col = return_column
        self._price_col = price_column
        self._data: pd.DataFrame | None = None

    def _load(self) -> None:
        """Load and validate the CSV file."""
        path = Path(self._filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._filepath}")

        df = pd.read_csv(self._filepath, parse_dates=[self._date_col])

        if self._ticker_col not in df.columns:
            raise ValueError(
                f"Ticker column '{self._ticker_col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

        # Auto-detect return column
        if self._return_col is None:
            for candidate in ("ret", "return", "returns"):
                if candidate in df.columns:
                    self._return_col = candidate
                    break

        # Auto-detect price column
        if self._price_col is None:
            for candidate in ("price", "adj_close", "adjclose"):
                if candidate in df.columns:
                    self._price_col = candidate
                    break

        if self._return_col is None and self._price_col is None:
            raise ValueError(
                "CSV must have a return column (ret/return/returns) "
                "or price column (price/adj_close/adjclose)"
            )

        self._data = df

    def get_returns(
        self,
        tickers: list[str],
        start: date,
        end: date,
        frequency: str = "monthly",
    ) -> pd.DataFrame:
        """Extract returns for the specified tickers and date range."""
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")
        if not tickers:
            raise ValueError("tickers must not be empty")

        if self._data is None:
            self._load()

        df = self._data

        if self._return_col is not None:
            # Use pre-computed returns
            pivot = df.pivot_table(
                index=self._date_col,
                columns=self._ticker_col,
                values=self._return_col,
            )
        else:
            # Compute returns from prices
            pivot = df.pivot_table(
                index=self._date_col,
                columns=self._ticker_col,
                values=self._price_col,
            )
            pivot = pivot.pct_change()

        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index()

        # Filter tickers
        available = [t for t in tickers if t in pivot.columns]
        missing = [t for t in tickers if t not in pivot.columns]
        if missing:
            logger.warning("Tickers not found in CSV: %s", missing)

        if not available:
            return pd.DataFrame()

        result = pivot[available]

        # Filter date range
        result = result[
            (result.index.date >= start) & (result.index.date <= end)
        ]

        # Resample if needed
        if frequency == "monthly" and self._return_col is not None:
            # Compound daily returns to monthly
            result = result.resample("ME").apply(
                lambda x: (1 + x).prod() - 1 if len(x) > 0 else float("nan")
            )
        elif frequency == "monthly" and self._price_col is not None:
            # Already computed pct_change, resample to monthly
            prices_pivot = df.pivot_table(
                index=self._date_col,
                columns=self._ticker_col,
                values=self._price_col,
            )
            prices_pivot.index = pd.to_datetime(prices_pivot.index)
            prices_pivot = prices_pivot.sort_index()
            monthly = prices_pivot[available].resample("ME").last()
            result = monthly.pct_change().dropna(how="all")
            result = result[
                (result.index.date >= start) & (result.index.date <= end)
            ]

        return result.dropna(how="all")

    def get_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Extract prices if available, otherwise raise ValueError."""
        if start >= end:
            raise ValueError(f"start ({start}) must be before end ({end})")
        if not tickers:
            raise ValueError("tickers must not be empty")

        if self._data is None:
            self._load()

        if self._price_col is None:
            raise ValueError("CSV does not contain price data")

        df = self._data
        pivot = df.pivot_table(
            index=self._date_col,
            columns=self._ticker_col,
            values=self._price_col,
        )
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index()

        available = [t for t in tickers if t in pivot.columns]
        missing = [t for t in tickers if t not in pivot.columns]
        if missing:
            logger.warning("Tickers not found in CSV: %s", missing)

        if not available:
            return pd.DataFrame()

        result = pivot[available]
        result = result[
            (result.index.date >= start) & (result.index.date <= end)
        ]
        return result
