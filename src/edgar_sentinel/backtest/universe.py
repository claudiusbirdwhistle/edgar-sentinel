"""Universe providers for survivorship-bias-free backtesting.

A ``UniverseProvider`` answers the question: *which tickers were valid
members of the investment universe on a given date?*  Using a
time-varying provider instead of a static ticker list prevents
survivorship bias — the tendency to backtest only companies that
*survived* to the present day, which artificially inflates performance.

Usage
-----
Static (backward-compatible)::

    provider = StaticUniverseProvider(["AAPL", "MSFT", "GOOG"])

Historical S&P 500 (survivorship-bias-free)::

    provider = Sp500HistoricalProvider()
    # Downloads sp500_ticker_start_end.csv from GitHub on first use and
    # caches it in ~/.edgar_sentinel/data/
    tickers = provider.get_tickers(date(2018, 3, 31))

Data source for Sp500HistoricalProvider
-----------------------------------------
https://github.com/fja05680/sp500 (fja05680/sp500)

The CSV ``sp500_ticker_start_end.csv`` lists every S&P 500 constituent
with its entry and exit dates.  A blank ``end_date`` means the ticker
is still in the index.
"""

from __future__ import annotations

import csv
import logging
import urllib.request
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Default directory for downloaded/cached data
_DEFAULT_CACHE_DIR = Path.home() / ".edgar_sentinel" / "data"


@runtime_checkable
class UniverseProvider(Protocol):
    """Protocol for providing the valid ticker universe at a given date.

    Any class that implements ``get_tickers(rebalance_date: date) -> list[str]``
    satisfies this protocol (structural typing — no inheritance required).
    """

    def get_tickers(self, rebalance_date: date) -> list[str]:
        """Return the list of valid tickers for a given rebalance date.

        Parameters
        ----------
        rebalance_date : date
            The date on which the portfolio is being rebalanced.

        Returns
        -------
        list[str]
            Ticker symbols (uppercase) that belong to the universe on this date.
        """
        ...  # pragma: no cover


class StaticUniverseProvider:
    """Returns the same fixed list of tickers for all rebalance dates.

    This is the backward-compatible default.  Every invocation of
    ``get_tickers`` returns a copy of the list passed to the constructor,
    regardless of the requested date.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to include in the universe.
    """

    def __init__(self, tickers: list[str]) -> None:
        self._tickers: list[str] = list(tickers)

    def get_tickers(self, rebalance_date: date) -> list[str]:  # noqa: ARG002
        return list(self._tickers)


class Sp500HistoricalProvider:
    """Provides point-in-time S&P 500 constituent data.

    On first use, downloads ``sp500_ticker_start_end.csv`` from the
    `fja05680/sp500 <https://github.com/fja05680/sp500>`_ GitHub
    repository and caches it locally.  Subsequent calls read from the
    in-memory cache.

    The CSV format is::

        ticker,start_date,end_date
        AAPL,1997-01-02,
        GE,1996-01-02,2018-11-19
        AAL,1996-01-02,1997-01-15
        AAL,2015-03-23,2024-09-23

    A blank ``end_date`` means the ticker is still in the index.

    Parameters
    ----------
    cache_dir : Path | None
        Directory where the downloaded CSV is cached.  Defaults to
        ``~/.edgar_sentinel/data/``.  Pass a custom path in tests.

    Examples
    --------
    >>> from datetime import date
    >>> provider = Sp500HistoricalProvider()
    >>> tickers_2018 = provider.get_tickers(date(2018, 6, 30))
    """

    CSV_URL: str = (
        "https://raw.githubusercontent.com/fja05680/sp500/master/"
        "sp500_ticker_start_end.csv"
    )
    CACHE_FILENAME: str = "sp500_ticker_start_end.csv"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir: Path = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_path: Path = self._cache_dir / self.CACHE_FILENAME
        self._records: list[dict] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_tickers(self, rebalance_date: date) -> list[str]:
        """Return S&P 500 members on the given date.

        A ticker is included if and only if::

            start_date <= rebalance_date <= end_date

        where a blank ``end_date`` is treated as *still in the index*.
        """
        self._ensure_data()
        assert self._records is not None  # noqa: S101  (guaranteed by _ensure_data)
        return [
            r["ticker"]
            for r in self._records
            if r["start_date"] <= rebalance_date
            and (r["end_date"] is None or r["end_date"] >= rebalance_date)
        ]

    def refresh(self) -> None:
        """Force re-download of the constituent data from GitHub.

        Useful when the cached file is stale (e.g., quarterly refresh).
        """
        self._records = None
        if self._cache_path.exists():
            self._cache_path.unlink()
        self._download()
        self._records = self._parse_csv(self._cache_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_data(self) -> None:
        """Load data into memory, downloading if the cache file is absent."""
        if self._records is not None:
            return
        if not self._cache_path.exists():
            self._download()
        self._records = self._parse_csv(self._cache_path)

    def _download(self) -> None:
        """Download the CSV from GitHub and save to the cache directory."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading S&P 500 constituent data from %s → %s",
            self.CSV_URL,
            self._cache_path,
        )
        urllib.request.urlretrieve(self.CSV_URL, self._cache_path)  # noqa: S310
        logger.info("Constituent data cached at %s", self._cache_path)

    @staticmethod
    def _parse_csv(path: Path) -> list[dict]:
        """Parse ``sp500_ticker_start_end.csv`` into a list of records.

        Each record is a dict with keys:
        - ``ticker`` (str, uppercase)
        - ``start_date`` (date)
        - ``end_date`` (date | None)

        Rows that are missing a ticker, missing a start_date, or contain
        unparsable dates are skipped with a warning.
        """
        records: list[dict] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get("ticker", "").strip().upper()
                start_str = row.get("start_date", "").strip()
                end_str = row.get("end_date", "").strip()

                if not ticker or not start_str:
                    continue

                try:
                    start_dt = date.fromisoformat(start_str)
                    end_dt = date.fromisoformat(end_str) if end_str else None
                except ValueError:
                    logger.warning("Skipping malformed row: %s", row)
                    continue

                records.append(
                    {
                        "ticker": ticker,
                        "start_date": start_dt,
                        "end_date": end_dt,
                    }
                )
        return records
