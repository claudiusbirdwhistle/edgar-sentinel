"""SQLite-backed price storage with caching.

Provides persistent storage for price bars fetched from any provider.
Uses aiosqlite for async SQLite access (already a project dependency).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

import aiosqlite

from edgar_sentinel.prices.models import PriceBar

logger = logging.getLogger(__name__)


@runtime_checkable
class PriceStore(Protocol):
    """Protocol for price data persistence backends."""

    async def store_bars(self, bars: list[PriceBar]) -> int:
        """Store price bars. Returns count of rows upserted."""
        ...

    async def get_bars(
        self, ticker: str, start: date, end: date
    ) -> list[PriceBar]:
        """Retrieve stored bars for a ticker and date range."""
        ...

    async def has_data(
        self, ticker: str, start: date, end: date
    ) -> bool:
        """Check if complete data exists for the range."""
        ...

    async def get_tickers(self) -> list[str]:
        """Return all tickers with stored data."""
        ...


class SqlitePriceStore:
    """SQLite-backed implementation of PriceStore.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
        Created automatically if it doesn't exist.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._initialized = False

    async def _ensure_table(self) -> None:
        """Create the prices table if it doesn't exist."""
        if self._initialized:
            return

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """CREATE TABLE IF NOT EXISTS prices (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    adj_close REAL,
                    source TEXT NOT NULL DEFAULT 'unknown',
                    PRIMARY KEY (ticker, date)
                )"""
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices (ticker)"
            )
            await db.commit()
        self._initialized = True

    async def store_bars(self, bars: list[PriceBar]) -> int:
        """Store price bars with upsert semantics (replace on conflict)."""
        if not bars:
            return 0

        await self._ensure_table()

        async with aiosqlite.connect(self._db_path) as db:
            rows = [
                (
                    bar.ticker,
                    bar.date.isoformat(),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    bar.adj_close,
                    bar.source,
                )
                for bar in bars
            ]
            await db.executemany(
                """INSERT OR REPLACE INTO prices
                   (ticker, date, open, high, low, close, volume, adj_close, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            await db.commit()

        logger.info("Stored %d price bars", len(bars))
        return len(bars)

    async def get_bars(
        self, ticker: str, start: date, end: date
    ) -> list[PriceBar]:
        """Retrieve stored bars for a ticker and date range, sorted by date."""
        await self._ensure_table()

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """SELECT ticker, date, open, high, low, close, volume, adj_close, source
                   FROM prices
                   WHERE ticker = ? AND date >= ? AND date <= ?
                   ORDER BY date""",
                (ticker, start.isoformat(), end.isoformat()),
            )
            rows = await cursor.fetchall()

        return [
            PriceBar(
                ticker=row[0],
                date=date.fromisoformat(row[1]),
                open=row[2],
                high=row[3],
                low=row[4],
                close=row[5],
                volume=row[6],
                adj_close=row[7],
                source=row[8],
            )
            for row in rows
        ]

    async def has_data(
        self, ticker: str, start: date, end: date
    ) -> bool:
        """Check if any data exists for the ticker in the range."""
        await self._ensure_table()

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """SELECT COUNT(*) FROM prices
                   WHERE ticker = ? AND date >= ? AND date <= ?""",
                (ticker, start.isoformat(), end.isoformat()),
            )
            row = await cursor.fetchone()

        return row is not None and row[0] > 0

    async def get_tickers(self) -> list[str]:
        """Return all distinct tickers in the store."""
        await self._ensure_table()

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
            )
            rows = await cursor.fetchall()

        return [row[0] for row in rows]
