"""Storage backend: Protocol definition, SQLite implementation, factory."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import ClassVar, Protocol, runtime_checkable

import aiosqlite

from edgar_sentinel.core.config import StorageConfig
from edgar_sentinel.core.exceptions import StorageError
from edgar_sentinel.core.models import (
    CompositeSignal,
    Filing,
    FilingMetadata,
    FilingSection,
    FormType,
    SentimentResult,
    Signal,
    SimilarityResult,
    StorageBackend as StorageBackendEnum,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class StorageProtocol(Protocol):
    """Abstract storage interface for edgar-sentinel data."""

    async def save_filing(self, filing: Filing) -> None: ...
    async def get_filing(self, accession_number: str) -> Filing | None: ...
    async def list_filings(
        self,
        ticker: str | None = None,
        form_type: FormType | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
    ) -> list[FilingMetadata]: ...
    async def get_latest_filing(
        self, cik: str, form_type: FormType
    ) -> Filing | None: ...
    async def filing_exists(self, accession_number: str) -> bool: ...
    async def save_sentiment(self, result: SentimentResult) -> None: ...
    async def save_similarity(self, result: SimilarityResult) -> None: ...
    async def get_sentiments(
        self,
        filing_id: str,
        section_name: str | None = None,
        analyzer_name: str | None = None,
    ) -> list[SentimentResult]: ...
    async def get_similarity(
        self, filing_id: str, section_name: str | None = None
    ) -> list[SimilarityResult]: ...
    async def save_signal(self, signal: Signal) -> None: ...
    async def save_signals_batch(self, signals: list[Signal]) -> None: ...
    async def get_signals(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
        signal_name: str | None = None,
    ) -> list[Signal]: ...
    async def save_composite(self, composite: CompositeSignal) -> None: ...
    async def get_composites(
        self,
        ticker: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[CompositeSignal]: ...
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    async def health_check(self) -> bool: ...


class SqliteStore:
    """SQLite implementation of the storage protocol.

    Uses aiosqlite for async access, WAL mode for concurrent reads,
    and a version-tracked migration system.
    """

    _MIGRATIONS: ClassVar[dict[int, tuple[str, list[str]]]] = {
        1: (
            "Initial schema",
            [
                """CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT DEFAULT (datetime('now'))
                )""",
                """CREATE TABLE IF NOT EXISTS filings (
                    accession_number TEXT PRIMARY KEY,
                    cik TEXT NOT NULL,
                    ticker TEXT,
                    company_name TEXT NOT NULL,
                    form_type TEXT NOT NULL,
                    filed_date TEXT NOT NULL,
                    fiscal_year_end TEXT,
                    url TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )""",
                """CREATE TABLE IF NOT EXISTS filing_sections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filing_id TEXT NOT NULL REFERENCES filings(accession_number),
                    section_name TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    extracted_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(filing_id, section_name)
                )""",
                """CREATE TABLE IF NOT EXISTS sentiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filing_id TEXT NOT NULL REFERENCES filings(accession_number),
                    section_name TEXT NOT NULL,
                    analyzer_name TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    metadata_json TEXT,
                    analyzed_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(filing_id, section_name, analyzer_name)
                )""",
                """CREATE TABLE IF NOT EXISTS similarity_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filing_id TEXT NOT NULL REFERENCES filings(accession_number),
                    prior_filing_id TEXT NOT NULL REFERENCES filings(accession_number),
                    section_name TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    change_score REAL NOT NULL,
                    analyzed_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(filing_id, section_name)
                )""",
                """CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    signal_name TEXT NOT NULL,
                    raw_value REAL NOT NULL,
                    z_score REAL,
                    percentile REAL,
                    decay_weight REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, signal_date, signal_name)
                )""",
                """CREATE TABLE IF NOT EXISTS composite_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    composite_score REAL NOT NULL,
                    components_json TEXT,
                    rank INTEGER,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(ticker, signal_date)
                )""",
                """CREATE TABLE IF NOT EXISTS returns_cache (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    adj_close REAL NOT NULL,
                    daily_return REAL,
                    PRIMARY KEY(ticker, date)
                )""",
                # Indexes
                "CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker)",
                "CREATE INDEX IF NOT EXISTS idx_filings_filed_date ON filings(filed_date)",
                "CREATE INDEX IF NOT EXISTS idx_filings_form_type ON filings(form_type)",
                "CREATE INDEX IF NOT EXISTS idx_filings_cik ON filings(cik)",
                "CREATE INDEX IF NOT EXISTS idx_sections_filing_id ON filing_sections(filing_id)",
                "CREATE INDEX IF NOT EXISTS idx_sentiment_filing ON sentiment_results(filing_id)",
                "CREATE INDEX IF NOT EXISTS idx_similarity_filing ON similarity_results(filing_id)",
                "CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON signals(ticker, signal_date)",
                "CREATE INDEX IF NOT EXISTS idx_composite_date ON composite_signals(signal_date)",
                "CREATE INDEX IF NOT EXISTS idx_composite_ticker ON composite_signals(ticker)",
            ],
        ),
    }

    def __init__(self, config: StorageConfig) -> None:
        self._path = config.sqlite_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open connection, enable WAL + FK, run migrations."""
        try:
            self._db = await aiosqlite.connect(self._path)
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA foreign_keys=ON")
            current = await self._get_schema_version()
            await self._apply_migrations(current)
            await self._db.commit()
        except Exception as e:
            raise StorageError(
                f"Failed to initialize SQLite store: {e}",
                context={"operation": "initialize", "path": self._path},
            ) from e

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def health_check(self) -> bool:
        if self._db is None:
            return False
        try:
            async with self._db.execute("SELECT 1") as cursor:
                row = await cursor.fetchone()
            return row is not None
        except Exception:
            return False

    # --- Schema Migration ---

    async def _get_schema_version(self) -> int:
        try:
            async with self._db.execute(
                "SELECT MAX(version) FROM schema_version"
            ) as cursor:
                row = await cursor.fetchone()
            return row[0] if row[0] is not None else 0
        except aiosqlite.OperationalError:
            return 0

    async def _apply_migrations(self, current_version: int) -> None:
        for version in sorted(self._MIGRATIONS.keys()):
            if version <= current_version:
                continue
            desc, statements = self._MIGRATIONS[version]
            logger.info("Applying migration %d: %s", version, desc)
            for sql in statements:
                await self._db.execute(sql)
            await self._db.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (version,)
            )

    # --- Filing Operations ---

    async def save_filing(self, filing: Filing) -> None:
        try:
            meta = filing.metadata
            await self._db.execute(
                """INSERT OR REPLACE INTO filings
                   (accession_number, cik, ticker, company_name, form_type,
                    filed_date, fiscal_year_end, url)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    meta.accession_number,
                    meta.cik,
                    meta.ticker,
                    meta.company_name,
                    str(meta.form_type),
                    meta.filed_date.isoformat(),
                    meta.fiscal_year_end.isoformat() if meta.fiscal_year_end else None,
                    meta.url,
                ),
            )
            # Delete existing sections, then re-insert
            await self._db.execute(
                "DELETE FROM filing_sections WHERE filing_id = ?",
                (meta.accession_number,),
            )
            for section in filing.sections.values():
                await self._db.execute(
                    """INSERT INTO filing_sections
                       (filing_id, section_name, raw_text, word_count, extracted_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        section.filing_id,
                        section.section_name,
                        section.raw_text,
                        section.word_count,
                        section.extracted_at.isoformat(),
                    ),
                )
            await self._db.commit()
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to save filing: {e}",
                context={
                    "operation": "insert",
                    "table": "filings",
                    "accession_number": filing.metadata.accession_number,
                },
            ) from e

    async def get_filing(self, accession_number: str) -> Filing | None:
        try:
            async with self._db.execute(
                "SELECT * FROM filings WHERE accession_number = ?",
                (accession_number,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None
            metadata = self._row_to_filing_metadata(row)
            sections = await self._get_sections(accession_number)
            return Filing(metadata=metadata, sections=sections)
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to get filing: {e}",
                context={
                    "operation": "query",
                    "table": "filings",
                    "accession_number": accession_number,
                },
            ) from e

    async def list_filings(
        self,
        ticker: str | None = None,
        form_type: FormType | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
    ) -> list[FilingMetadata]:
        try:
            query = "SELECT * FROM filings WHERE 1=1"
            params: list = []
            if ticker is not None:
                query += " AND UPPER(ticker) = UPPER(?)"
                params.append(ticker)
            if form_type is not None:
                query += " AND form_type = ?"
                params.append(str(form_type))
            if start_date is not None:
                query += " AND filed_date >= ?"
                params.append(start_date.isoformat())
            if end_date is not None:
                query += " AND filed_date <= ?"
                params.append(end_date.isoformat())
            query += " ORDER BY filed_date DESC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            async with self._db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
            return [self._row_to_filing_metadata(r) for r in rows]
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to list filings: {e}",
                context={"operation": "query", "table": "filings"},
            ) from e

    async def get_latest_filing(
        self, cik: str, form_type: FormType
    ) -> Filing | None:
        try:
            # Zero-pad CIK for comparison
            padded_cik = cik.strip().zfill(10)
            async with self._db.execute(
                """SELECT accession_number FROM filings
                   WHERE cik = ? AND form_type = ?
                   ORDER BY filed_date DESC LIMIT 1""",
                (padded_cik, str(form_type)),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return None
            return await self.get_filing(row["accession_number"])
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to get latest filing: {e}",
                context={"operation": "query", "table": "filings"},
            ) from e

    async def filing_exists(self, accession_number: str) -> bool:
        try:
            async with self._db.execute(
                "SELECT 1 FROM filings WHERE accession_number = ?",
                (accession_number,),
            ) as cursor:
                row = await cursor.fetchone()
            return row is not None
        except Exception as e:
            raise StorageError(
                f"Failed to check filing existence: {e}",
                context={"operation": "query", "table": "filings"},
            ) from e

    # --- Sentiment Operations ---

    async def save_sentiment(self, result: SentimentResult) -> None:
        try:
            await self._db.execute(
                """INSERT OR REPLACE INTO sentiment_results
                   (filing_id, section_name, analyzer_name, sentiment_score,
                    confidence, metadata_json, analyzed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.filing_id,
                    result.section_name,
                    result.analyzer_name,
                    result.sentiment_score,
                    result.confidence,
                    json.dumps(result.metadata) if result.metadata else None,
                    result.analyzed_at.isoformat(),
                ),
            )
            await self._db.commit()
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to save sentiment: {e}",
                context={"operation": "insert", "table": "sentiment_results"},
            ) from e

    async def get_sentiments(
        self,
        filing_id: str,
        section_name: str | None = None,
        analyzer_name: str | None = None,
    ) -> list[SentimentResult]:
        try:
            query = "SELECT * FROM sentiment_results WHERE filing_id = ?"
            params: list = [filing_id]
            if section_name is not None:
                query += " AND section_name = ?"
                params.append(section_name)
            if analyzer_name is not None:
                query += " AND analyzer_name = ?"
                params.append(analyzer_name)
            async with self._db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
            return [self._row_to_sentiment_result(r) for r in rows]
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to get sentiments: {e}",
                context={"operation": "query", "table": "sentiment_results"},
            ) from e

    # --- Similarity Operations ---

    async def save_similarity(self, result: SimilarityResult) -> None:
        try:
            await self._db.execute(
                """INSERT OR REPLACE INTO similarity_results
                   (filing_id, prior_filing_id, section_name,
                    similarity_score, change_score, analyzed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    result.filing_id,
                    result.prior_filing_id,
                    result.section_name,
                    result.similarity_score,
                    result.change_score,
                    result.analyzed_at.isoformat(),
                ),
            )
            await self._db.commit()
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to save similarity: {e}",
                context={"operation": "insert", "table": "similarity_results"},
            ) from e

    async def get_similarity(
        self, filing_id: str, section_name: str | None = None
    ) -> list[SimilarityResult]:
        try:
            query = "SELECT * FROM similarity_results WHERE filing_id = ?"
            params: list = [filing_id]
            if section_name is not None:
                query += " AND section_name = ?"
                params.append(section_name)
            async with self._db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
            return [self._row_to_similarity_result(r) for r in rows]
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to get similarity: {e}",
                context={"operation": "query", "table": "similarity_results"},
            ) from e

    # --- Signal Operations ---

    async def save_signal(self, signal: Signal) -> None:
        try:
            await self._db.execute(
                """INSERT OR REPLACE INTO signals
                   (ticker, signal_date, signal_name, raw_value,
                    z_score, percentile, decay_weight)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal.ticker,
                    signal.signal_date.isoformat(),
                    signal.signal_name,
                    signal.raw_value,
                    signal.z_score,
                    signal.percentile,
                    signal.decay_weight,
                ),
            )
            await self._db.commit()
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to save signal: {e}",
                context={"operation": "insert", "table": "signals"},
            ) from e

    async def save_signals_batch(self, signals: list[Signal]) -> None:
        try:
            for signal in signals:
                await self._db.execute(
                    """INSERT OR REPLACE INTO signals
                       (ticker, signal_date, signal_name, raw_value,
                        z_score, percentile, decay_weight)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        signal.ticker,
                        signal.signal_date.isoformat(),
                        signal.signal_name,
                        signal.raw_value,
                        signal.z_score,
                        signal.percentile,
                        signal.decay_weight,
                    ),
                )
            await self._db.commit()
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to save signals batch: {e}",
                context={"operation": "insert", "table": "signals"},
            ) from e

    async def get_signals(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
        signal_name: str | None = None,
    ) -> list[Signal]:
        try:
            query = "SELECT * FROM signals WHERE ticker = ?"
            params: list = [ticker]
            if start_date is not None:
                query += " AND signal_date >= ?"
                params.append(start_date.isoformat())
            if end_date is not None:
                query += " AND signal_date <= ?"
                params.append(end_date.isoformat())
            if signal_name is not None:
                query += " AND signal_name = ?"
                params.append(signal_name)
            query += " ORDER BY signal_date ASC"
            async with self._db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
            return [self._row_to_signal(r) for r in rows]
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to get signals: {e}",
                context={"operation": "query", "table": "signals"},
            ) from e

    # --- Composite Signal Operations ---

    async def save_composite(self, composite: CompositeSignal) -> None:
        try:
            await self._db.execute(
                """INSERT OR REPLACE INTO composite_signals
                   (ticker, signal_date, composite_score, components_json, rank)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    composite.ticker,
                    composite.signal_date.isoformat(),
                    composite.composite_score,
                    json.dumps(composite.components),
                    composite.rank,
                ),
            )
            await self._db.commit()
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to save composite: {e}",
                context={"operation": "insert", "table": "composite_signals"},
            ) from e

    async def get_composites(
        self,
        ticker: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[CompositeSignal]:
        try:
            query = "SELECT * FROM composite_signals WHERE 1=1"
            params: list = []
            if ticker is not None:
                query += " AND ticker = ?"
                params.append(ticker)
            if start_date is not None:
                query += " AND signal_date >= ?"
                params.append(start_date.isoformat())
            if end_date is not None:
                query += " AND signal_date <= ?"
                params.append(end_date.isoformat())
            query += " ORDER BY signal_date ASC"
            async with self._db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
            return [self._row_to_composite(r) for r in rows]
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to get composites: {e}",
                context={"operation": "query", "table": "composite_signals"},
            ) from e

    # --- Row Mapping Helpers ---

    async def _get_sections(
        self, accession_number: str
    ) -> dict[str, FilingSection]:
        async with self._db.execute(
            "SELECT * FROM filing_sections WHERE filing_id = ?",
            (accession_number,),
        ) as cursor:
            rows = await cursor.fetchall()
        sections = {}
        for row in rows:
            section = self._row_to_filing_section(row)
            sections[section.section_name] = section
        return sections

    @staticmethod
    def _row_to_filing_metadata(row: aiosqlite.Row) -> FilingMetadata:
        return FilingMetadata(
            cik=row["cik"],
            ticker=row["ticker"],
            company_name=row["company_name"],
            form_type=FormType(row["form_type"]),
            filed_date=date.fromisoformat(row["filed_date"]),
            accession_number=row["accession_number"],
            fiscal_year_end=(
                date.fromisoformat(row["fiscal_year_end"])
                if row["fiscal_year_end"]
                else None
            ),
            url=row["url"],
        )

    @staticmethod
    def _row_to_filing_section(row: aiosqlite.Row) -> FilingSection:
        return FilingSection(
            filing_id=row["filing_id"],
            section_name=row["section_name"],
            raw_text=row["raw_text"],
            word_count=row["word_count"],
            extracted_at=datetime.fromisoformat(row["extracted_at"]),
        )

    @staticmethod
    def _row_to_sentiment_result(row: aiosqlite.Row) -> SentimentResult:
        meta_json = row["metadata_json"]
        return SentimentResult(
            filing_id=row["filing_id"],
            section_name=row["section_name"],
            analyzer_name=row["analyzer_name"],
            sentiment_score=row["sentiment_score"],
            confidence=row["confidence"],
            metadata=json.loads(meta_json) if meta_json else {},
            analyzed_at=datetime.fromisoformat(row["analyzed_at"]),
        )

    @staticmethod
    def _row_to_similarity_result(row: aiosqlite.Row) -> SimilarityResult:
        return SimilarityResult(
            filing_id=row["filing_id"],
            prior_filing_id=row["prior_filing_id"],
            section_name=row["section_name"],
            similarity_score=row["similarity_score"],
            change_score=row["change_score"],
            analyzed_at=datetime.fromisoformat(row["analyzed_at"]),
        )

    @staticmethod
    def _row_to_signal(row: aiosqlite.Row) -> Signal:
        return Signal(
            ticker=row["ticker"],
            signal_date=date.fromisoformat(row["signal_date"]),
            signal_name=row["signal_name"],
            raw_value=row["raw_value"],
            z_score=row["z_score"],
            percentile=row["percentile"],
            decay_weight=row["decay_weight"],
        )

    @staticmethod
    def _row_to_composite(row: aiosqlite.Row) -> CompositeSignal:
        comp_json = row["components_json"]
        return CompositeSignal(
            ticker=row["ticker"],
            signal_date=date.fromisoformat(row["signal_date"]),
            composite_score=row["composite_score"],
            components=json.loads(comp_json) if comp_json else {},
            rank=row["rank"],
        )


async def create_store(config: StorageConfig) -> SqliteStore:
    """Create and initialize a storage backend based on configuration."""
    if config.backend == StorageBackendEnum.SQLITE:
        store = SqliteStore(config)
        await store.initialize()
        return store
    raise StorageError(
        f"Unsupported storage backend: {config.backend}",
        context={"operation": "create_store", "backend": str(config.backend)},
    )
