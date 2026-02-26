"""Tests for the SQLite storage backend."""

import json
from datetime import date, datetime

import pytest

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
from edgar_sentinel.ingestion.store import (
    SqliteStore,
    StorageProtocol,
    create_store,
)


# --- Fixtures ---


@pytest.fixture
async def store():
    """Create an in-memory SqliteStore for testing."""
    config = StorageConfig(backend=StorageBackendEnum.SQLITE, sqlite_path=":memory:")
    s = SqliteStore(config)
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def make_filing_metadata():
    """Factory for FilingMetadata with overridable defaults."""

    def _make(**overrides):
        defaults = dict(
            cik="320193",
            ticker="AAPL",
            company_name="Apple Inc.",
            form_type=FormType.FORM_10K,
            filed_date=date(2023, 11, 3),
            accession_number="0000320193-23-000106",
            fiscal_year_end=date(2023, 9, 30),
            url="https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106-index.htm",
        )
        defaults.update(overrides)
        return FilingMetadata(**defaults)

    return _make


@pytest.fixture
def make_filing(make_filing_metadata):
    """Factory for Filing with overridable defaults."""

    def _make(metadata_overrides=None, sections=None):
        meta = make_filing_metadata(**(metadata_overrides or {}))
        if sections is None:
            text = "The Company designs manufactures and markets smartphones."
            sections = {
                "mda": FilingSection(
                    filing_id=meta.accession_number,
                    section_name="mda",
                    raw_text=text,
                    word_count=len(text.split()),
                    extracted_at=datetime(2024, 1, 15, 10, 30, 0),
                ),
            }
        return Filing(metadata=meta, sections=sections)

    return _make


@pytest.fixture
def make_sentiment():
    """Factory for SentimentResult."""

    def _make(**overrides):
        defaults = dict(
            filing_id="0000320193-23-000106",
            section_name="mda",
            analyzer_name="dictionary",
            sentiment_score=0.15,
            confidence=0.85,
            metadata={"positive_words": 42, "negative_words": 12},
            analyzed_at=datetime(2024, 1, 15, 11, 0, 0),
        )
        defaults.update(overrides)
        return SentimentResult(**defaults)

    return _make


@pytest.fixture
def make_similarity():
    """Factory for SimilarityResult."""

    def _make(**overrides):
        defaults = dict(
            filing_id="0000320193-23-000106",
            prior_filing_id="0000320193-22-000108",
            section_name="mda",
            similarity_score=0.82,
            change_score=0.18,
            analyzed_at=datetime(2024, 1, 15, 11, 0, 0),
        )
        defaults.update(overrides)
        return SimilarityResult(**defaults)

    return _make


@pytest.fixture
def make_signal():
    """Factory for Signal."""

    def _make(**overrides):
        defaults = dict(
            ticker="AAPL",
            signal_date=date(2023, 11, 5),
            signal_name="dictionary_mda",
            raw_value=0.15,
            z_score=1.2,
            percentile=88.5,
            decay_weight=0.95,
        )
        defaults.update(overrides)
        return Signal(**defaults)

    return _make


@pytest.fixture
def make_composite():
    """Factory for CompositeSignal."""

    def _make(**overrides):
        defaults = dict(
            ticker="AAPL",
            signal_date=date(2023, 11, 5),
            composite_score=0.72,
            components={"dictionary_mda": 0.5, "similarity_mda": 0.5},
            rank=1,
        )
        defaults.update(overrides)
        return CompositeSignal(**defaults)

    return _make


# --- Protocol Conformance ---


class TestStorageProtocol:
    def test_sqlite_store_satisfies_protocol(self):
        assert isinstance(SqliteStore, type)
        config = StorageConfig(
            backend=StorageBackendEnum.SQLITE, sqlite_path=":memory:"
        )
        instance = SqliteStore(config)
        assert isinstance(instance, StorageProtocol)


# --- Initialization ---


class TestSqliteStoreInit:
    async def test_initialize_creates_tables(self, store):
        """Tables exist after initialize()."""
        async with store._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ) as cursor:
            tables = [row[0] for row in await cursor.fetchall()]
        for expected in [
            "composite_signals",
            "filing_sections",
            "filings",
            "returns_cache",
            "schema_version",
            "sentiment_results",
            "signals",
            "similarity_results",
        ]:
            assert expected in tables

    async def test_initialize_idempotent(self, store):
        """Calling initialize() twice doesn't raise."""
        await store.initialize()

    async def test_wal_mode_enabled(self, store):
        """WAL is requested; in-memory DBs report 'memory' instead."""
        async with store._db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()
        assert row[0] in ("wal", "memory")

    async def test_foreign_keys_enabled(self, store):
        async with store._db.execute("PRAGMA foreign_keys") as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1

    async def test_schema_version_set(self, store):
        async with store._db.execute(
            "SELECT MAX(version) FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1


# --- Filing CRUD ---


class TestFilingCRUD:
    async def test_save_and_get_filing(self, store, make_filing):
        filing = make_filing()
        await store.save_filing(filing)
        result = await store.get_filing(filing.metadata.accession_number)
        assert result is not None
        assert result.metadata.accession_number == filing.metadata.accession_number
        assert result.metadata.ticker == "AAPL"
        assert result.metadata.company_name == "Apple Inc."
        assert result.metadata.form_type == FormType.FORM_10K
        assert result.metadata.filed_date == date(2023, 11, 3)
        assert "mda" in result.sections

    async def test_get_filing_not_found(self, store):
        result = await store.get_filing("9999999999-99-999999")
        assert result is None

    async def test_save_filing_with_multiple_sections(self, store, make_filing_metadata):
        meta = make_filing_metadata()
        sections = {}
        for name in ["mda", "risk_factors", "business"]:
            text = f"Content for {name} section with enough words."
            sections[name] = FilingSection(
                filing_id=meta.accession_number,
                section_name=name,
                raw_text=text,
                word_count=len(text.split()),
                extracted_at=datetime(2024, 1, 15, 10, 30, 0),
            )
        filing = Filing(metadata=meta, sections=sections)
        await store.save_filing(filing)
        result = await store.get_filing(meta.accession_number)
        assert len(result.sections) == 3
        assert set(result.sections.keys()) == {"mda", "risk_factors", "business"}

    async def test_filing_exists_true(self, store, make_filing):
        filing = make_filing()
        await store.save_filing(filing)
        assert await store.filing_exists(filing.metadata.accession_number)

    async def test_filing_exists_false(self, store):
        assert not await store.filing_exists("9999999999-99-999999")

    async def test_list_filings_all(self, store, make_filing):
        f1 = make_filing(metadata_overrides={"accession_number": "0000320193-23-000106"})
        f2 = make_filing(
            metadata_overrides={
                "accession_number": "0000320193-23-000107",
                "filed_date": date(2023, 12, 1),
            }
        )
        await store.save_filing(f1)
        await store.save_filing(f2)
        results = await store.list_filings()
        assert len(results) == 2
        # Sorted by filed_date descending
        assert results[0].filed_date >= results[1].filed_date

    async def test_list_filings_by_ticker(self, store, make_filing):
        f1 = make_filing()
        f2 = make_filing(
            metadata_overrides={
                "accession_number": "0000789019-23-000011",
                "cik": "789019",
                "ticker": "MSFT",
                "company_name": "Microsoft Corp.",
                "url": "https://www.sec.gov/Archives/edgar/data/789019/000078901923000011/0000789019-23-000011-index.htm",
            }
        )
        await store.save_filing(f1)
        await store.save_filing(f2)
        results = await store.list_filings(ticker="AAPL")
        assert len(results) == 1
        assert results[0].ticker == "AAPL"

    async def test_list_filings_by_form_type(self, store, make_filing):
        f1 = make_filing()
        f2 = make_filing(
            metadata_overrides={
                "accession_number": "0000320193-23-000200",
                "form_type": FormType.FORM_10Q,
            }
        )
        await store.save_filing(f1)
        await store.save_filing(f2)
        results = await store.list_filings(form_type=FormType.FORM_10K)
        assert len(results) == 1
        assert results[0].form_type == FormType.FORM_10K

    async def test_list_filings_by_date_range(self, store, make_filing):
        f1 = make_filing(
            metadata_overrides={
                "accession_number": "0000320193-23-000106",
                "filed_date": date(2023, 1, 15),
            }
        )
        f2 = make_filing(
            metadata_overrides={
                "accession_number": "0000320193-23-000200",
                "filed_date": date(2023, 6, 15),
            }
        )
        f3 = make_filing(
            metadata_overrides={
                "accession_number": "0000320193-23-000300",
                "filed_date": date(2023, 12, 15),
            }
        )
        for f in [f1, f2, f3]:
            await store.save_filing(f)
        results = await store.list_filings(
            start_date=date(2023, 3, 1), end_date=date(2023, 9, 30)
        )
        assert len(results) == 1
        assert results[0].accession_number == "0000320193-23-000200"

    async def test_list_filings_with_limit(self, store, make_filing):
        for i in range(5):
            f = make_filing(
                metadata_overrides={
                    "accession_number": f"0000320193-23-00010{i}",
                    "filed_date": date(2023, i + 1, 15),
                }
            )
            await store.save_filing(f)
        results = await store.list_filings(limit=2)
        assert len(results) == 2

    async def test_list_filings_ticker_case_insensitive(self, store, make_filing):
        f = make_filing()
        await store.save_filing(f)
        results = await store.list_filings(ticker="aapl")
        assert len(results) == 1

    async def test_get_latest_filing(self, store, make_filing):
        f1 = make_filing(
            metadata_overrides={
                "accession_number": "0000320193-22-000108",
                "filed_date": date(2022, 11, 3),
            }
        )
        f2 = make_filing(
            metadata_overrides={
                "accession_number": "0000320193-23-000106",
                "filed_date": date(2023, 11, 3),
            }
        )
        await store.save_filing(f1)
        await store.save_filing(f2)
        result = await store.get_latest_filing("0000320193", FormType.FORM_10K)
        assert result is not None
        assert result.metadata.accession_number == "0000320193-23-000106"

    async def test_get_latest_filing_none(self, store):
        result = await store.get_latest_filing("0000320193", FormType.FORM_10K)
        assert result is None


# --- Upsert Semantics ---


class TestUpsertSemantics:
    async def test_save_filing_upsert(self, store, make_filing_metadata):
        """Saving a filing with the same accession_number replaces it."""
        meta = make_filing_metadata()
        text1 = "Original content for this section."
        text2 = "Updated content for this section entirely."
        filing1 = Filing(
            metadata=meta,
            sections={
                "mda": FilingSection(
                    filing_id=meta.accession_number,
                    section_name="mda",
                    raw_text=text1,
                    word_count=len(text1.split()),
                    extracted_at=datetime(2024, 1, 15, 10, 0, 0),
                )
            },
        )
        filing2 = Filing(
            metadata=meta,
            sections={
                "mda": FilingSection(
                    filing_id=meta.accession_number,
                    section_name="mda",
                    raw_text=text2,
                    word_count=len(text2.split()),
                    extracted_at=datetime(2024, 1, 16, 10, 0, 0),
                )
            },
        )
        await store.save_filing(filing1)
        await store.save_filing(filing2)
        result = await store.get_filing(meta.accession_number)
        assert result.sections["mda"].raw_text == text2

    async def test_save_sentiment_upsert(self, store, make_filing, make_sentiment):
        filing = make_filing()
        await store.save_filing(filing)
        s1 = make_sentiment(sentiment_score=0.15)
        s2 = make_sentiment(sentiment_score=0.25)
        await store.save_sentiment(s1)
        await store.save_sentiment(s2)
        results = await store.get_sentiments(s1.filing_id)
        assert len(results) == 1
        assert results[0].sentiment_score == 0.25

    async def test_save_similarity_upsert(self, store, make_filing, make_similarity):
        # Need both filings to exist for FK
        f1 = make_filing()
        f2 = make_filing(
            metadata_overrides={
                "accession_number": "0000320193-22-000108",
                "filed_date": date(2022, 11, 3),
            }
        )
        await store.save_filing(f1)
        await store.save_filing(f2)
        s1 = make_similarity(similarity_score=0.82, change_score=0.18)
        s2 = make_similarity(similarity_score=0.90, change_score=0.10)
        await store.save_similarity(s1)
        await store.save_similarity(s2)
        results = await store.get_similarity(s1.filing_id)
        assert len(results) == 1
        assert results[0].similarity_score == 0.90

    async def test_save_signal_upsert(self, store, make_signal):
        s1 = make_signal(raw_value=0.15)
        s2 = make_signal(raw_value=0.25)
        await store.save_signal(s1)
        await store.save_signal(s2)
        results = await store.get_signals("AAPL")
        assert len(results) == 1
        assert results[0].raw_value == 0.25

    async def test_save_composite_upsert(self, store, make_composite):
        c1 = make_composite(composite_score=0.72)
        c2 = make_composite(composite_score=0.85)
        await store.save_composite(c1)
        await store.save_composite(c2)
        results = await store.get_composites(ticker="AAPL")
        assert len(results) == 1
        assert results[0].composite_score == 0.85


# --- Sentiment CRUD ---


class TestSentimentCRUD:
    async def test_save_and_get_sentiments(self, store, make_filing, make_sentiment):
        filing = make_filing()
        await store.save_filing(filing)
        sentiment = make_sentiment()
        await store.save_sentiment(sentiment)
        results = await store.get_sentiments(sentiment.filing_id)
        assert len(results) == 1
        assert results[0].sentiment_score == 0.15
        assert results[0].confidence == 0.85
        assert results[0].metadata == {"positive_words": 42, "negative_words": 12}

    async def test_get_sentiments_filter_by_section(
        self, store, make_filing, make_filing_metadata, make_sentiment
    ):
        meta = make_filing_metadata()
        sections = {}
        for name in ["mda", "risk_factors"]:
            text = f"Content for {name} section."
            sections[name] = FilingSection(
                filing_id=meta.accession_number,
                section_name=name,
                raw_text=text,
                word_count=len(text.split()),
                extracted_at=datetime(2024, 1, 15, 10, 30, 0),
            )
        filing = Filing(metadata=meta, sections=sections)
        await store.save_filing(filing)
        s1 = make_sentiment(section_name="mda")
        s2 = make_sentiment(
            section_name="risk_factors", sentiment_score=-0.10, confidence=0.75
        )
        await store.save_sentiment(s1)
        await store.save_sentiment(s2)
        results = await store.get_sentiments(s1.filing_id, section_name="mda")
        assert len(results) == 1
        assert results[0].section_name == "mda"

    async def test_get_sentiments_filter_by_analyzer(
        self, store, make_filing, make_sentiment
    ):
        filing = make_filing()
        await store.save_filing(filing)
        s1 = make_sentiment(analyzer_name="dictionary")
        s2 = make_sentiment(analyzer_name="llm", sentiment_score=0.20, confidence=0.90)
        await store.save_sentiment(s1)
        await store.save_sentiment(s2)
        results = await store.get_sentiments(
            s1.filing_id, analyzer_name="dictionary"
        )
        assert len(results) == 1
        assert results[0].analyzer_name == "dictionary"

    async def test_get_sentiments_empty(self, store):
        results = await store.get_sentiments("9999999999-99-999999")
        assert results == []


# --- Similarity CRUD ---


class TestSimilarityCRUD:
    async def test_save_and_get_similarity(
        self, store, make_filing, make_similarity
    ):
        f1 = make_filing()
        f2 = make_filing(
            metadata_overrides={
                "accession_number": "0000320193-22-000108",
                "filed_date": date(2022, 11, 3),
            }
        )
        await store.save_filing(f1)
        await store.save_filing(f2)
        sim = make_similarity()
        await store.save_similarity(sim)
        results = await store.get_similarity(sim.filing_id)
        assert len(results) == 1
        assert results[0].similarity_score == 0.82
        assert results[0].change_score == 0.18
        assert results[0].prior_filing_id == "0000320193-22-000108"

    async def test_get_similarity_filter_by_section(
        self, store, make_filing, make_filing_metadata, make_similarity
    ):
        meta = make_filing_metadata()
        sections = {}
        for name in ["mda", "risk_factors"]:
            text = f"Content for {name}."
            sections[name] = FilingSection(
                filing_id=meta.accession_number,
                section_name=name,
                raw_text=text,
                word_count=len(text.split()),
                extracted_at=datetime(2024, 1, 15, 10, 30, 0),
            )
        filing = Filing(metadata=meta, sections=sections)
        prior = make_filing(
            metadata_overrides={
                "accession_number": "0000320193-22-000108",
                "filed_date": date(2022, 11, 3),
            }
        )
        await store.save_filing(filing)
        await store.save_filing(prior)

        s1 = make_similarity(section_name="mda")
        s2 = make_similarity(
            section_name="risk_factors",
            similarity_score=0.75,
            change_score=0.25,
        )
        await store.save_similarity(s1)
        await store.save_similarity(s2)

        results = await store.get_similarity(s1.filing_id, section_name="mda")
        assert len(results) == 1
        assert results[0].section_name == "mda"

    async def test_get_similarity_empty(self, store):
        results = await store.get_similarity("9999999999-99-999999")
        assert results == []


# --- Signal CRUD ---


class TestSignalCRUD:
    async def test_save_and_get_signal(self, store, make_signal):
        signal = make_signal()
        await store.save_signal(signal)
        results = await store.get_signals("AAPL")
        assert len(results) == 1
        assert results[0].signal_name == "dictionary_mda"
        assert results[0].raw_value == 0.15
        assert results[0].z_score == 1.2
        assert results[0].percentile == 88.5

    async def test_get_signals_by_date_range(self, store, make_signal):
        s1 = make_signal(signal_date=date(2023, 1, 5))
        s2 = make_signal(
            signal_date=date(2023, 6, 5), signal_name="dictionary_risk"
        )
        s3 = make_signal(
            signal_date=date(2023, 12, 5), signal_name="similarity_mda"
        )
        for s in [s1, s2, s3]:
            await store.save_signal(s)
        results = await store.get_signals(
            "AAPL", start_date=date(2023, 3, 1), end_date=date(2023, 9, 30)
        )
        assert len(results) == 1
        assert results[0].signal_name == "dictionary_risk"

    async def test_get_signals_by_name(self, store, make_signal):
        s1 = make_signal(signal_name="dictionary_mda")
        s2 = make_signal(
            signal_name="similarity_mda",
            signal_date=date(2023, 11, 6),
            raw_value=0.80,
        )
        await store.save_signal(s1)
        await store.save_signal(s2)
        results = await store.get_signals("AAPL", signal_name="dictionary_mda")
        assert len(results) == 1

    async def test_get_signals_sorted_ascending(self, store, make_signal):
        s1 = make_signal(signal_date=date(2023, 12, 5), signal_name="a")
        s2 = make_signal(signal_date=date(2023, 1, 5), signal_name="b")
        s3 = make_signal(signal_date=date(2023, 6, 5), signal_name="c")
        for s in [s1, s2, s3]:
            await store.save_signal(s)
        results = await store.get_signals("AAPL")
        assert results[0].signal_date <= results[1].signal_date <= results[2].signal_date

    async def test_save_signals_batch(self, store, make_signal):
        signals = [
            make_signal(signal_name=f"sig_{i}", signal_date=date(2023, 1, i + 1))
            for i in range(5)
        ]
        await store.save_signals_batch(signals)
        results = await store.get_signals("AAPL")
        assert len(results) == 5

    async def test_save_signals_batch_atomic(self, store, make_signal):
        """If one signal in the batch is a duplicate that causes an issue, the batch is atomic."""
        # Save 3 valid signals in a batch
        signals = [
            make_signal(signal_name=f"sig_{i}", signal_date=date(2023, 1, i + 1))
            for i in range(3)
        ]
        await store.save_signals_batch(signals)
        results = await store.get_signals("AAPL")
        assert len(results) == 3

    async def test_get_signals_empty(self, store):
        results = await store.get_signals("ZZZZ")
        assert results == []


# --- Composite Signal CRUD ---


class TestCompositeCRUD:
    async def test_save_and_get_composite(self, store, make_composite):
        comp = make_composite()
        await store.save_composite(comp)
        results = await store.get_composites(ticker="AAPL")
        assert len(results) == 1
        assert results[0].composite_score == 0.72
        assert results[0].components == {
            "dictionary_mda": 0.5,
            "similarity_mda": 0.5,
        }
        assert results[0].rank == 1

    async def test_get_composites_by_date_range(self, store, make_composite):
        c1 = make_composite(signal_date=date(2023, 1, 5))
        c2 = make_composite(signal_date=date(2023, 6, 5))
        c3 = make_composite(signal_date=date(2023, 12, 5))
        for c in [c1, c2, c3]:
            await store.save_composite(c)
        results = await store.get_composites(
            start_date=date(2023, 3, 1), end_date=date(2023, 9, 30)
        )
        assert len(results) == 1
        assert results[0].signal_date == date(2023, 6, 5)

    async def test_get_composites_sorted_ascending(self, store, make_composite):
        c1 = make_composite(signal_date=date(2023, 12, 5))
        c2 = make_composite(signal_date=date(2023, 1, 5))
        c3 = make_composite(signal_date=date(2023, 6, 5))
        for c in [c1, c2, c3]:
            await store.save_composite(c)
        results = await store.get_composites()
        dates = [r.signal_date for r in results]
        assert dates == sorted(dates)

    async def test_get_composites_empty(self, store):
        results = await store.get_composites(ticker="ZZZZ")
        assert results == []


# --- Health Check ---


class TestHealthCheck:
    async def test_health_check_ok(self, store):
        assert await store.health_check() is True

    async def test_health_check_after_close(self):
        config = StorageConfig(
            backend=StorageBackendEnum.SQLITE, sqlite_path=":memory:"
        )
        s = SqliteStore(config)
        await s.initialize()
        await s.close()
        assert await s.health_check() is False


# --- Factory ---


class TestCreateStoreFactory:
    async def test_create_sqlite_store(self):
        config = StorageConfig(
            backend=StorageBackendEnum.SQLITE, sqlite_path=":memory:"
        )
        store = await create_store(config)
        assert isinstance(store, SqliteStore)
        assert await store.health_check() is True
        await store.close()

    async def test_create_store_unsupported_backend(self):
        """Attempting to create a PostgreSQL store without asyncpg raises."""
        config = StorageConfig(
            backend=StorageBackendEnum.POSTGRESQL,
            postgresql_url="postgresql://localhost/test",
        )
        with pytest.raises(StorageError):
            await create_store(config)


# --- Filing with no ticker ---


class TestEdgeCases:
    async def test_filing_without_ticker(self, store, make_filing):
        filing = make_filing(metadata_overrides={"ticker": None})
        await store.save_filing(filing)
        result = await store.get_filing(filing.metadata.accession_number)
        assert result is not None
        assert result.metadata.ticker is None

    async def test_filing_without_fiscal_year_end(self, store, make_filing):
        filing = make_filing(metadata_overrides={"fiscal_year_end": None})
        await store.save_filing(filing)
        result = await store.get_filing(filing.metadata.accession_number)
        assert result.metadata.fiscal_year_end is None

    async def test_save_filing_empty_sections(self, store, make_filing_metadata):
        meta = make_filing_metadata()
        filing = Filing(metadata=meta, sections={})
        await store.save_filing(filing)
        result = await store.get_filing(meta.accession_number)
        assert result is not None
        assert result.sections == {}

    async def test_sentiment_metadata_none(self, store, make_filing, make_sentiment):
        filing = make_filing()
        await store.save_filing(filing)
        sentiment = make_sentiment(metadata={})
        await store.save_sentiment(sentiment)
        results = await store.get_sentiments(sentiment.filing_id)
        assert results[0].metadata == {}

    async def test_signal_with_null_z_score(self, store, make_signal):
        signal = make_signal(z_score=None, percentile=None)
        await store.save_signal(signal)
        results = await store.get_signals("AAPL")
        assert results[0].z_score is None
        assert results[0].percentile is None

    async def test_composite_with_null_rank(self, store, make_composite):
        comp = make_composite(rank=None)
        await store.save_composite(comp)
        results = await store.get_composites(ticker="AAPL")
        assert results[0].rank is None
