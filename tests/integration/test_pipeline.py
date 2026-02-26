"""Integration tests for the full analysis pipeline.

Tests modules working together with real SQLite I/O, no network calls.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from edgar_sentinel.core.models import (
    CompositeSignal,
    Filing,
    FilingMetadata,
    FilingSection,
    FormType,
    SentimentResult,
    Signal,
)


pytestmark = pytest.mark.integration


class TestStorageRoundTrip:
    """Test filing save -> retrieve round trips."""

    async def test_save_and_retrieve_filing(self, integration_store):
        """Filing saved to storage can be retrieved intact."""
        meta = FilingMetadata(
            cik="320193",
            ticker="AAPL",
            company_name="Apple Inc.",
            form_type=FormType.FORM_10K,
            filed_date=date(2023, 11, 3),
            accession_number="0000320193-23-000106",
            url="https://www.sec.gov/test/filing",
        )
        section = FilingSection(
            filing_id="0000320193-23-000106",
            section_name="mda",
            raw_text="Revenue grew significantly due to strong demand.",
            word_count=7,
            extracted_at=datetime(2023, 11, 4, 10, 0, 0),
        )
        filing = Filing(metadata=meta, sections={"mda": section})

        await integration_store.save_filing(filing)
        retrieved = await integration_store.get_filing("0000320193-23-000106")

        assert retrieved is not None
        assert retrieved.metadata.ticker == "AAPL"
        assert retrieved.metadata.accession_number == "0000320193-23-000106"
        assert "mda" in retrieved.sections
        assert "Revenue grew" in retrieved.sections["mda"].raw_text

    async def test_list_filings_filters_by_ticker(self, populated_store):
        """list_filings with ticker filter returns only matching filings."""
        filings = await populated_store.list_filings(ticker="AAPL")
        assert len(filings) == 5
        assert all(f.ticker == "AAPL" for f in filings)

    async def test_list_filings_filters_by_date_range(self, populated_store):
        """list_filings with date range returns only matching filings."""
        filings = await populated_store.list_filings(
            start_date=date(2021, 1, 1),
            end_date=date(2022, 12, 31),
        )
        assert len(filings) == 2
        for f in filings:
            assert f.filed_date >= date(2021, 1, 1)
            assert f.filed_date <= date(2022, 12, 31)


class TestSentimentStorageIntegration:
    """Test sentiment results round trip through storage."""

    async def test_save_and_retrieve_sentiment(self, populated_store):
        """Sentiment result can be saved and retrieved by filing_id."""
        result = SentimentResult(
            filing_id="0000320193-23-000106",
            section_name="mda",
            analyzer_name="dictionary",
            sentiment_score=0.15,
            confidence=0.85,
            analyzed_at=datetime(2024, 1, 15, 11, 0, 0),
        )
        await populated_store.save_sentiment(result)

        retrieved = await populated_store.get_sentiments(
            filing_id="0000320193-23-000106",
            analyzer_name="dictionary",
        )
        assert len(retrieved) == 1
        assert retrieved[0].sentiment_score == pytest.approx(0.15)
        assert retrieved[0].confidence == pytest.approx(0.85)


class TestSignalStorageIntegration:
    """Test signal storage round trip."""

    async def test_save_and_retrieve_signals(self, populated_store):
        """Signals can be batch-saved and retrieved by ticker."""
        signals = [
            Signal(
                ticker="AAPL",
                signal_date=date(2023, 11, 5),
                signal_name="dictionary_mda",
                raw_value=0.15,
                z_score=1.2,
                percentile=0.88,
            ),
            Signal(
                ticker="AAPL",
                signal_date=date(2022, 11, 5),
                signal_name="dictionary_mda",
                raw_value=0.10,
                z_score=0.8,
                percentile=0.72,
            ),
        ]
        await populated_store.save_signals_batch(signals)

        retrieved = await populated_store.get_signals(
            ticker="AAPL", signal_name="dictionary_mda"
        )
        assert len(retrieved) == 2

    async def test_composite_signal_round_trip(self, populated_store):
        """Composite signal can be saved and retrieved."""
        composite = CompositeSignal(
            ticker="AAPL",
            signal_date=date(2023, 11, 5),
            composite_score=0.72,
            components={"dictionary_mda": 0.5, "similarity_mda": 0.5},
            rank=1,
        )
        await populated_store.save_composite(composite)

        retrieved = await populated_store.get_composites(ticker="AAPL")
        assert len(retrieved) == 1
        assert retrieved[0].composite_score == pytest.approx(0.72)
        assert retrieved[0].components["dictionary_mda"] == pytest.approx(0.5)


class TestStatistics:
    """Test storage statistics gathering."""

    async def test_statistics_reflect_stored_data(self, populated_store):
        """get_statistics returns correct counts."""
        stats = await populated_store.get_statistics()
        assert stats["total_filings"] == 5

    async def test_filing_exists_returns_true_for_stored(self, populated_store):
        """filing_exists returns True for stored filings."""
        assert await populated_store.filing_exists("0000320193-23-000106")

    async def test_filing_exists_returns_false_for_missing(self, populated_store):
        """filing_exists returns False for nonexistent filings."""
        assert not await populated_store.filing_exists("0000000000-00-000000")
