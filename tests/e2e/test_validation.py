"""End-to-end validation tests.

These tests hit real EDGAR servers and validate the complete pipeline:
1. Fetch filing metadata from SEC EDGAR
2. Download and parse actual filings
3. Run dictionary and similarity analysis
4. Build signals
5. Verify output integrity

Run with: pytest tests/e2e/ -m e2e -v
Skip with: pytest -m "not e2e"
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from edgar_sentinel.analyzers.base import AnalysisResults
from edgar_sentinel.analyzers.dictionary import DictionaryAnalyzer
from edgar_sentinel.analyzers.similarity import SimilarityAnalyzer
from edgar_sentinel.core.config import SentinelConfig
from edgar_sentinel.core.models import FormType
from edgar_sentinel.ingestion.client import EdgarClient
from edgar_sentinel.ingestion.parser import FilingParser
from edgar_sentinel.ingestion.store import SqliteStore
from edgar_sentinel.signals.builder import FilingDateMapping, SignalBuilder
from edgar_sentinel.signals.composite import SignalComposite


@pytest.mark.e2e
class TestEdgarIngestion:
    """Validate real EDGAR data fetching and parsing."""

    @pytest.mark.slow
    async def test_fetch_apple_10k_filing_index(self, e2e_config: SentinelConfig):
        """Fetch Apple's filing index from EDGAR."""
        async with EdgarClient(e2e_config.edgar) as client:
            filings = await client.get_filings_for_ticker(
                ticker="AAPL",
                form_types=[FormType.FORM_10K],
                start_date=date(2023, 1, 1),
                end_date=date(2024, 12, 31),
            )

        assert len(filings) >= 1
        filing = filings[0]
        assert filing.ticker == "AAPL"
        assert filing.form_type == FormType.FORM_10K
        assert filing.cik.lstrip("0") == "320193"

    @pytest.mark.slow
    async def test_parse_real_filing_sections(self, e2e_config: SentinelConfig):
        """Fetch and parse a real 10-K filing, extracting MD&A and Risk Factors."""
        async with EdgarClient(e2e_config.edgar) as client:
            filings = await client.get_filings_for_ticker(
                ticker="AAPL",
                form_types=[FormType.FORM_10K],
                start_date=date(2023, 1, 1),
                end_date=date(2024, 12, 31),
            )
            assert len(filings) >= 1

            raw_html = await client.get_filing_document(filings[0].url)

        parser = FilingParser()
        sections = parser.parse(raw_html, FormType.FORM_10K, filings[0].accession_number)

        # Should extract at least one section from a real 10-K
        assert len(sections) > 0

        # Check section content is substantial
        for name, section in sections.items():
            assert section.word_count > 50, f"Section {name} too short ({section.word_count} words)"


@pytest.mark.e2e
class TestFullPipeline:
    """End-to-end pipeline validation with real EDGAR data."""

    @pytest.mark.slow
    async def test_mini_pipeline_single_ticker(
        self, e2e_config: SentinelConfig, e2e_workspace
    ):
        """Complete pipeline for one ticker, one year.

        Steps:
        1. Fetch filing metadata from EDGAR
        2. Download and parse the filing
        3. Run dictionary analysis
        4. Build signals
        5. Verify output structure
        """
        # 1. Ingest
        async with EdgarClient(e2e_config.edgar) as client:
            filings = await client.get_filings_for_ticker(
                ticker="MSFT",
                form_types=[FormType.FORM_10K],
                start_date=date(2023, 1, 1),
                end_date=date(2024, 12, 31),
            )
            assert len(filings) >= 1
            filing_meta = filings[0]

            raw_html = await client.get_filing_document(filing_meta.url)

        # 2. Parse
        parser = FilingParser()
        sections = parser.parse(
            raw_html, filing_meta.form_type, filing_meta.accession_number
        )

        if not sections:
            pytest.skip("Parser could not extract sections from this filing format")

        # 3. Analyze — dictionary
        dict_analyzer = DictionaryAnalyzer(e2e_config.analyzers.dictionary)
        sentiment_results = []
        for section in sections.values():
            result = dict_analyzer.analyze(section)
            sentiment_results.append(result)

        assert len(sentiment_results) > 0
        for r in sentiment_results:
            assert -1.0 <= r.sentiment_score <= 1.0
            assert 0.0 <= r.confidence <= 1.0

        # 4. Build signals
        filing_dates = {
            filing_meta.accession_number: FilingDateMapping(
                ticker="MSFT",
                filing_id=filing_meta.accession_number,
                filed_date=filing_meta.filed_date,
                signal_date=filing_meta.filed_date + timedelta(days=2),
            )
        }

        analysis = AnalysisResults(
            sentiment_results=sentiment_results,
            similarity_results=[],
        )
        builder = SignalBuilder(e2e_config.signals)
        signals = builder.build(
            results=analysis,
            filing_dates=filing_dates,
            as_of_date=date.today(),
        )

        # With only one ticker, cross-sectional normalization produces z_score=0
        # unless there are multiple signals of the same type.
        # Signals should still be structurally valid.
        for s in signals:
            assert s.ticker == "MSFT"
            assert s.signal_name  # non-empty
            assert s.raw_value is not None

    @pytest.mark.slow
    async def test_store_roundtrip(self, e2e_config: SentinelConfig):
        """Verify filing metadata can be stored and retrieved."""
        async with EdgarClient(e2e_config.edgar) as client:
            filings = await client.get_filings_for_ticker(
                ticker="AAPL",
                form_types=[FormType.FORM_10K],
                start_date=date(2023, 1, 1),
                end_date=date(2024, 12, 31),
            )
            assert len(filings) >= 1
            filing_meta = filings[0]

            raw_html = await client.get_filing_document(filing_meta.url)

        parser = FilingParser()
        sections = parser.parse(
            raw_html, filing_meta.form_type, filing_meta.accession_number
        )

        if not sections:
            pytest.skip("Parser could not extract sections from this filing format")

        from edgar_sentinel.core.models import Filing

        filing = Filing(metadata=filing_meta, sections=sections)

        store = SqliteStore(e2e_config.storage)
        await store.initialize()
        try:
            await store.save_filing(filing)
            retrieved = await store.get_filing(filing_meta.accession_number)

            assert retrieved is not None
            assert retrieved.metadata.ticker == "AAPL"
            assert retrieved.metadata.accession_number == filing_meta.accession_number
            assert len(retrieved.sections) == len(sections)
        finally:
            await store.close()

    @pytest.mark.slow
    async def test_similarity_across_years(self, e2e_config: SentinelConfig):
        """Validate similarity analysis between consecutive annual filings."""
        async with EdgarClient(e2e_config.edgar) as client:
            filings = await client.get_filings_for_ticker(
                ticker="AAPL",
                form_types=[FormType.FORM_10K],
                start_date=date(2022, 1, 1),
                end_date=date(2024, 12, 31),
            )

        if len(filings) < 2:
            pytest.skip("Need at least 2 annual filings for similarity analysis")

        # Fetch and parse both filings
        parser = FilingParser()
        parsed_filings = []
        async with EdgarClient(e2e_config.edgar) as client:
            for meta in filings[:2]:
                raw_html = await client.get_filing_document(meta.url)
                sections = parser.parse(raw_html, meta.form_type, meta.accession_number)
                parsed_filings.append((meta, sections))

        # Find a common section between the two filings
        common_sections = set(parsed_filings[0][1].keys()) & set(
            parsed_filings[1][1].keys()
        )
        if not common_sections:
            pytest.skip("No common sections found between filings")

        section_name = next(iter(common_sections))
        current_section = parsed_filings[0][1][section_name]
        prior_section = parsed_filings[1][1][section_name]

        # Run similarity analysis
        sim_analyzer = SimilarityAnalyzer(e2e_config.analyzers.similarity)
        result = sim_analyzer.analyze(current_section, prior_section)

        assert 0.0 <= result.similarity_score <= 1.0
        assert 0.0 <= result.change_score <= 1.0
        # Consecutive Apple 10-Ks should have reasonable similarity
        assert result.similarity_score > 0.1, (
            f"Suspiciously low similarity ({result.similarity_score}) "
            "between consecutive AAPL 10-K filings"
        )


@pytest.mark.e2e
class TestTickerResolution:
    """Validate ticker/CIK resolution against live EDGAR data."""

    @pytest.mark.slow
    async def test_resolve_major_tickers(self, e2e_config: SentinelConfig):
        """Verify well-known tickers resolve correctly."""
        expected = {
            "AAPL": "320193",
            "MSFT": "789019",
            "GOOGL": "1652044",
        }
        async with EdgarClient(e2e_config.edgar) as client:
            for ticker, expected_cik in expected.items():
                cik = await client.resolve_ticker(ticker)
                # CIK is zero-padded to 10 digits
                assert cik.lstrip("0") == expected_cik, (
                    f"{ticker} resolved to CIK {cik}, expected {expected_cik}"
                )
