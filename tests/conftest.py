"""Shared pytest fixtures for edgar-sentinel."""

import pytest
from datetime import date, datetime

from edgar_sentinel.core.models import (
    FilingMetadata,
    FilingSection,
    Filing,
    FormType,
    SentimentResult,
    SimilarityResult,
    Signal,
    CompositeSignal,
    BacktestConfig,
    MonthlyReturn,
    BacktestResult,
)


@pytest.fixture
def sample_filing_metadata() -> FilingMetadata:
    return FilingMetadata(
        cik="320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type=FormType.FORM_10K,
        filed_date=date(2023, 11, 3),
        accession_number="0000320193-23-000106",
        fiscal_year_end=date(2023, 9, 30),
        url="https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106-index.htm",
    )


@pytest.fixture
def sample_filing_section() -> FilingSection:
    text = "The Company designs manufactures and markets smartphones."
    return FilingSection(
        filing_id="0000320193-23-000106",
        section_name="mda",
        raw_text=text,
        word_count=len(text.split()),
        extracted_at=datetime(2024, 1, 15, 10, 30, 0),
    )


@pytest.fixture
def sample_filing(sample_filing_metadata, sample_filing_section) -> Filing:
    return Filing(
        metadata=sample_filing_metadata,
        sections={"mda": sample_filing_section},
    )


@pytest.fixture
def sample_sentiment_result() -> SentimentResult:
    return SentimentResult(
        filing_id="0000320193-23-000106",
        section_name="mda",
        analyzer_name="dictionary",
        sentiment_score=0.15,
        confidence=0.85,
        metadata={"positive_words": 42, "negative_words": 12},
        analyzed_at=datetime(2024, 1, 15, 11, 0, 0),
    )


@pytest.fixture
def sample_similarity_result() -> SimilarityResult:
    return SimilarityResult(
        filing_id="0000320193-23-000106",
        prior_filing_id="0000320193-22-000108",
        section_name="mda",
        similarity_score=0.82,
        change_score=0.18,
        analyzed_at=datetime(2024, 1, 15, 11, 0, 0),
    )


@pytest.fixture
def sample_signal() -> Signal:
    return Signal(
        ticker="AAPL",
        signal_date=date(2023, 11, 5),
        signal_name="dictionary_mda",
        raw_value=0.15,
        z_score=1.2,
        percentile=88.5,
        decay_weight=0.95,
    )


@pytest.fixture
def sample_composite_signal() -> CompositeSignal:
    return CompositeSignal(
        ticker="AAPL",
        signal_date=date(2023, 11, 5),
        composite_score=0.72,
        components={"dictionary_mda": 0.5, "similarity_mda": 0.5},
        rank=1,
    )


@pytest.fixture
def sample_backtest_config() -> BacktestConfig:
    return BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
        universe=["AAPL", "MSFT", "GOOGL"],
    )
