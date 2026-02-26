"""Tests for edgar_sentinel.core.models."""

import pytest
from datetime import date, datetime

from pydantic import ValidationError

from edgar_sentinel.core.models import (
    BacktestConfig,
    BacktestResult,
    CompositeSignal,
    Filing,
    FilingMetadata,
    FilingSection,
    FormType,
    MonthlyReturn,
    RebalanceFrequency,
    SentimentResult,
    Signal,
    SimilarityResult,
)


class TestFilingMetadata:
    def test_valid_construction(self, sample_filing_metadata):
        assert sample_filing_metadata.cik == "0000320193"
        assert sample_filing_metadata.ticker == "AAPL"
        assert sample_filing_metadata.form_type == FormType.FORM_10K

    def test_cik_normalization(self):
        m = FilingMetadata(
            cik="320193",
            company_name="Apple Inc.",
            form_type=FormType.FORM_10K,
            filed_date=date(2023, 11, 3),
            accession_number="0000320193-23-000106",
            url="https://sec.gov/test",
        )
        assert m.cik == "0000320193"

    def test_invalid_cik_rejected(self):
        with pytest.raises(ValidationError, match="CIK must be numeric"):
            FilingMetadata(
                cik="abc",
                company_name="Test",
                form_type=FormType.FORM_10K,
                filed_date=date(2023, 1, 1),
                accession_number="0000320193-23-000106",
                url="https://sec.gov/test",
            )

    def test_accession_format_normalization(self):
        m = FilingMetadata(
            cik="320193",
            company_name="Apple",
            form_type=FormType.FORM_10K,
            filed_date=date(2023, 11, 3),
            accession_number="000032019323000106",
            url="https://sec.gov/test",
        )
        assert m.accession_number == "0000320193-23-000106"

    def test_invalid_accession_rejected(self):
        with pytest.raises(ValidationError, match="Invalid accession number"):
            FilingMetadata(
                cik="320193",
                company_name="Test",
                form_type=FormType.FORM_10K,
                filed_date=date(2023, 1, 1),
                accession_number="short",
                url="https://sec.gov/test",
            )

    def test_url_must_be_https(self):
        with pytest.raises(ValidationError, match="HTTPS"):
            FilingMetadata(
                cik="320193",
                company_name="Test",
                form_type=FormType.FORM_10K,
                filed_date=date(2023, 1, 1),
                accession_number="0000320193-23-000106",
                url="http://sec.gov/test",
            )

    def test_filing_id_property(self, sample_filing_metadata):
        assert sample_filing_metadata.filing_id == sample_filing_metadata.accession_number

    def test_frozen(self, sample_filing_metadata):
        with pytest.raises(ValidationError):
            sample_filing_metadata.cik = "999"


class TestFilingSection:
    def test_valid_construction(self, sample_filing_section):
        assert sample_filing_section.section_name == "mda"
        assert sample_filing_section.word_count > 0

    def test_negative_word_count_rejected(self):
        with pytest.raises(ValidationError, match="cannot be negative"):
            FilingSection(
                filing_id="0000320193-23-000106",
                section_name="mda",
                raw_text="hello world",
                word_count=-1,
                extracted_at=datetime(2024, 1, 1),
            )

    def test_word_count_mismatch_warns(self):
        with pytest.warns(UserWarning, match="differs from actual"):
            FilingSection(
                filing_id="0000320193-23-000106",
                section_name="mda",
                raw_text="one two three four five",
                word_count=50,
                extracted_at=datetime(2024, 1, 1),
            )


class TestFiling:
    def test_valid_construction(self, sample_filing):
        assert sample_filing.filing_id == "0000320193-23-000106"
        assert sample_filing.ticker == "AAPL"

    def test_get_section(self, sample_filing):
        section = sample_filing.get_section("mda")
        assert section is not None
        assert section.section_name == "mda"

    def test_get_missing_section(self, sample_filing):
        assert sample_filing.get_section("risk_factors") is None

    def test_section_names(self, sample_filing):
        assert sample_filing.section_names() == ["mda"]


class TestSentimentResult:
    def test_valid_construction(self, sample_sentiment_result):
        assert sample_sentiment_result.sentiment_score == 0.15
        assert sample_sentiment_result.confidence == 0.85

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError, match="sentiment_score"):
            SentimentResult(
                filing_id="0000320193-23-000106",
                section_name="mda",
                analyzer_name="dictionary",
                sentiment_score=1.5,
                confidence=0.5,
                analyzed_at=datetime(2024, 1, 1),
            )

    def test_negative_score_out_of_range(self):
        with pytest.raises(ValidationError, match="sentiment_score"):
            SentimentResult(
                filing_id="0000320193-23-000106",
                section_name="mda",
                analyzer_name="dictionary",
                sentiment_score=-1.5,
                confidence=0.5,
                analyzed_at=datetime(2024, 1, 1),
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError, match="confidence"):
            SentimentResult(
                filing_id="0000320193-23-000106",
                section_name="mda",
                analyzer_name="dictionary",
                sentiment_score=0.1,
                confidence=1.5,
                analyzed_at=datetime(2024, 1, 1),
            )


class TestSimilarityResult:
    def test_valid_construction(self, sample_similarity_result):
        assert sample_similarity_result.similarity_score == 0.82
        assert sample_similarity_result.change_score == pytest.approx(0.18, abs=1e-5)

    def test_similarity_out_of_range(self):
        with pytest.raises(ValidationError, match="similarity_score"):
            SimilarityResult(
                filing_id="0000320193-23-000106",
                prior_filing_id="0000320193-22-000108",
                section_name="mda",
                similarity_score=1.5,
                change_score=-0.5,
                analyzed_at=datetime(2024, 1, 1),
            )

    def test_change_must_be_complement(self):
        with pytest.raises(ValidationError, match="change_score"):
            SimilarityResult(
                filing_id="0000320193-23-000106",
                prior_filing_id="0000320193-22-000108",
                section_name="mda",
                similarity_score=0.8,
                change_score=0.5,
                analyzed_at=datetime(2024, 1, 1),
            )


class TestSignal:
    def test_valid_construction(self, sample_signal):
        assert sample_signal.ticker == "AAPL"
        assert sample_signal.percentile == 88.5

    def test_percentile_out_of_range(self):
        with pytest.raises(ValidationError, match="percentile"):
            Signal(
                ticker="AAPL",
                signal_date=date(2023, 11, 5),
                signal_name="test",
                raw_value=0.5,
                percentile=101.0,
            )

    def test_decay_out_of_range(self):
        with pytest.raises(ValidationError, match="decay_weight"):
            Signal(
                ticker="AAPL",
                signal_date=date(2023, 11, 5),
                signal_name="test",
                raw_value=0.5,
                decay_weight=1.5,
            )


class TestCompositeSignal:
    def test_valid_construction(self, sample_composite_signal):
        assert sample_composite_signal.rank == 1
        assert len(sample_composite_signal.components) == 2

    def test_rank_must_be_positive(self):
        with pytest.raises(ValidationError, match="rank"):
            CompositeSignal(
                ticker="AAPL",
                signal_date=date(2023, 11, 5),
                composite_score=0.7,
                components={"test": 1.0},
                rank=0,
            )


class TestBacktestConfig:
    def test_valid_construction(self, sample_backtest_config):
        assert len(sample_backtest_config.universe) == 3

    def test_dates_must_be_ordered(self):
        with pytest.raises(ValidationError, match="start_date must be before"):
            BacktestConfig(
                start_date=date(2025, 1, 1),
                end_date=date(2020, 1, 1),
                universe=["AAPL"],
            )

    def test_universe_not_empty(self):
        with pytest.raises(ValidationError, match="at least one ticker"):
            BacktestConfig(
                start_date=date(2020, 1, 1),
                end_date=date(2025, 1, 1),
                universe=[],
            )

    def test_quantile_bounds(self):
        with pytest.raises(ValidationError, match="long_quantile"):
            BacktestConfig(
                start_date=date(2020, 1, 1),
                end_date=date(2025, 1, 1),
                universe=["AAPL"],
                long_quantile=10,
                num_quantiles=5,
            )


class TestModelRoundtrip:
    """Verify model_dump -> model_validate identity for all models."""

    def test_filing_metadata_roundtrip(self, sample_filing_metadata):
        data = sample_filing_metadata.model_dump()
        restored = FilingMetadata.model_validate(data)
        assert restored == sample_filing_metadata

    def test_filing_section_roundtrip(self, sample_filing_section):
        data = sample_filing_section.model_dump()
        restored = FilingSection.model_validate(data)
        assert restored == sample_filing_section

    def test_filing_roundtrip(self, sample_filing):
        data = sample_filing.model_dump()
        restored = Filing.model_validate(data)
        assert restored == sample_filing

    def test_sentiment_result_roundtrip(self, sample_sentiment_result):
        data = sample_sentiment_result.model_dump()
        restored = SentimentResult.model_validate(data)
        assert restored == sample_sentiment_result

    def test_similarity_result_roundtrip(self, sample_similarity_result):
        data = sample_similarity_result.model_dump()
        restored = SimilarityResult.model_validate(data)
        assert restored == sample_similarity_result

    def test_signal_roundtrip(self, sample_signal):
        data = sample_signal.model_dump()
        restored = Signal.model_validate(data)
        assert restored == sample_signal

    def test_composite_signal_roundtrip(self, sample_composite_signal):
        data = sample_composite_signal.model_dump()
        restored = CompositeSignal.model_validate(data)
        assert restored == sample_composite_signal


class TestJsonRoundtrip:
    """Verify JSON serialization -> deserialization identity."""

    def test_filing_metadata_json(self, sample_filing_metadata):
        json_str = sample_filing_metadata.model_dump_json()
        restored = FilingMetadata.model_validate_json(json_str)
        assert restored == sample_filing_metadata

    def test_sentiment_result_json(self, sample_sentiment_result):
        json_str = sample_sentiment_result.model_dump_json()
        restored = SentimentResult.model_validate_json(json_str)
        assert restored == sample_sentiment_result

    def test_signal_json(self, sample_signal):
        json_str = sample_signal.model_dump_json()
        restored = Signal.model_validate_json(json_str)
        assert restored == sample_signal

    def test_backtest_config_json(self, sample_backtest_config):
        json_str = sample_backtest_config.model_dump_json()
        restored = BacktestConfig.model_validate_json(json_str)
        assert restored == sample_backtest_config


class TestEnumSerialization:
    """Verify StrEnum values serialize as strings, not names."""

    def test_form_type_serializes_as_value(self, sample_filing_metadata):
        data = sample_filing_metadata.model_dump()
        assert data["form_type"] == "10-K"

    def test_rebalance_frequency_serializes_as_value(self, sample_backtest_config):
        data = sample_backtest_config.model_dump()
        assert data["rebalance_frequency"] == "quarterly"
