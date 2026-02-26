"""Tests for edgar_sentinel.signals.builder — SignalBuilder + FilingDateMapping."""

from datetime import date, datetime

import pytest

from edgar_sentinel.analyzers.base import AnalysisResults
from edgar_sentinel.core import Signal
from edgar_sentinel.core.config import SignalsConfig
from edgar_sentinel.core.models import SentimentResult, SimilarityResult
from edgar_sentinel.signals.builder import FilingDateMapping, SignalBuilder
from edgar_sentinel.signals.decay import SignalDecay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 3, 15, 12, 0, 0)
AS_OF = date(2024, 3, 15)


def _sentiment(
    filing_id: str,
    section: str = "mda",
    analyzer: str = "dictionary",
    score: float = 0.1,
) -> SentimentResult:
    return SentimentResult(
        filing_id=filing_id,
        section_name=section,
        analyzer_name=analyzer,
        sentiment_score=score,
        confidence=0.8,
        analyzed_at=NOW,
    )


def _similarity(
    filing_id: str,
    prior_id: str = "0000000000-24-000000",
    section: str = "mda",
    similarity: float = 0.9,
) -> SimilarityResult:
    return SimilarityResult(
        filing_id=filing_id,
        prior_filing_id=prior_id,
        section_name=section,
        similarity_score=similarity,
        change_score=round(1.0 - similarity, 6),
        analyzed_at=NOW,
    )


def _mapping(
    filing_id: str,
    ticker: str = "AAPL",
    filed_date: date = date(2024, 3, 1),
) -> FilingDateMapping:
    return FilingDateMapping(
        ticker=ticker,
        filing_id=filing_id,
        filed_date=filed_date,
        signal_date=filed_date,  # caller computes but we set it for clarity
    )


def _results(
    sentiments: list[SentimentResult] | None = None,
    similarities: list[SimilarityResult] | None = None,
) -> AnalysisResults:
    return AnalysisResults(
        sentiment_results=sentiments or [],
        similarity_results=similarities or [],
    )


def _builder(buffer_days: int = 2, half_life: int = 90) -> SignalBuilder:
    config = SignalsConfig(buffer_days=buffer_days, decay_half_life=half_life)
    return SignalBuilder(config)


# ---------------------------------------------------------------------------
# FilingDateMapping
# ---------------------------------------------------------------------------


class TestFilingDateMapping:
    def test_creation(self):
        m = FilingDateMapping(
            ticker="AAPL",
            filing_id="0000000001-24-000001",
            filed_date=date(2024, 3, 1),
            signal_date=date(2024, 3, 3),
        )
        assert m.ticker == "AAPL"
        assert m.filing_id == "0000000001-24-000001"
        assert m.filed_date == date(2024, 3, 1)
        assert m.signal_date == date(2024, 3, 3)

    def test_frozen(self):
        m = FilingDateMapping(
            ticker="AAPL",
            filing_id="0000000001-24-000001",
            filed_date=date(2024, 3, 1),
            signal_date=date(2024, 3, 3),
        )
        with pytest.raises(Exception):
            m.ticker = "MSFT"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SignalBuilder — construction
# ---------------------------------------------------------------------------


class TestBuilderConstruction:
    def test_default_config(self):
        config = SignalsConfig()
        builder = SignalBuilder(config)
        assert builder._buffer_days == 2

    def test_custom_buffer(self):
        config = SignalsConfig(buffer_days=5)
        builder = SignalBuilder(config)
        assert builder._buffer_days == 5

    def test_custom_decay(self):
        config = SignalsConfig()
        custom_decay = SignalDecay(half_life_days=30)
        builder = SignalBuilder(config, decay=custom_decay)
        assert builder._decay.half_life_days == 30

    def test_default_decay_from_config(self):
        config = SignalsConfig(decay_half_life=45)
        builder = SignalBuilder(config)
        assert builder._decay.half_life_days == 45


# ---------------------------------------------------------------------------
# _compute_signal_date
# ---------------------------------------------------------------------------


class TestComputeSignalDate:
    def test_buffer_zero(self):
        builder = _builder(buffer_days=0)
        assert builder._compute_signal_date(date(2024, 3, 1)) == date(2024, 3, 1)

    def test_buffer_two(self):
        builder = _builder(buffer_days=2)
        assert builder._compute_signal_date(date(2024, 3, 1)) == date(2024, 3, 3)

    def test_buffer_across_month(self):
        builder = _builder(buffer_days=5)
        assert builder._compute_signal_date(date(2024, 1, 29)) == date(2024, 2, 3)


# ---------------------------------------------------------------------------
# _extract_raw_signals — sentiment
# ---------------------------------------------------------------------------


class TestExtractSentiment:
    def test_basic_extraction(self):
        builder = _builder(buffer_days=0)
        results = _results(sentiments=[_sentiment("0000000001-24-000001")])
        mapping = {"0000000001-24-000001": _mapping("0000000001-24-000001")}
        raw = builder._extract_raw_signals(results, mapping, AS_OF)
        assert len(raw) == 1
        assert raw[0].ticker == "AAPL"
        assert raw[0].signal_name == "dictionary_mda"
        assert raw[0].raw_value == 0.1

    def test_signal_name_convention(self):
        builder = _builder(buffer_days=0)
        sr = _sentiment("0000000001-24-000001", section="risk_factors", analyzer="llm")
        results = _results(sentiments=[sr])
        mapping = {"0000000001-24-000001": _mapping("0000000001-24-000001")}
        raw = builder._extract_raw_signals(results, mapping, AS_OF)
        assert raw[0].signal_name == "llm_risk_factors"

    def test_missing_mapping_skipped(self):
        builder = _builder(buffer_days=0)
        results = _results(sentiments=[_sentiment("0000000001-24-000001")])
        raw = builder._extract_raw_signals(results, {}, AS_OF)
        assert len(raw) == 0

    def test_future_signal_excluded(self):
        builder = _builder(buffer_days=10)
        results = _results(sentiments=[_sentiment("0000000001-24-000001")])
        # filed 2024-03-10, buffer 10 -> signal_date 2024-03-20, as_of 2024-03-15
        mapping = {
            "0000000001-24-000001": _mapping(
                "0000000001-24-000001", filed_date=date(2024, 3, 10)
            )
        }
        raw = builder._extract_raw_signals(results, mapping, AS_OF)
        assert len(raw) == 0

    def test_multiple_sentiments(self):
        builder = _builder(buffer_days=0)
        sentiments = [
            _sentiment("0000000001-24-000001", score=0.2),
            _sentiment("0000000001-24-000002", score=-0.3),
        ]
        results = _results(sentiments=sentiments)
        mapping = {
            "0000000001-24-000001": _mapping("0000000001-24-000001", ticker="AAPL"),
            "0000000001-24-000002": _mapping("0000000001-24-000002", ticker="MSFT"),
        }
        raw = builder._extract_raw_signals(results, mapping, AS_OF)
        assert len(raw) == 2
        tickers = {s.ticker for s in raw}
        assert tickers == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# _extract_raw_signals — similarity
# ---------------------------------------------------------------------------


class TestExtractSimilarity:
    def test_basic_extraction(self):
        builder = _builder(buffer_days=0)
        results = _results(similarities=[_similarity("0000000001-24-000001")])
        mapping = {"0000000001-24-000001": _mapping("0000000001-24-000001")}
        raw = builder._extract_raw_signals(results, mapping, AS_OF)
        assert len(raw) == 1
        assert raw[0].signal_name == "similarity_mda"
        assert raw[0].raw_value == 0.1  # change_score = 1 - 0.9

    def test_uses_change_score_not_similarity(self):
        builder = _builder(buffer_days=0)
        sim = _similarity("0000000001-24-000001", similarity=0.7)
        results = _results(similarities=[sim])
        mapping = {"0000000001-24-000001": _mapping("0000000001-24-000001")}
        raw = builder._extract_raw_signals(results, mapping, AS_OF)
        assert raw[0].raw_value == pytest.approx(0.3, abs=1e-5)

    def test_missing_mapping_skipped(self):
        builder = _builder(buffer_days=0)
        results = _results(similarities=[_similarity("0000000001-24-000001")])
        raw = builder._extract_raw_signals(results, {}, AS_OF)
        assert len(raw) == 0


# ---------------------------------------------------------------------------
# _extract_raw_signals — mixed
# ---------------------------------------------------------------------------


class TestExtractMixed:
    def test_sentiment_and_similarity_combined(self):
        builder = _builder(buffer_days=0)
        results = _results(
            sentiments=[_sentiment("0000000001-24-000001")],
            similarities=[_similarity("0000000001-24-000001")],
        )
        mapping = {"0000000001-24-000001": _mapping("0000000001-24-000001")}
        raw = builder._extract_raw_signals(results, mapping, AS_OF)
        assert len(raw) == 2
        names = {s.signal_name for s in raw}
        assert names == {"dictionary_mda", "similarity_mda"}

    def test_empty_results(self):
        builder = _builder(buffer_days=0)
        results = _results()
        raw = builder._extract_raw_signals(results, {}, AS_OF)
        assert raw == []


# ---------------------------------------------------------------------------
# _normalize_cross_section
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_skip_normalization_for_single_signal(self):
        builder = _builder()
        signal = Signal(
            ticker="AAPL",
            signal_date=date(2024, 3, 3),
            signal_name="dictionary_mda",
            raw_value=0.1,
        )
        result = builder._normalize_cross_section([signal])
        assert len(result) == 1
        assert result[0].z_score is None
        assert result[0].percentile is None

    def test_skip_normalization_for_two_signals(self):
        builder = _builder()
        signals = [
            Signal(
                ticker="AAPL",
                signal_date=date(2024, 3, 3),
                signal_name="dictionary_mda",
                raw_value=0.1,
            ),
            Signal(
                ticker="MSFT",
                signal_date=date(2024, 3, 3),
                signal_name="dictionary_mda",
                raw_value=0.2,
            ),
        ]
        result = builder._normalize_cross_section(signals)
        assert len(result) == 2
        assert result[0].z_score is None

    def test_normalization_with_three_signals(self):
        builder = _builder()
        signals = [
            Signal(
                ticker="AAPL",
                signal_date=date(2024, 3, 3),
                signal_name="dictionary_mda",
                raw_value=0.1,
            ),
            Signal(
                ticker="MSFT",
                signal_date=date(2024, 3, 3),
                signal_name="dictionary_mda",
                raw_value=0.2,
            ),
            Signal(
                ticker="GOOG",
                signal_date=date(2024, 3, 3),
                signal_name="dictionary_mda",
                raw_value=0.3,
            ),
        ]
        result = builder._normalize_cross_section(signals)
        assert len(result) == 3

        # All should have z_score and percentile
        for s in result:
            assert s.z_score is not None
            assert s.percentile is not None

        # Z-scores should sum to ~0 for centered data
        z_sum = sum(s.z_score for s in result)
        assert abs(z_sum) < 1e-5

        # Highest raw_value should have highest z_score
        assert result[2].z_score > result[0].z_score

    def test_identical_values_zero_std(self):
        builder = _builder()
        signals = [
            Signal(
                ticker=t,
                signal_date=date(2024, 3, 3),
                signal_name="dictionary_mda",
                raw_value=0.5,
            )
            for t in ["AAPL", "MSFT", "GOOG"]
        ]
        result = builder._normalize_cross_section(signals)
        for s in result:
            assert s.z_score == 0.0
            assert s.percentile == 50.0

    def test_percentile_ordering(self):
        builder = _builder()
        signals = [
            Signal(
                ticker="LOW",
                signal_date=date(2024, 3, 3),
                signal_name="dictionary_mda",
                raw_value=-0.5,
            ),
            Signal(
                ticker="MID",
                signal_date=date(2024, 3, 3),
                signal_name="dictionary_mda",
                raw_value=0.0,
            ),
            Signal(
                ticker="HIGH",
                signal_date=date(2024, 3, 3),
                signal_name="dictionary_mda",
                raw_value=0.5,
            ),
        ]
        result = builder._normalize_cross_section(signals)
        assert result[0].percentile < result[1].percentile < result[2].percentile

    def test_preserves_ticker_and_date(self):
        builder = _builder()
        signals = [
            Signal(
                ticker=t,
                signal_date=date(2024, 3, 3),
                signal_name="dictionary_mda",
                raw_value=v,
            )
            for t, v in [("AAPL", 0.1), ("MSFT", 0.2), ("GOOG", 0.3)]
        ]
        result = builder._normalize_cross_section(signals)
        for i, s in enumerate(result):
            assert s.ticker == signals[i].ticker
            assert s.signal_date == date(2024, 3, 3)
            assert s.raw_value == signals[i].raw_value


# ---------------------------------------------------------------------------
# build (integration)
# ---------------------------------------------------------------------------


class TestBuild:
    def test_empty_results_returns_empty(self):
        builder = _builder(buffer_days=0)
        result = builder.build(_results(), {}, AS_OF)
        assert result == []

    def test_basic_build(self):
        builder = _builder(buffer_days=0)
        sentiments = [
            _sentiment("0000000001-24-000001", score=0.1),
            _sentiment("0000000001-24-000002", score=0.2),
            _sentiment("0000000001-24-000003", score=0.3),
        ]
        results = _results(sentiments=sentiments)
        mapping = {
            "0000000001-24-000001": _mapping(
                "0000000001-24-000001", ticker="AAPL", filed_date=date(2024, 3, 1)
            ),
            "0000000001-24-000002": _mapping(
                "0000000001-24-000002", ticker="MSFT", filed_date=date(2024, 3, 1)
            ),
            "0000000001-24-000003": _mapping(
                "0000000001-24-000003", ticker="GOOG", filed_date=date(2024, 3, 1)
            ),
        }
        signals = builder.build(results, mapping, AS_OF)
        assert len(signals) == 3
        # All normalized (3+ in cohort)
        for s in signals:
            assert s.z_score is not None
            assert s.percentile is not None

    def test_build_applies_decay(self):
        builder = _builder(buffer_days=0, half_life=90)
        # Filing from 90 days ago -> decay = 0.5
        old_date = date(2023, 12, 16)  # ~90 days before 2024-03-15
        sentiments = [
            _sentiment("0000000001-24-000001", score=0.1),
            _sentiment("0000000001-24-000002", score=0.2),
            _sentiment("0000000001-24-000003", score=0.3),
        ]
        results = _results(sentiments=sentiments)
        mapping = {
            "0000000001-24-000001": _mapping(
                "0000000001-24-000001", ticker="AAPL", filed_date=old_date
            ),
            "0000000001-24-000002": _mapping(
                "0000000001-24-000002", ticker="MSFT", filed_date=old_date
            ),
            "0000000001-24-000003": _mapping(
                "0000000001-24-000003", ticker="GOOG", filed_date=old_date
            ),
        }
        signals = builder.build(results, mapping, AS_OF)
        # All should have decay < 1.0
        for s in signals:
            assert s.decay_weight < 1.0

    def test_build_with_mixed_results(self):
        builder = _builder(buffer_days=0)
        sentiments = [
            _sentiment("0000000001-24-000001", score=0.1),
            _sentiment("0000000001-24-000002", score=0.2),
            _sentiment("0000000001-24-000003", score=0.3),
        ]
        similarities = [
            _similarity("0000000001-24-000001", similarity=0.9),
            _similarity("0000000001-24-000002", similarity=0.8),
            _similarity("0000000001-24-000003", similarity=0.7),
        ]
        results = _results(sentiments=sentiments, similarities=similarities)
        mapping = {
            "0000000001-24-000001": _mapping(
                "0000000001-24-000001", ticker="AAPL", filed_date=date(2024, 3, 1)
            ),
            "0000000001-24-000002": _mapping(
                "0000000001-24-000002", ticker="MSFT", filed_date=date(2024, 3, 1)
            ),
            "0000000001-24-000003": _mapping(
                "0000000001-24-000003", ticker="GOOG", filed_date=date(2024, 3, 1)
            ),
        }
        signals = builder.build(results, mapping, AS_OF)
        assert len(signals) == 6  # 3 sentiment + 3 similarity
        names = {s.signal_name for s in signals}
        assert "dictionary_mda" in names
        assert "similarity_mda" in names

    def test_build_excludes_future_signals(self):
        builder = _builder(buffer_days=5)
        sentiments = [_sentiment("0000000001-24-000001")]
        results = _results(sentiments=sentiments)
        # Filed 2024-03-13, buffer 5 -> signal_date 2024-03-18, as_of 2024-03-15
        mapping = {
            "0000000001-24-000001": _mapping(
                "0000000001-24-000001", filed_date=date(2024, 3, 13)
            )
        }
        signals = builder.build(results, mapping, AS_OF)
        assert len(signals) == 0

    def test_build_frozen_signals(self):
        builder = _builder(buffer_days=0)
        sentiments = [
            _sentiment(f"000000000{i}-24-000001", score=v)
            for i, v in enumerate([0.1, 0.2, 0.3], 1)
        ]
        results = _results(sentiments=sentiments)
        mapping = {
            f"000000000{i}-24-000001": _mapping(
                f"000000000{i}-24-000001", ticker=t, filed_date=date(2024, 3, 1)
            )
            for i, t in enumerate(["AAPL", "MSFT", "GOOG"], 1)
        }
        signals = builder.build(results, mapping, AS_OF)
        for s in signals:
            with pytest.raises(Exception):
                s.raw_value = 999.0  # type: ignore[misc]

    def test_build_small_cohort_no_normalization(self):
        builder = _builder(buffer_days=0)
        sentiments = [_sentiment("0000000001-24-000001", score=0.5)]
        results = _results(sentiments=sentiments)
        mapping = {
            "0000000001-24-000001": _mapping(
                "0000000001-24-000001", filed_date=date(2024, 3, 1)
            )
        }
        signals = builder.build(results, mapping, AS_OF)
        assert len(signals) == 1
        assert signals[0].z_score is None
        assert signals[0].percentile is None


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_import_from_signals_package(self):
        from edgar_sentinel.signals import FilingDateMapping, SignalBuilder, SignalDecay

        assert SignalBuilder is not None
        assert SignalDecay is not None
        assert FilingDateMapping is not None

    def test_builder_in_all(self):
        import edgar_sentinel.signals as signals_mod

        assert "SignalBuilder" in signals_mod.__all__
        assert "SignalDecay" in signals_mod.__all__
        assert "FilingDateMapping" in signals_mod.__all__
