"""Tests for the TF-IDF cosine similarity analyzer."""

from datetime import datetime, timezone

import pytest

from edgar_sentinel.core import AnalysisError, AnalyzerType, FilingSection, SimilarityResult
from edgar_sentinel.core.config import SimilarityAnalyzerConfig
from edgar_sentinel.analyzers.base import Analyzer
from edgar_sentinel.analyzers.similarity import (
    SimilarityAnalyzer,
    TFIDF_PARAMS,
    _compute_cosine_similarity,
)


NOW = datetime.now(timezone.utc)


def _make_section(
    filing_id: str = "0000000001-24-000001",
    section_name: str = "mda",
    text: str = "The company reported strong revenue growth across all segments.",
    word_count: int | None = None,
) -> FilingSection:
    """Create a FilingSection for testing."""
    if word_count is None:
        word_count = len(text.split())
    return FilingSection(
        filing_id=filing_id,
        section_name=section_name,
        raw_text=text,
        word_count=word_count,
        extracted_at=NOW,
    )


@pytest.fixture
def config():
    return SimilarityAnalyzerConfig(enabled=True)


@pytest.fixture
def disabled_config():
    return SimilarityAnalyzerConfig(enabled=False)


@pytest.fixture
def analyzer(config):
    return SimilarityAnalyzer(config)


# --- TFIDF_PARAMS Tests ---


class TestTfidfParams:
    def test_max_features(self):
        assert TFIDF_PARAMS["max_features"] == 10_000

    def test_min_df(self):
        assert TFIDF_PARAMS["min_df"] == 1

    def test_max_df(self):
        assert TFIDF_PARAMS["max_df"] == 1.0

    def test_sublinear_tf(self):
        assert TFIDF_PARAMS["sublinear_tf"] is True

    def test_ngram_range(self):
        assert TFIDF_PARAMS["ngram_range"] == (1, 2)

    def test_stop_words(self):
        assert TFIDF_PARAMS["stop_words"] == "english"

    def test_strip_accents(self):
        assert TFIDF_PARAMS["strip_accents"] == "unicode"

    def test_lowercase(self):
        assert TFIDF_PARAMS["lowercase"] is True

    def test_token_pattern_filters_single_chars(self):
        """Token pattern requires at least 2 alphabetic chars."""
        import re
        pattern = TFIDF_PARAMS["token_pattern"]
        assert re.match(pattern, "ab")  # 2 chars OK
        assert not re.match(pattern, "a")  # 1 char rejected
        assert not re.match(pattern, "1")  # digit rejected
        assert not re.match(pattern, "42")  # digits rejected


# --- _compute_cosine_similarity Tests ---


class TestComputeCosineSimilarity:
    def test_identical_texts(self):
        text = "The company achieved significant revenue growth in fiscal year"
        result = _compute_cosine_similarity(text, text)
        assert result == 1.0

    def test_completely_different_texts(self):
        text_a = "pharmaceutical clinical trials laboratory research molecular"
        text_b = "automotive manufacturing assembly transmission hydraulic"
        result = _compute_cosine_similarity(text_a, text_b)
        assert result < 0.1

    def test_similar_texts(self):
        text_a = "The company reported strong revenue growth and increased profit margins"
        text_b = "The company reported moderate revenue growth and stable profit margins"
        result = _compute_cosine_similarity(text_a, text_b)
        assert 0.3 < result < 1.0

    def test_result_clamped_to_unit_interval(self):
        text = "revenue growth profit margin operating income net earnings"
        result = _compute_cosine_similarity(text, text)
        assert 0.0 <= result <= 1.0

    def test_result_is_float(self):
        text_a = "revenue growth operating income"
        text_b = "debt obligations liabilities expenses"
        result = _compute_cosine_similarity(text_a, text_b)
        assert isinstance(result, float)

    def test_result_precision(self):
        """Result should be rounded to 6 decimal places."""
        text_a = "strong earnings revenue growth profit expansion"
        text_b = "moderate earnings revenue stability profit maintenance"
        result = _compute_cosine_similarity(text_a, text_b)
        # Check that rounding is applied
        assert result == round(result, 6)

    def test_all_stop_words_raises(self):
        """Texts containing only stop words should raise AnalysisError."""
        text_a = "the and is are was were"
        text_b = "this that these those been being"
        with pytest.raises(AnalysisError, match="TF-IDF vectorization failed"):
            _compute_cosine_similarity(text_a, text_b)

    def test_order_independence(self):
        """Cosine similarity is symmetric."""
        text_a = "quarterly revenue increased significantly year over year"
        text_b = "annual expenses decreased moderately compared previous period"
        result_ab = _compute_cosine_similarity(text_a, text_b)
        result_ba = _compute_cosine_similarity(text_b, text_a)
        assert result_ab == result_ba

    def test_long_texts(self):
        """Should handle reasonably long texts without error."""
        text_a = " ".join(["revenue growth profit margin"] * 200)
        text_b = " ".join(["revenue growth operating income"] * 200)
        result = _compute_cosine_similarity(text_a, text_b)
        assert 0.0 <= result <= 1.0

    def test_unicode_handling(self):
        """Unicode characters should be handled via strip_accents."""
        text_a = "résumé café naïve exposé"
        text_b = "resume cafe naive expose"
        result = _compute_cosine_similarity(text_a, text_b)
        # After stripping accents, these should be very similar
        assert result > 0.8


# --- SimilarityAnalyzer Properties Tests ---


class TestAnalyzerProperties:
    def test_name(self, analyzer):
        assert analyzer.name == "similarity"

    def test_analyzer_type(self, analyzer):
        assert analyzer.analyzer_type == AnalyzerType.SIMILARITY

    def test_conforms_to_protocol(self, analyzer):
        assert isinstance(analyzer, Analyzer)

    def test_config_stored(self, analyzer, config):
        assert analyzer._config is config


# --- SimilarityAnalyzer.analyze Tests ---


class TestAnalyze:
    def test_basic_similarity(self, analyzer):
        section = _make_section(
            filing_id="0000000001-24-000002",
            text="The company reported strong revenue growth and profit improvement",
        )
        prior = _make_section(
            filing_id="0000000001-23-000001",
            text="The company reported moderate revenue growth and stable margins",
        )
        result = analyzer.analyze(section, prior)
        assert isinstance(result, SimilarityResult)
        assert result.filing_id == "0000000001-24-000002"
        assert result.prior_filing_id == "0000000001-23-000001"
        assert result.section_name == "mda"
        assert 0.0 <= result.similarity_score <= 1.0
        assert 0.0 <= result.change_score <= 1.0

    def test_change_score_is_complement(self, analyzer):
        section = _make_section(
            filing_id="0000000001-24-000002",
            text="Material weakness identified in internal controls over financial reporting",
        )
        prior = _make_section(
            filing_id="0000000001-23-000001",
            text="Revenue increased substantially driven by international expansion",
        )
        result = analyzer.analyze(section, prior)
        expected_change = round(1.0 - result.similarity_score, 6)
        assert abs(result.change_score - expected_change) < 1e-5

    def test_identical_sections(self, analyzer):
        text = "The company achieved record revenue of fifty billion dollars this quarter"
        section = _make_section(filing_id="0000000001-24-000002", text=text)
        prior = _make_section(filing_id="0000000001-23-000001", text=text)
        result = analyzer.analyze(section, prior)
        assert result.similarity_score == 1.0
        assert result.change_score == 0.0

    def test_different_sections(self, analyzer):
        section = _make_section(
            filing_id="0000000001-24-000002",
            text="pharmaceutical clinical trials molecular research laboratory",
        )
        prior = _make_section(
            filing_id="0000000001-23-000001",
            text="automotive manufacturing assembly production hydraulic",
        )
        result = analyzer.analyze(section, prior)
        assert result.similarity_score < 0.1
        assert result.change_score > 0.9

    def test_no_prior_section_raises(self, analyzer):
        section = _make_section()
        with pytest.raises(AnalysisError, match="requires prior_section"):
            analyzer.analyze(section, None)

    def test_no_prior_section_error_context(self, analyzer):
        section = _make_section(filing_id="0000000001-24-000002")
        with pytest.raises(AnalysisError) as exc_info:
            analyzer.analyze(section, None)
        assert exc_info.value.context["analyzer_name"] == "similarity"
        assert exc_info.value.context["filing_id"] == "0000000001-24-000002"

    def test_empty_current_text_raises(self, analyzer):
        section = _make_section(text="", word_count=0)
        prior = _make_section(filing_id="0000000001-23-000001", text="valid text here")
        with pytest.raises(AnalysisError, match="Empty section text"):
            analyzer.analyze(section, prior)

    def test_empty_prior_text_raises(self, analyzer):
        section = _make_section(text="valid text here")
        prior = _make_section(filing_id="0000000001-23-000001", text="", word_count=0)
        with pytest.raises(AnalysisError, match="Empty section text"):
            analyzer.analyze(section, prior)

    def test_whitespace_only_current_text_raises(self, analyzer):
        section = _make_section(text="   \n\t  ", word_count=0)
        prior = _make_section(filing_id="0000000001-23-000001", text="valid text here")
        with pytest.raises(AnalysisError, match="Empty section text"):
            analyzer.analyze(section, prior)

    def test_whitespace_only_prior_text_raises(self, analyzer):
        section = _make_section(text="valid text here")
        prior = _make_section(filing_id="0000000001-23-000001", text="   \n\t  ", word_count=0)
        with pytest.raises(AnalysisError, match="Empty section text"):
            analyzer.analyze(section, prior)

    def test_all_stop_words_raises(self, analyzer):
        section = _make_section(text="the and is are was were")
        prior = _make_section(
            filing_id="0000000001-23-000001",
            text="this that these those been being",
        )
        with pytest.raises(AnalysisError, match="TF-IDF vectorization failed"):
            analyzer.analyze(section, prior)

    def test_result_has_analyzed_at(self, analyzer):
        section = _make_section(
            filing_id="0000000001-24-000002",
            text="revenue growth accelerated this quarter significantly",
        )
        prior = _make_section(
            filing_id="0000000001-23-000001",
            text="revenue growth decelerated last quarter substantially",
        )
        result = analyzer.analyze(section, prior)
        assert result.analyzed_at is not None
        assert result.analyzed_at.tzinfo is not None

    def test_section_name_preserved(self, analyzer):
        section = _make_section(
            filing_id="0000000001-24-000002",
            section_name="risk_factors",
            text="material weakness identified in financial controls",
        )
        prior = _make_section(
            filing_id="0000000001-23-000001",
            section_name="risk_factors",
            text="no material weaknesses identified during audit",
        )
        result = analyzer.analyze(section, prior)
        assert result.section_name == "risk_factors"

    def test_result_is_frozen(self, analyzer):
        section = _make_section(
            filing_id="0000000001-24-000002",
            text="revenue growth profit margin expansion",
        )
        prior = _make_section(
            filing_id="0000000001-23-000001",
            text="revenue decline margin compression contraction",
        )
        result = analyzer.analyze(section, prior)
        with pytest.raises(Exception):
            result.similarity_score = 0.5


# --- SimilarityAnalyzer.analyze_batch Tests ---


class TestAnalyzeBatch:
    def test_batch_multiple_pairs(self, analyzer):
        pairs = [
            (
                _make_section(filing_id="0000000001-24-000002", text="strong revenue growth"),
                _make_section(filing_id="0000000001-23-000001", text="moderate revenue growth"),
            ),
            (
                _make_section(filing_id="0000000002-24-000003", text="declining profit margins"),
                _make_section(filing_id="0000000002-23-000002", text="improving profit margins"),
            ),
        ]
        results = analyzer.analyze_batch(pairs)
        assert len(results) == 2
        assert all(isinstance(r, SimilarityResult) for r in results)

    def test_batch_skips_none_prior(self, analyzer):
        pairs = [
            (
                _make_section(filing_id="0000000001-24-000002", text="strong revenue growth"),
                None,
            ),
            (
                _make_section(filing_id="0000000002-24-000003", text="declining profit margins"),
                _make_section(filing_id="0000000002-23-000002", text="improving profit margins"),
            ),
        ]
        results = analyzer.analyze_batch(pairs)
        assert len(results) == 1
        assert results[0].filing_id == "0000000002-24-000003"

    def test_batch_all_none_prior(self, analyzer):
        pairs = [
            (_make_section(text="some text here"), None),
            (_make_section(filing_id="0000000002-24-000002", text="more text here"), None),
        ]
        results = analyzer.analyze_batch(pairs)
        assert len(results) == 0

    def test_batch_empty(self, analyzer):
        results = analyzer.analyze_batch([])
        assert len(results) == 0

    def test_batch_handles_errors_gracefully(self, analyzer):
        """Batch should continue past analysis errors (e.g., all stop words)."""
        pairs = [
            (
                _make_section(filing_id="0000000001-24-000002", text="the and is are was were"),
                _make_section(filing_id="0000000001-23-000001", text="this that these those been being"),
            ),
            (
                _make_section(filing_id="0000000002-24-000003", text="strong revenue growth profit"),
                _make_section(filing_id="0000000002-23-000002", text="moderate revenue growth expansion"),
            ),
        ]
        results = analyzer.analyze_batch(pairs)
        assert len(results) == 1
        assert results[0].filing_id == "0000000002-24-000003"

    def test_batch_preserves_order(self, analyzer):
        pairs = [
            (
                _make_section(filing_id="0000000001-24-000002", text="quarterly revenue increased"),
                _make_section(filing_id="0000000001-23-000001", text="quarterly revenue decreased"),
            ),
            (
                _make_section(filing_id="0000000002-24-000003", text="operating expenses expanded"),
                _make_section(filing_id="0000000002-23-000002", text="operating expenses contracted"),
            ),
            (
                _make_section(filing_id="0000000003-24-000004", text="material weakness identified"),
                _make_section(filing_id="0000000003-23-000003", text="material weakness remediated"),
            ),
        ]
        results = analyzer.analyze_batch(pairs)
        assert len(results) == 3
        assert results[0].filing_id == "0000000001-24-000002"
        assert results[1].filing_id == "0000000002-24-000003"
        assert results[2].filing_id == "0000000003-24-000004"


# --- Registration Tests ---


class TestRegistration:
    def test_registered_in_registry(self):
        from edgar_sentinel.analyzers import registry
        assert "similarity" in registry.list_names()

    def test_registry_creates_analyzer(self):
        from edgar_sentinel.analyzers import registry
        factory = registry.get("similarity")
        config = SimilarityAnalyzerConfig(enabled=True)
        analyzer = factory(config)
        assert isinstance(analyzer, SimilarityAnalyzer)
        assert analyzer.name == "similarity"

    def test_exported_from_package(self):
        from edgar_sentinel.analyzers import SimilarityAnalyzer as SA
        assert SA is SimilarityAnalyzer


# --- Integration with AnalysisResults ---


class TestIntegrationWithResults:
    def test_run_analyzers_includes_similarity(self):
        from edgar_sentinel.analyzers.base import run_analyzers
        from edgar_sentinel.core.config import AnalyzersConfig, DictionaryAnalyzerConfig

        config = AnalyzersConfig(
            dictionary=DictionaryAnalyzerConfig(enabled=False),
            similarity=SimilarityAnalyzerConfig(enabled=True),
        )
        pairs = [
            (
                _make_section(filing_id="0000000001-24-000002", text="revenue growth accelerated"),
                _make_section(filing_id="0000000001-23-000001", text="revenue growth decelerated"),
            ),
        ]
        results = run_analyzers(pairs, config, analyzer_names=["similarity"])
        assert len(results.similarity_results) == 1
        assert results.similarity_results[0].filing_id == "0000000001-24-000002"

    def test_run_analyzers_skips_disabled_similarity(self):
        from edgar_sentinel.analyzers.base import run_analyzers
        from edgar_sentinel.core.config import AnalyzersConfig, DictionaryAnalyzerConfig

        config = AnalyzersConfig(
            dictionary=DictionaryAnalyzerConfig(enabled=False),
            similarity=SimilarityAnalyzerConfig(enabled=False),
        )
        pairs = [
            (
                _make_section(filing_id="0000000001-24-000002", text="revenue growth"),
                _make_section(filing_id="0000000001-23-000001", text="revenue decline"),
            ),
        ]
        results = run_analyzers(pairs, config)
        assert len(results.similarity_results) == 0
