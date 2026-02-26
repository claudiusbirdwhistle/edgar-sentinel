"""Tests for the analyzer base module: protocol, registry, orchestration."""

from datetime import datetime, timezone

import pytest

from edgar_sentinel.core import (
    AnalysisError,
    AnalyzerType,
    FilingSection,
    SentimentResult,
    SimilarityResult,
)
from edgar_sentinel.core.config import (
    AnalyzersConfig,
    DictionaryAnalyzerConfig,
    LLMAnalyzerConfig,
    SimilarityAnalyzerConfig,
)
from edgar_sentinel.analyzers.base import (
    Analyzer,
    AnalyzerRegistry,
    AnalysisResults,
    run_analyzers,
)


# --- Fixtures ---


NOW = datetime.now(timezone.utc)


def make_section(filing_id="0000320193-23-000106", section_name="mda", text="hello world"):
    return FilingSection(
        filing_id=filing_id,
        section_name=section_name,
        raw_text=text,
        word_count=len(text.split()),
        extracted_at=NOW,
    )


class FakeSentimentAnalyzer:
    """A minimal analyzer that satisfies the Analyzer protocol."""

    @property
    def name(self) -> str:
        return "fake_sentiment"

    @property
    def analyzer_type(self) -> AnalyzerType:
        return AnalyzerType.DICTIONARY

    def analyze(self, section, prior_section=None):
        return SentimentResult(
            filing_id=section.filing_id,
            section_name=section.section_name,
            analyzer_name=self.name,
            sentiment_score=0.5,
            confidence=0.8,
            metadata={},
            analyzed_at=NOW,
        )

    def analyze_batch(self, sections):
        results = []
        for section, prior in sections:
            results.append(self.analyze(section, prior))
        return results


class FakeSimilarityAnalyzer:
    """A minimal analyzer producing SimilarityResults."""

    @property
    def name(self) -> str:
        return "fake_similarity"

    @property
    def analyzer_type(self) -> AnalyzerType:
        return AnalyzerType.SIMILARITY

    def analyze(self, section, prior_section=None):
        if prior_section is None:
            raise AnalysisError("No prior section")
        return SimilarityResult(
            filing_id=section.filing_id,
            prior_filing_id=prior_section.filing_id,
            section_name=section.section_name,
            similarity_score=0.85,
            change_score=0.15,
            analyzed_at=NOW,
        )

    def analyze_batch(self, sections):
        results = []
        for section, prior in sections:
            if prior is not None:
                results.append(self.analyze(section, prior))
        return results


class FailingAnalyzer:
    """An analyzer that always raises."""

    @property
    def name(self) -> str:
        return "failing"

    @property
    def analyzer_type(self) -> AnalyzerType:
        return AnalyzerType.DICTIONARY

    def analyze(self, section, prior_section=None):
        raise AnalysisError("Always fails")

    def analyze_batch(self, sections):
        raise AnalysisError("Batch always fails")


# --- Protocol Tests ---


class TestAnalyzerProtocol:
    def test_fake_sentiment_satisfies_protocol(self):
        analyzer = FakeSentimentAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_fake_similarity_satisfies_protocol(self):
        analyzer = FakeSimilarityAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_non_analyzer_fails_protocol(self):
        class NotAnAnalyzer:
            pass

        assert not isinstance(NotAnAnalyzer(), Analyzer)


# --- Registry Tests ---


class TestAnalyzerRegistry:
    def test_register_and_get(self):
        reg = AnalyzerRegistry()
        reg.register("test", lambda: FakeSentimentAnalyzer())
        factory = reg.get("test")
        analyzer = factory()
        assert analyzer.name == "fake_sentiment"

    def test_register_duplicate_raises(self):
        reg = AnalyzerRegistry()
        reg.register("test", lambda: FakeSentimentAnalyzer())
        with pytest.raises(ValueError, match="already registered"):
            reg.register("test", lambda: FakeSentimentAnalyzer())

    def test_replace_existing(self):
        reg = AnalyzerRegistry()
        reg.register("test", lambda: FakeSentimentAnalyzer())
        reg.replace("test", lambda: FakeSimilarityAnalyzer())
        factory = reg.get("test")
        analyzer = factory()
        assert analyzer.name == "fake_similarity"

    def test_replace_nonexistent_raises(self):
        reg = AnalyzerRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.replace("nonexistent", lambda: FakeSentimentAnalyzer())

    def test_get_nonexistent_raises(self):
        reg = AnalyzerRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_list_names(self):
        reg = AnalyzerRegistry()
        reg.register("alpha", lambda: FakeSentimentAnalyzer())
        reg.register("beta", lambda: FakeSimilarityAnalyzer())
        names = reg.list_names()
        assert "alpha" in names
        assert "beta" in names
        assert len(names) == 2

    def test_list_names_empty(self):
        reg = AnalyzerRegistry()
        assert reg.list_names() == []

    def test_create_enabled_dictionary(self):
        reg = AnalyzerRegistry()
        reg.register("dictionary", lambda cfg: FakeSentimentAnalyzer())
        config = AnalyzersConfig(
            dictionary=DictionaryAnalyzerConfig(enabled=True),
            similarity=SimilarityAnalyzerConfig(enabled=False),
            llm=LLMAnalyzerConfig(enabled=False),
        )
        analyzers = reg.create_enabled(config)
        assert len(analyzers) == 1
        assert analyzers[0].name == "fake_sentiment"

    def test_create_enabled_skips_disabled(self):
        reg = AnalyzerRegistry()
        reg.register("dictionary", lambda cfg: FakeSentimentAnalyzer())
        config = AnalyzersConfig(
            dictionary=DictionaryAnalyzerConfig(enabled=False),
            similarity=SimilarityAnalyzerConfig(enabled=False),
            llm=LLMAnalyzerConfig(enabled=False),
        )
        analyzers = reg.create_enabled(config)
        assert len(analyzers) == 0

    def test_create_enabled_custom_analyzer_always_included(self):
        reg = AnalyzerRegistry()
        reg.register("custom", lambda: FakeSentimentAnalyzer())
        config = AnalyzersConfig(
            dictionary=DictionaryAnalyzerConfig(enabled=False),
            similarity=SimilarityAnalyzerConfig(enabled=False),
            llm=LLMAnalyzerConfig(enabled=False),
        )
        analyzers = reg.create_enabled(config)
        assert len(analyzers) == 1
        assert analyzers[0].name == "fake_sentiment"


# --- AnalysisResults Tests ---


class TestAnalysisResults:
    def test_empty_results(self):
        results = AnalysisResults([], [])
        assert results.total_count == 0
        assert results.sentiment_by_filing("x") == []
        assert results.similarity_by_filing("x") == []

    def test_mixed_results(self):
        sent = SentimentResult(
            filing_id="0000320193-23-000106",
            section_name="mda",
            analyzer_name="test",
            sentiment_score=0.1,
            confidence=0.5,
            metadata={},
            analyzed_at=NOW,
        )
        sim = SimilarityResult(
            filing_id="0000320193-23-000106",
            prior_filing_id="0000320193-22-000106",
            section_name="mda",
            similarity_score=0.9,
            change_score=0.1,
            analyzed_at=NOW,
        )
        results = AnalysisResults([sent], [sim])
        assert results.total_count == 2

    def test_filter_by_filing(self):
        s1 = SentimentResult(
            filing_id="0000320193-23-000106",
            section_name="mda",
            analyzer_name="test",
            sentiment_score=0.1,
            confidence=0.5,
            metadata={},
            analyzed_at=NOW,
        )
        s2 = SentimentResult(
            filing_id="0000320193-24-000200",
            section_name="mda",
            analyzer_name="test",
            sentiment_score=-0.1,
            confidence=0.5,
            metadata={},
            analyzed_at=NOW,
        )
        results = AnalysisResults([s1, s2], [])
        filing1 = results.sentiment_by_filing("0000320193-23-000106")
        assert len(filing1) == 1
        assert filing1[0].sentiment_score == 0.1


# --- run_analyzers Tests ---


class TestRunAnalyzers:
    def test_run_with_no_analyzers(self):
        """Patching module registry is complex; test with empty registry."""
        from edgar_sentinel.analyzers import base

        original = base.registry
        try:
            base.registry = AnalyzerRegistry()
            config = AnalyzersConfig()
            section = make_section()
            results = run_analyzers([(section, None)], config)
            assert results.total_count == 0
        finally:
            base.registry = original

    def test_run_with_sentiment_analyzer(self):
        from edgar_sentinel.analyzers import base

        original = base.registry
        try:
            reg = AnalyzerRegistry()
            reg.register("dictionary", lambda cfg: FakeSentimentAnalyzer())
            base.registry = reg
            config = AnalyzersConfig()
            section = make_section()
            results = run_analyzers([(section, None)], config)
            assert len(results.sentiment_results) == 1
            assert results.sentiment_results[0].sentiment_score == 0.5
        finally:
            base.registry = original

    def test_run_filters_by_name(self):
        from edgar_sentinel.analyzers import base

        original = base.registry
        try:
            reg = AnalyzerRegistry()
            reg.register("dictionary", lambda cfg: FakeSentimentAnalyzer())
            base.registry = reg
            config = AnalyzersConfig()
            section = make_section()
            # Filter to only "similarity" — dictionary should be excluded
            results = run_analyzers([(section, None)], config, analyzer_names=["similarity"])
            assert results.total_count == 0
        finally:
            base.registry = original

    def test_run_handles_analyzer_failure(self):
        from edgar_sentinel.analyzers import base

        original = base.registry
        try:
            reg = AnalyzerRegistry()
            reg.register("custom", lambda: FailingAnalyzer())
            base.registry = reg
            config = AnalyzersConfig()
            section = make_section()
            # Should not raise — errors are caught and logged
            results = run_analyzers([(section, None)], config)
            assert results.total_count == 0
        finally:
            base.registry = original
