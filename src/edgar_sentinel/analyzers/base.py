"""Analyzer protocol, registry, and orchestration."""

from __future__ import annotations

import logging
from typing import Callable, Protocol, Sequence, runtime_checkable

from edgar_sentinel.core import (
    AnalysisError,
    AnalyzerType,
    FilingSection,
    SentimentResult,
    SimilarityResult,
)
from edgar_sentinel.core.config import AnalyzersConfig

logger = logging.getLogger("edgar_sentinel.analyzers")


@runtime_checkable
class Analyzer(Protocol):
    """Protocol for all filing section analyzers."""

    @property
    def name(self) -> str: ...

    @property
    def analyzer_type(self) -> AnalyzerType: ...

    def analyze(
        self,
        section: FilingSection,
        prior_section: FilingSection | None = None,
    ) -> SentimentResult | SimilarityResult: ...

    def analyze_batch(
        self,
        sections: list[tuple[FilingSection, FilingSection | None]],
    ) -> list[SentimentResult | SimilarityResult]: ...


AnalyzerFactory = Callable[..., Analyzer]


class AnalyzerRegistry:
    """Registry of available analyzers."""

    def __init__(self) -> None:
        self._factories: dict[str, AnalyzerFactory] = {}

    def register(self, name: str, factory: AnalyzerFactory) -> None:
        if name in self._factories:
            raise ValueError(
                f"Analyzer '{name}' is already registered. Use replace() to override."
            )
        self._factories[name] = factory

    def replace(self, name: str, factory: AnalyzerFactory) -> None:
        if name not in self._factories:
            raise KeyError(f"Analyzer '{name}' is not registered.")
        self._factories[name] = factory

    def get(self, name: str) -> AnalyzerFactory:
        return self._factories[name]

    def list_names(self) -> list[str]:
        return list(self._factories.keys())

    def create_enabled(self, config: AnalyzersConfig) -> list[Analyzer]:
        """Instantiate all enabled analyzers from config."""
        analyzers: list[Analyzer] = []
        config_map = {
            "dictionary": config.dictionary,
            "similarity": config.similarity,
            "llm": config.llm,
        }
        for name, factory in self._factories.items():
            analyzer_config = config_map.get(name)
            if analyzer_config is not None:
                if getattr(analyzer_config, "enabled", True):
                    analyzers.append(factory(analyzer_config))
            else:
                # Custom analyzers without config are always enabled
                analyzers.append(factory())
        return analyzers


class AnalysisResults:
    """Container for results from all analyzers."""

    def __init__(
        self,
        sentiment_results: list[SentimentResult],
        similarity_results: list[SimilarityResult],
    ) -> None:
        self.sentiment_results = sentiment_results
        self.similarity_results = similarity_results

    @property
    def total_count(self) -> int:
        return len(self.sentiment_results) + len(self.similarity_results)

    def sentiment_by_filing(self, filing_id: str) -> list[SentimentResult]:
        return [r for r in self.sentiment_results if r.filing_id == filing_id]

    def similarity_by_filing(self, filing_id: str) -> list[SimilarityResult]:
        return [r for r in self.similarity_results if r.filing_id == filing_id]


def run_analyzers(
    sections: Sequence[tuple[FilingSection, FilingSection | None]],
    config: AnalyzersConfig,
    analyzer_names: list[str] | None = None,
) -> AnalysisResults:
    """Run enabled analyzers on a batch of filing sections."""
    analyzers = registry.create_enabled(config)
    if analyzer_names:
        analyzers = [a for a in analyzers if a.name in analyzer_names]

    sentiment_results: list[SentimentResult] = []
    similarity_results: list[SimilarityResult] = []

    for analyzer in analyzers:
        logger.info(
            "Running analyzer '%s' on %d sections",
            analyzer.name,
            len(sections),
        )
        try:
            results = analyzer.analyze_batch(list(sections))
            for result in results:
                if isinstance(result, SimilarityResult):
                    similarity_results.append(result)
                else:
                    sentiment_results.append(result)
            logger.info(
                "Analyzer '%s' produced %d results",
                analyzer.name,
                len(results),
            )
        except Exception as e:
            logger.error("Analyzer '%s' batch failed: %s", analyzer.name, e)

    return AnalysisResults(sentiment_results, similarity_results)


# Module-level singleton registry
registry = AnalyzerRegistry()
