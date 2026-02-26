"""edgar_sentinel.analyzers — Filing section analysis."""

from edgar_sentinel.analyzers.base import (
    Analyzer,
    AnalyzerFactory,
    AnalyzerRegistry,
    AnalysisResults,
    registry,
    run_analyzers,
)
from edgar_sentinel.analyzers.dictionary import (
    DictionaryAnalyzer,
    LMDictionary,
)
from edgar_sentinel.analyzers.llm import LLMAnalyzer
from edgar_sentinel.analyzers.similarity import SimilarityAnalyzer

# Register built-in analyzers
registry.register("dictionary", DictionaryAnalyzer)
registry.register("similarity", SimilarityAnalyzer)
registry.register("llm", LLMAnalyzer)

__all__ = [
    "Analyzer",
    "AnalyzerFactory",
    "AnalyzerRegistry",
    "AnalysisResults",
    "DictionaryAnalyzer",
    "LLMAnalyzer",
    "LMDictionary",
    "SimilarityAnalyzer",
    "registry",
    "run_analyzers",
]
