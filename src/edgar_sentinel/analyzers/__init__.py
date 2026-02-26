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

# Register built-in analyzers
registry.register("dictionary", DictionaryAnalyzer)

__all__ = [
    "Analyzer",
    "AnalyzerFactory",
    "AnalyzerRegistry",
    "AnalysisResults",
    "DictionaryAnalyzer",
    "LMDictionary",
    "registry",
    "run_analyzers",
]
