"""edgar_sentinel.core — Foundation types, config, and exceptions."""

from edgar_sentinel.core.config import (
    AnalyzersConfig,
    APIConfig,
    EdgarConfig,
    SentinelConfig,
    SignalsConfig,
    StorageConfig,
    load_config,
)
from edgar_sentinel.core.exceptions import (
    AnalysisError,
    BacktestError,
    ConfigError,
    EdgarSentinelError,
    IngestionError,
    LLMError,
    ParsingError,
    RateLimitError,
    StorageError,
)
from edgar_sentinel.core.models import (
    AccessionNumber,
    AnalyzerName,
    AnalyzerType,
    BacktestConfig,
    BacktestResult,
    CIK,
    CompositeMethod,
    CompositeSignal,
    Filing,
    FilingMetadata,
    FilingSection,
    FormType,
    LLMProvider,
    MonthlyReturn,
    RebalanceFrequency,
    SectionName,
    SectionType,
    SentimentResult,
    Signal,
    SignalName,
    SimilarityResult,
    StorageBackend,
    Ticker,
)

__all__ = [
    # Type aliases
    "AccessionNumber",
    "CIK",
    "Ticker",
    "SectionName",
    "AnalyzerName",
    "SignalName",
    # Enums
    "FormType",
    "SectionType",
    "AnalyzerType",
    "RebalanceFrequency",
    "StorageBackend",
    "LLMProvider",
    "CompositeMethod",
    # Filing models
    "FilingMetadata",
    "FilingSection",
    "Filing",
    # Analysis models
    "SentimentResult",
    "SimilarityResult",
    # Signal models
    "Signal",
    "CompositeSignal",
    # Backtest models
    "BacktestConfig",
    "MonthlyReturn",
    "BacktestResult",
    # Config
    "SentinelConfig",
    "EdgarConfig",
    "StorageConfig",
    "AnalyzersConfig",
    "SignalsConfig",
    "APIConfig",
    "load_config",
    # Exceptions
    "EdgarSentinelError",
    "ConfigError",
    "IngestionError",
    "RateLimitError",
    "ParsingError",
    "AnalysisError",
    "LLMError",
    "StorageError",
    "BacktestError",
]
