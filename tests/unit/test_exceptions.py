"""Tests for edgar_sentinel.core.exceptions."""

import pytest

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


class TestExceptionHierarchy:
    """Verify the exception class hierarchy."""

    def test_config_is_subclass(self):
        assert issubclass(ConfigError, EdgarSentinelError)

    def test_ingestion_is_subclass(self):
        assert issubclass(IngestionError, EdgarSentinelError)

    def test_rate_limit_is_subclass_of_ingestion(self):
        assert issubclass(RateLimitError, IngestionError)
        assert issubclass(RateLimitError, EdgarSentinelError)

    def test_parsing_is_subclass_of_ingestion(self):
        assert issubclass(ParsingError, IngestionError)
        assert issubclass(ParsingError, EdgarSentinelError)

    def test_analysis_is_subclass(self):
        assert issubclass(AnalysisError, EdgarSentinelError)

    def test_llm_is_subclass_of_analysis(self):
        assert issubclass(LLMError, AnalysisError)
        assert issubclass(LLMError, EdgarSentinelError)

    def test_storage_is_subclass(self):
        assert issubclass(StorageError, EdgarSentinelError)

    def test_backtest_is_subclass(self):
        assert issubclass(BacktestError, EdgarSentinelError)


class TestExceptionContext:
    """Verify context dict behavior."""

    def test_context_preserved(self):
        exc = IngestionError(
            "Failed to fetch",
            context={"accession_number": "0000320193-23-000106", "url": "https://example.com"},
        )
        assert exc.context["accession_number"] == "0000320193-23-000106"
        assert exc.context["url"] == "https://example.com"

    def test_default_context_is_empty_dict(self):
        exc = EdgarSentinelError("test error")
        assert exc.context == {}

    def test_str_returns_message(self):
        exc = ConfigError("invalid field")
        assert str(exc) == "invalid field"

    def test_context_none_becomes_empty_dict(self):
        exc = StorageError("db fail", context=None)
        assert exc.context == {}

    def test_exception_can_be_caught_as_base(self):
        with pytest.raises(EdgarSentinelError):
            raise RateLimitError("too fast", context={"retry_after": 10})

    def test_exception_can_be_caught_as_parent(self):
        with pytest.raises(IngestionError):
            raise ParsingError("bad html", context={"section_name": "mda"})
