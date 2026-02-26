"""Custom exception hierarchy for edgar-sentinel."""

from typing import Any


class EdgarSentinelError(Exception):
    """Base exception for all edgar-sentinel errors.

    All exceptions carry an optional `context` dict for structured error
    metadata that can be logged or serialized without parsing the message.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}


class ConfigError(EdgarSentinelError):
    """Invalid or missing configuration.

    Raised by load_config() during startup. Should be treated as fatal.

    Context keys:
        field: str — the config field that failed validation
        value: Any — the invalid value (redacted for secrets)
    """


class IngestionError(EdgarSentinelError):
    """Failed to fetch or parse EDGAR data.

    Policy: log and skip the filing. Do not abort the batch.

    Context keys:
        accession_number: str — the filing that failed
        url: str — the URL that was being fetched
    """


class RateLimitError(IngestionError):
    """SEC EDGAR rate limit exceeded (HTTP 429 or similar).

    Policy: backoff and retry (handled by EdgarClient internally).

    Context keys:
        retry_after: int | None — seconds to wait
    """


class ParsingError(IngestionError):
    """Could not extract sections from a filing document.

    Policy: log and skip the section. Store successfully extracted sections.

    Context keys:
        section_name: str — the section that failed to parse
        reason: str — why parsing failed
    """


class AnalysisError(EdgarSentinelError):
    """Analyzer failed to produce a result.

    Policy: log and skip. Other analyzers' results remain valid.

    Context keys:
        analyzer_name: str — which analyzer failed
        filing_id: str — which filing was being analyzed
    """


class LLMError(AnalysisError):
    """LLM provider returned an error or malformed response.

    Policy: retry up to 3 times with exponential backoff.

    Context keys:
        provider: str — "claude_cli", "anthropic", or "openai"
        status_code: int | None — HTTP status code if applicable
        response_body: str | None — truncated response for debugging
    """


class StorageError(EdgarSentinelError):
    """Database operation failed.

    Policy: raise immediately. Data integrity is critical.

    Context keys:
        operation: str — "insert", "query", "migrate", etc.
        table: str — the table involved
    """


class BacktestError(EdgarSentinelError):
    """Backtest execution failed.

    Policy: raise immediately. Partial backtest results are misleading.

    Context keys:
        stage: str — "data_loading", "signal_ranking", etc.
    """
