"""Tests for the LLM-based sentiment analyzer."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from edgar_sentinel.core import (
    AnalysisError,
    AnalyzerType,
    FilingSection,
    LLMProvider,
    SentimentResult,
)
from edgar_sentinel.core.config import (
    AnalyzersConfig,
    DictionaryAnalyzerConfig,
    LLMAnalyzerConfig,
    SimilarityAnalyzerConfig,
)
from edgar_sentinel.core.exceptions import LLMError
from edgar_sentinel.analyzers.base import Analyzer, registry
from edgar_sentinel.analyzers.llm import (
    SENTIMENT_PROMPT_TEMPLATE,
    MAX_TEXT_LENGTH,
    TRUNCATION_NOTE,
    LLMProviderBackend,
    LLMResponse,
    LLMAnalyzer,
    ClaudeCLIProvider,
    AnthropicAPIProvider,
    OpenAIAPIProvider,
    create_provider,
    parse_llm_response,
    _validate_response,
)

NOW = datetime.now(timezone.utc)

VALID_LLM_JSON = json.dumps({
    "sentiment_score": 0.35,
    "confidence": 0.8,
    "themes": ["revenue growth", "market expansion", "cost reduction"],
    "tone": "positive",
    "reasoning": "Management expressed confidence in revenue growth driven by new markets.",
})

BEARISH_LLM_JSON = json.dumps({
    "sentiment_score": -0.6,
    "confidence": 0.75,
    "themes": ["declining margins", "competitive pressure"],
    "tone": "negative",
    "reasoning": "Filing reveals significant margin compression and increased competition.",
})


def _make_section(
    filing_id: str = "0000000001-24-000001",
    section_name: str = "mda",
    text: str = "The company reported strong revenue growth across all segments.",
    word_count: int | None = None,
) -> FilingSection:
    if word_count is None:
        word_count = len(text.split())
    return FilingSection(
        filing_id=filing_id,
        section_name=section_name,
        raw_text=text,
        word_count=word_count,
        extracted_at=NOW,
    )


def _make_config(**kwargs) -> LLMAnalyzerConfig:
    defaults = {
        "enabled": True,
        "provider": LLMProvider.CLAUDE_CLI,
        "model": "claude-sonnet-4-6",
        "max_concurrent": 2,
        "timeout_seconds": 60,
    }
    defaults.update(kwargs)
    return LLMAnalyzerConfig(**defaults)


class MockProvider:
    """A mock LLM provider for testing."""

    def __init__(self, response: str = VALID_LLM_JSON) -> None:
        self._response = response
        self.call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    async def query(self, prompt: str, max_tokens: int = 1024) -> str:
        self.call_count += 1
        return self._response


class FailingProvider:
    """A provider that always raises LLMError."""

    def __init__(self, message: str = "Provider failed") -> None:
        self._message = message
        self.call_count = 0

    @property
    def name(self) -> str:
        return "failing"

    async def query(self, prompt: str, max_tokens: int = 1024) -> str:
        self.call_count += 1
        raise LLMError(self._message, context={"provider": self.name})


# --- Prompt Template Tests ---


class TestPromptTemplate:
    def test_template_has_section_name_placeholder(self):
        assert "{section_name}" in SENTIMENT_PROMPT_TEMPLATE

    def test_template_has_form_type_placeholder(self):
        assert "{form_type}" in SENTIMENT_PROMPT_TEMPLATE

    def test_template_has_text_placeholder(self):
        assert "{text}" in SENTIMENT_PROMPT_TEMPLATE

    def test_template_asks_for_json(self):
        assert "JSON" in SENTIMENT_PROMPT_TEMPLATE

    def test_template_specifies_score_range(self):
        assert "-1.0" in SENTIMENT_PROMPT_TEMPLATE
        assert "1.0" in SENTIMENT_PROMPT_TEMPLATE

    def test_max_text_length(self):
        assert MAX_TEXT_LENGTH == 50_000

    def test_truncation_note(self):
        assert "truncated" in TRUNCATION_NOTE


# --- LLMResponse Tests ---


class TestLLMResponse:
    def test_creation(self):
        r = LLMResponse(
            sentiment_score=0.5,
            confidence=0.8,
            themes=["growth"],
            tone="positive",
            reasoning="Good results.",
        )
        assert r.sentiment_score == 0.5
        assert r.confidence == 0.8
        assert r.themes == ["growth"]
        assert r.tone == "positive"
        assert r.reasoning == "Good results."

    def test_slots(self):
        r = LLMResponse(0.0, 0.5, [], "neutral", "")
        assert hasattr(r, "__slots__")
        with pytest.raises(AttributeError):
            r.extra = "nope"


# --- parse_llm_response Tests ---


class TestParseLLMResponse:
    def test_valid_json(self):
        r = parse_llm_response(VALID_LLM_JSON)
        assert r.sentiment_score == 0.35
        assert r.confidence == 0.8
        assert len(r.themes) == 3
        assert r.tone == "positive"
        assert "confidence" in r.reasoning.lower() or len(r.reasoning) > 0

    def test_json_with_markdown_fences(self):
        wrapped = f"```json\n{VALID_LLM_JSON}\n```"
        r = parse_llm_response(wrapped)
        assert r.sentiment_score == 0.35

    def test_json_with_bare_fences(self):
        wrapped = f"```\n{VALID_LLM_JSON}\n```"
        r = parse_llm_response(wrapped)
        assert r.sentiment_score == 0.35

    def test_json_with_whitespace(self):
        padded = f"  \n\n  {VALID_LLM_JSON}  \n\n  "
        r = parse_llm_response(padded)
        assert r.sentiment_score == 0.35

    def test_invalid_json_raises(self):
        with pytest.raises(LLMError, match="Failed to parse"):
            parse_llm_response("not json at all")

    def test_non_object_json_raises(self):
        with pytest.raises(LLMError, match="Expected JSON object"):
            parse_llm_response("[1, 2, 3]")

    def test_missing_sentiment_score_raises(self):
        data = json.dumps({"confidence": 0.5, "tone": "neutral"})
        with pytest.raises(LLMError, match="sentiment_score"):
            parse_llm_response(data)

    def test_string_sentiment_score_raises(self):
        data = json.dumps({"sentiment_score": "high"})
        with pytest.raises(LLMError, match="sentiment_score"):
            parse_llm_response(data)

    def test_score_out_of_range_raises(self):
        data = json.dumps({"sentiment_score": 1.5})
        with pytest.raises(LLMError, match="out of range"):
            parse_llm_response(data)

    def test_score_negative_out_of_range_raises(self):
        data = json.dumps({"sentiment_score": -1.5})
        with pytest.raises(LLMError, match="out of range"):
            parse_llm_response(data)

    def test_missing_confidence_defaults(self):
        data = json.dumps({"sentiment_score": 0.2, "tone": "positive"})
        r = parse_llm_response(data)
        assert r.confidence == 0.5

    def test_confidence_clamped_high(self):
        data = json.dumps({"sentiment_score": 0.0, "confidence": 2.0})
        r = parse_llm_response(data)
        assert r.confidence == 1.0

    def test_confidence_clamped_low(self):
        data = json.dumps({"sentiment_score": 0.0, "confidence": -0.5})
        r = parse_llm_response(data)
        assert r.confidence == 0.0

    def test_missing_themes_defaults(self):
        data = json.dumps({"sentiment_score": 0.0})
        r = parse_llm_response(data)
        assert r.themes == []

    def test_non_list_themes_defaults(self):
        data = json.dumps({"sentiment_score": 0.0, "themes": "growth"})
        r = parse_llm_response(data)
        assert r.themes == []

    def test_themes_capped_at_10(self):
        data = json.dumps({
            "sentiment_score": 0.0,
            "themes": [f"theme_{i}" for i in range(15)],
        })
        r = parse_llm_response(data)
        assert len(r.themes) == 10

    def test_invalid_tone_defaults_neutral(self):
        data = json.dumps({"sentiment_score": 0.0, "tone": "bullish"})
        r = parse_llm_response(data)
        assert r.tone == "neutral"

    def test_valid_tones(self):
        for tone in ("positive", "negative", "neutral", "mixed"):
            data = json.dumps({"sentiment_score": 0.0, "tone": tone})
            r = parse_llm_response(data)
            assert r.tone == tone

    def test_missing_reasoning_defaults(self):
        data = json.dumps({"sentiment_score": 0.0})
        r = parse_llm_response(data)
        assert r.reasoning == ""

    def test_integer_score_accepted(self):
        data = json.dumps({"sentiment_score": 1})
        r = parse_llm_response(data)
        assert r.sentiment_score == 1.0

    def test_boundary_scores(self):
        for score in (-1.0, -0.5, 0.0, 0.5, 1.0):
            data = json.dumps({"sentiment_score": score})
            r = parse_llm_response(data)
            assert r.sentiment_score == score


# --- Provider Protocol Tests ---


class TestProviderProtocol:
    def test_mock_provider_satisfies_protocol(self):
        provider = MockProvider()
        assert isinstance(provider, LLMProviderBackend)

    def test_claude_cli_satisfies_protocol(self):
        provider = ClaudeCLIProvider()
        assert isinstance(provider, LLMProviderBackend)

    def test_provider_has_name(self):
        provider = MockProvider()
        assert provider.name == "mock"

    def test_claude_cli_name(self):
        provider = ClaudeCLIProvider()
        assert provider.name == "claude_cli"


# --- ClaudeCLIProvider Tests ---


class TestClaudeCLIProvider:
    def test_default_model(self):
        provider = ClaudeCLIProvider()
        assert provider._model == "claude-sonnet-4-6"

    def test_custom_model(self):
        provider = ClaudeCLIProvider(model="claude-haiku-4-5-20251001")
        assert provider._model == "claude-haiku-4-5-20251001"

    def test_default_timeout(self):
        provider = ClaudeCLIProvider()
        assert provider._timeout == 60

    def test_custom_timeout(self):
        provider = ClaudeCLIProvider(timeout_seconds=120)
        assert provider._timeout == 120

    @pytest.mark.asyncio
    async def test_successful_query(self):
        provider = ClaudeCLIProvider()
        cli_output = json.dumps({"type": "result", "result": "LLM response text"})

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(cli_output.encode(), b"")
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await provider.query("test prompt")
            assert result == "LLM response text"

    @pytest.mark.asyncio
    async def test_plain_text_fallback(self):
        provider = ClaudeCLIProvider()

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(b"plain text response", b"")
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await provider.query("test prompt")
            assert result == "plain text response"

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises(self):
        provider = ClaudeCLIProvider()

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"error message")
        )
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(LLMError, match="exited with code 1"):
                await provider.query("test prompt")

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        provider = ClaudeCLIProvider(timeout_seconds=1)

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                with pytest.raises(LLMError, match="timed out"):
                    await provider.query("test prompt")

    @pytest.mark.asyncio
    async def test_not_found_raises(self):
        provider = ClaudeCLIProvider()

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            with pytest.raises(LLMError, match="not found"):
                await provider.query("test prompt")

    @pytest.mark.asyncio
    async def test_command_arguments(self):
        provider = ClaudeCLIProvider(model="test-model")
        cli_output = json.dumps({"result": "ok"})

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(cli_output.encode(), b"")
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await provider.query("my prompt")
            args = mock_exec.call_args[0]
            assert args[0] == "claude"
            assert "-p" in args
            assert "my prompt" in args
            assert "--output-format" in args
            assert "json" in args
            assert "--model" in args
            assert "test-model" in args
            assert "--max-turns" in args
            assert "1" in args


# --- create_provider Tests ---


class TestCreateProvider:
    def test_claude_cli(self):
        config = _make_config(provider=LLMProvider.CLAUDE_CLI)
        provider = create_provider(config)
        assert isinstance(provider, ClaudeCLIProvider)
        assert provider.name == "claude_cli"

    def test_claude_cli_passes_model(self):
        config = _make_config(provider=LLMProvider.CLAUDE_CLI, model="test-model")
        provider = create_provider(config)
        assert provider._model == "test-model"

    def test_claude_cli_passes_timeout(self):
        config = _make_config(provider=LLMProvider.CLAUDE_CLI, timeout_seconds=120)
        provider = create_provider(config)
        assert provider._timeout == 120


# --- LLMAnalyzer Properties Tests ---


class TestAnalyzerProperties:
    def _make_analyzer(self, **kwargs):
        config = _make_config(**kwargs)
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._config = config
        analyzer._provider = MockProvider()
        analyzer._max_retries = 3
        analyzer._semaphore = asyncio.Semaphore(config.max_concurrent)
        return analyzer

    def test_name(self):
        analyzer = self._make_analyzer()
        assert analyzer.name == "llm"

    def test_analyzer_type(self):
        analyzer = self._make_analyzer()
        assert analyzer.analyzer_type == AnalyzerType.LLM

    def test_satisfies_protocol(self):
        analyzer = self._make_analyzer()
        assert isinstance(analyzer, Analyzer)

    def test_config_stored(self):
        analyzer = self._make_analyzer(model="test-model")
        assert analyzer._config.model == "test-model"


# --- LLMAnalyzer._analyze_async Tests ---


class TestAnalyzeAsync:
    def _make_analyzer_with_mock(self, response=VALID_LLM_JSON, **kwargs):
        config = _make_config(**kwargs)
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._config = config
        analyzer._provider = MockProvider(response)
        analyzer._max_retries = 3
        analyzer._semaphore = asyncio.Semaphore(config.max_concurrent)
        return analyzer

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        analyzer = self._make_analyzer_with_mock()
        section = _make_section()
        result = await analyzer._analyze_async(section)
        assert isinstance(result, SentimentResult)
        assert result.sentiment_score == 0.35
        assert result.confidence == 0.8
        assert result.filing_id == section.filing_id
        assert result.section_name == "mda"
        assert result.analyzer_name == "llm"

    @pytest.mark.asyncio
    async def test_result_metadata(self):
        analyzer = self._make_analyzer_with_mock()
        section = _make_section()
        result = await analyzer._analyze_async(section)
        assert result.metadata["provider"] == "mock"
        assert result.metadata["model"] == "claude-sonnet-4-6"
        assert "themes" in result.metadata
        assert result.metadata["tone"] == "positive"
        assert result.metadata["truncated"] is False

    @pytest.mark.asyncio
    async def test_bearish_analysis(self):
        analyzer = self._make_analyzer_with_mock(BEARISH_LLM_JSON)
        section = _make_section(text="Margins declined significantly due to competitive pressure.")
        result = await analyzer._analyze_async(section)
        assert result.sentiment_score == -0.6
        assert result.metadata["tone"] == "negative"

    @pytest.mark.asyncio
    async def test_empty_text_raises(self):
        analyzer = self._make_analyzer_with_mock()
        section = _make_section(text="", word_count=0)
        with pytest.raises(AnalysisError, match="Empty section text"):
            await analyzer._analyze_async(section)

    @pytest.mark.asyncio
    async def test_whitespace_only_text_raises(self):
        analyzer = self._make_analyzer_with_mock()
        section = _make_section(text="   \n\t  ", word_count=0)
        with pytest.raises(AnalysisError, match="Empty section text"):
            await analyzer._analyze_async(section)

    @pytest.mark.asyncio
    async def test_text_truncation(self):
        long_text = "word " * (MAX_TEXT_LENGTH // 5 + 1000)
        analyzer = self._make_analyzer_with_mock()
        section = _make_section(text=long_text, word_count=len(long_text.split()))
        result = await analyzer._analyze_async(section)
        assert result.metadata["truncated"] is True
        assert result.metadata["text_length"] <= MAX_TEXT_LENGTH + len(TRUNCATION_NOTE) + 10

    @pytest.mark.asyncio
    async def test_result_is_frozen(self):
        analyzer = self._make_analyzer_with_mock()
        section = _make_section()
        result = await analyzer._analyze_async(section)
        with pytest.raises(Exception):
            result.sentiment_score = 999.0

    @pytest.mark.asyncio
    async def test_analyzed_at_is_set(self):
        analyzer = self._make_analyzer_with_mock()
        section = _make_section()
        result = await analyzer._analyze_async(section)
        assert result.analyzed_at is not None
        assert result.analyzed_at.tzinfo is not None


# --- Retry Logic Tests ---


class TestRetryLogic:
    def _make_analyzer_with_provider(self, provider, **kwargs):
        config = _make_config(**kwargs)
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._config = config
        analyzer._provider = provider
        analyzer._max_retries = 3
        analyzer._semaphore = asyncio.Semaphore(config.max_concurrent)
        return analyzer

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        provider = FailingProvider("always fails")
        analyzer = self._make_analyzer_with_provider(provider)
        section = _make_section()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMError, match="failed after 3 attempts"):
                await analyzer._analyze_async(section)

        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt(self):
        call_count = 0

        class FailOnceProvider:
            @property
            def name(self):
                return "fail_once"

            async def query(self, prompt, max_tokens=1024):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise LLMError("first try fails", context={"provider": "fail_once"})
                return VALID_LLM_JSON

        analyzer = self._make_analyzer_with_provider(FailOnceProvider())
        section = _make_section()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await analyzer._analyze_async(section)

        assert result.sentiment_score == 0.35
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_error_context(self):
        provider = FailingProvider()
        analyzer = self._make_analyzer_with_provider(provider)
        section = _make_section()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMError) as exc_info:
                await analyzer._analyze_async(section)

        assert exc_info.value.context["attempts"] == 3
        assert exc_info.value.context["filing_id"] == section.filing_id

    @pytest.mark.asyncio
    async def test_does_not_retry_analysis_error(self):
        """AnalysisError (input problem) should not be retried."""
        analyzer = self._make_analyzer_with_provider(MockProvider())
        section = _make_section(text="", word_count=0)

        with pytest.raises(AnalysisError):
            await analyzer._analyze_async(section)


# --- analyze (sync wrapper) Tests ---


class TestAnalyzeSync:
    def test_sync_analyze_calls_async(self):
        config = _make_config()
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._config = config
        analyzer._provider = MockProvider()
        analyzer._max_retries = 3
        analyzer._semaphore = asyncio.Semaphore(2)

        section = _make_section()
        result = analyzer.analyze(section)
        assert isinstance(result, SentimentResult)
        assert result.sentiment_score == 0.35

    def test_sync_analyze_ignores_prior_section(self):
        config = _make_config()
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._config = config
        analyzer._provider = MockProvider()
        analyzer._max_retries = 3
        analyzer._semaphore = asyncio.Semaphore(2)

        section = _make_section()
        prior = _make_section(filing_id="0000000001-23-000001")
        result = analyzer.analyze(section, prior_section=prior)
        assert isinstance(result, SentimentResult)


# --- analyze_batch Tests ---


class TestAnalyzeBatch:
    def _make_analyzer(self, provider=None, **kwargs):
        config = _make_config(**kwargs)
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer._config = config
        analyzer._provider = provider or MockProvider()
        analyzer._max_retries = 3
        analyzer._semaphore = asyncio.Semaphore(config.max_concurrent)
        return analyzer

    def test_batch_multiple_sections(self):
        analyzer = self._make_analyzer()
        sections = [
            (_make_section(filing_id="0000000001-24-000001"), None),
            (_make_section(filing_id="0000000001-24-000002"), None),
            (_make_section(filing_id="0000000001-24-000003"), None),
        ]
        results = analyzer.analyze_batch(sections)
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)

    def test_batch_preserves_filing_ids(self):
        analyzer = self._make_analyzer()
        sections = [
            (_make_section(filing_id="0000000001-24-000001"), None),
            (_make_section(filing_id="0000000001-24-000002"), None),
        ]
        results = analyzer.analyze_batch(sections)
        ids = {r.filing_id for r in results}
        assert "0000000001-24-000001" in ids
        assert "0000000001-24-000002" in ids

    def test_batch_skips_failures(self):
        """Failed analyses are logged and skipped, not raised."""
        call_count = 0

        class AlternatingProvider:
            @property
            def name(self):
                return "alternating"

            async def query(self, prompt, max_tokens=1024):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    raise LLMError("fail", context={"provider": "alternating"})
                return VALID_LLM_JSON

        analyzer = self._make_analyzer(provider=AlternatingProvider())
        # 3 retries per failure, so we need a provider that always fails on certain calls
        # Instead, let's use a simpler approach with the FailingProvider for specific sections

        sections = [
            (_make_section(filing_id="0000000001-24-000001", text="good text here"), None),
            (_make_section(filing_id="0000000001-24-000002", text=""), None),  # Empty = AnalysisError
        ]
        results = analyzer.analyze_batch(sections)
        # First should succeed, second fails with empty text
        assert len(results) == 1
        assert results[0].filing_id == "0000000001-24-000001"

    def test_batch_empty_input(self):
        analyzer = self._make_analyzer()
        results = analyzer.analyze_batch([])
        assert results == []

    def test_batch_single_section(self):
        analyzer = self._make_analyzer()
        sections = [(_make_section(), None)]
        results = analyzer.analyze_batch(sections)
        assert len(results) == 1

    def test_batch_ignores_prior_sections(self):
        analyzer = self._make_analyzer()
        prior = _make_section(filing_id="0000000001-23-000001")
        sections = [(_make_section(), prior)]
        results = analyzer.analyze_batch(sections)
        assert len(results) == 1


# --- Registration Tests ---


class TestRegistration:
    def test_llm_registered(self):
        assert "llm" in registry.list_names()

    def test_factory_creates_llm(self):
        factory = registry.get("llm")
        assert factory is LLMAnalyzer

    def test_import_from_package(self):
        from edgar_sentinel.analyzers import LLMAnalyzer as Imported
        assert Imported is LLMAnalyzer

    def test_create_enabled_with_llm(self):
        config = AnalyzersConfig(
            dictionary=DictionaryAnalyzerConfig(enabled=False),
            similarity=SimilarityAnalyzerConfig(enabled=False),
            llm=LLMAnalyzerConfig(enabled=True),
        )
        analyzers = registry.create_enabled(config)
        names = [a.name for a in analyzers]
        assert "llm" in names

    def test_create_enabled_without_llm(self):
        config = AnalyzersConfig(
            dictionary=DictionaryAnalyzerConfig(enabled=False),
            similarity=SimilarityAnalyzerConfig(enabled=False),
            llm=LLMAnalyzerConfig(enabled=False),
        )
        analyzers = registry.create_enabled(config)
        names = [a.name for a in analyzers]
        assert "llm" not in names


# --- Integration with AnalysisResults ---


class TestIntegrationWithResults:
    def test_llm_results_in_sentiment(self):
        from edgar_sentinel.analyzers.base import run_analyzers

        config = AnalyzersConfig(
            dictionary=DictionaryAnalyzerConfig(enabled=False),
            similarity=SimilarityAnalyzerConfig(enabled=False),
            llm=LLMAnalyzerConfig(enabled=True),
        )

        section = _make_section()
        sections = [(section, None)]

        # We need to mock the provider within the LLMAnalyzer
        with patch.object(LLMAnalyzer, "__init__", lambda self, cfg: None):
            analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
            analyzer._config = _make_config()
            analyzer._provider = MockProvider()
            analyzer._max_retries = 3
            analyzer._semaphore = asyncio.Semaphore(2)

            with patch.object(registry, "create_enabled", return_value=[analyzer]):
                results = run_analyzers(sections, config)
                assert results.total_count == 1
                assert len(results.sentiment_results) == 1
                assert results.sentiment_results[0].analyzer_name == "llm"

    def test_disabled_llm_no_results(self):
        from edgar_sentinel.analyzers.base import run_analyzers

        config = AnalyzersConfig(
            dictionary=DictionaryAnalyzerConfig(enabled=False),
            similarity=SimilarityAnalyzerConfig(enabled=False),
            llm=LLMAnalyzerConfig(enabled=False),
        )

        section = _make_section()
        sections = [(section, None)]

        results = run_analyzers(sections, config)
        assert results.total_count == 0
