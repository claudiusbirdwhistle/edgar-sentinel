"""LLM-based sentiment analyzer with pluggable provider backends."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from edgar_sentinel.core import (
    AnalysisError,
    AnalyzerType,
    FilingSection,
    LLMProvider,
    SentimentResult,
)
from edgar_sentinel.core.config import LLMAnalyzerConfig
from edgar_sentinel.core.exceptions import LLMError

logger = logging.getLogger("edgar_sentinel.analyzers.llm")

# --- Prompt Template ---

SENTIMENT_PROMPT_TEMPLATE = """You are a financial analyst evaluating SEC filing language. Analyze the following {section_name} section from a {form_type} filing.

Provide a structured sentiment assessment as JSON with exactly these fields:
- sentiment_score: float between -1.0 (very bearish) and 1.0 (very bullish), where 0.0 is neutral
- confidence: float between 0.0 and 1.0 indicating how confident you are in the score
- themes: list of 3-5 key themes identified (strings)
- tone: one of "positive", "negative", "neutral", "mixed"
- reasoning: 1-2 sentence explanation of the score

IMPORTANT:
- Base your assessment on the SUBSTANCE of the text, not just word choice
- Consider hedging language, forward-looking statements, and management tone
- A score of 0.0 means truly neutral, not uncertain
- Higher confidence for clear, decisive language; lower for ambiguous

Respond with ONLY the JSON object, no markdown formatting or explanation.

--- FILING TEXT ---
{text}
--- END TEXT ---"""

MAX_TEXT_LENGTH = 50_000
TRUNCATION_NOTE = "\n[... text truncated for length ...]\n"


# --- Provider Protocol ---


@runtime_checkable
class LLMProviderBackend(Protocol):
    """Protocol for LLM API backends."""

    @property
    def name(self) -> str: ...

    async def query(self, prompt: str, max_tokens: int = 1024) -> str: ...


# --- Provider Implementations ---


class ClaudeCLIProvider:
    """LLM provider using the local `claude` CLI binary.

    Runs `claude -p <prompt> --output-format json` as a subprocess.
    The CLI handles authentication via the user's existing session.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        timeout_seconds: int = 60,
    ) -> None:
        self._model = model
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return "claude_cli"

    async def query(self, prompt: str, max_tokens: int = 1024) -> str:
        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--model", self._model,
            "--max-turns", "1",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            raise LLMError(
                f"Claude CLI timed out after {self._timeout}s",
                context={"provider": self.name, "timeout": self._timeout},
            )
        except FileNotFoundError:
            raise LLMError(
                "Claude CLI not found. Install it: npm install -g @anthropic-ai/claude-code",
                context={"provider": self.name},
            )

        if process.returncode != 0:
            raise LLMError(
                f"Claude CLI exited with code {process.returncode}: "
                f"{stderr.decode('utf-8', errors='replace')[:500]}",
                context={
                    "provider": self.name,
                    "return_code": process.returncode,
                },
            )

        raw = stdout.decode("utf-8")
        try:
            data = json.loads(raw)
            return data.get("result", raw)
        except json.JSONDecodeError:
            return raw


class AnthropicAPIProvider:
    """LLM provider using the Anthropic Messages API.

    Requires: pip install anthropic
    Authentication: ANTHROPIC_API_KEY environment variable or explicit key.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        timeout_seconds: int = 60,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise LLMError(
                "anthropic package not installed. "
                "Install with: pip install edgar-sentinel[anthropic]",
                context={"provider": "anthropic"},
            )
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=timeout_seconds,
        )
        self._model = model

    @property
    def name(self) -> str:
        return "anthropic"

    async def query(self, prompt: str, max_tokens: int = 1024) -> str:
        import anthropic

        try:
            message = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except anthropic.APIStatusError as e:
            raise LLMError(
                f"Anthropic API error: {e.message}",
                context={
                    "provider": self.name,
                    "status_code": e.status_code,
                },
            ) from e
        except anthropic.APIConnectionError as e:
            raise LLMError(
                f"Anthropic connection error: {e}",
                context={"provider": self.name},
            ) from e


class OpenAIAPIProvider:
    """LLM provider using the OpenAI Chat Completions API.

    Requires: pip install openai
    Authentication: OPENAI_API_KEY environment variable or explicit key.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout_seconds: int = 60,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise LLMError(
                "openai package not installed. "
                "Install with: pip install edgar-sentinel[openai]",
                context={"provider": "openai"},
            )
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
        )
        self._model = model

    @property
    def name(self) -> str:
        return "openai"

    async def query(self, prompt: str, max_tokens: int = 1024) -> str:
        import openai

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except openai.APIStatusError as e:
            raise LLMError(
                f"OpenAI API error: {e.message}",
                context={
                    "provider": self.name,
                    "status_code": e.status_code,
                },
            ) from e
        except openai.APIConnectionError as e:
            raise LLMError(
                f"OpenAI connection error: {e}",
                context={"provider": self.name},
            ) from e


# --- Provider Factory ---


def create_provider(config: LLMAnalyzerConfig) -> LLMProviderBackend:
    """Create an LLM provider backend from config."""
    if config.provider == LLMProvider.CLAUDE_CLI:
        return ClaudeCLIProvider(
            model=config.model,
            timeout_seconds=config.timeout_seconds,
        )
    elif config.provider == LLMProvider.ANTHROPIC:
        return AnthropicAPIProvider(
            model=config.model,
            timeout_seconds=config.timeout_seconds,
        )
    elif config.provider == LLMProvider.OPENAI:
        return OpenAIAPIProvider(
            model=config.model,
            timeout_seconds=config.timeout_seconds,
        )
    else:
        raise LLMError(
            f"Unknown LLM provider: {config.provider}",
            context={"provider": str(config.provider)},
        )


# --- Response Parsing ---


class LLMResponse:
    """Parsed and validated LLM response. Immutable after construction."""

    __slots__ = (
        "sentiment_score", "confidence", "themes", "tone", "reasoning",
    )

    def __init__(
        self,
        sentiment_score: float,
        confidence: float,
        themes: list[str],
        tone: str,
        reasoning: str,
    ) -> None:
        self.sentiment_score = sentiment_score
        self.confidence = confidence
        self.themes = themes
        self.tone = tone
        self.reasoning = reasoning


def parse_llm_response(raw: str) -> LLMResponse:
    """Parse and validate LLM response JSON.

    Handles markdown code fences, whitespace, etc.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMError(
            f"Failed to parse LLM response as JSON: {e}",
            context={
                "response_body": raw[:500],
                "parse_error": str(e),
            },
        ) from e

    return _validate_response(data, raw)


def _validate_response(data: Any, raw: str) -> LLMResponse:
    """Validate parsed JSON against expected schema."""
    if not isinstance(data, dict):
        raise LLMError(
            f"Expected JSON object, got {type(data).__name__}",
            context={"response_body": raw[:500]},
        )

    score = data.get("sentiment_score")
    if not isinstance(score, (int, float)):
        raise LLMError(
            f"Missing or invalid sentiment_score: {score!r}",
            context={"response_body": raw[:500]},
        )
    score = float(score)
    if not -1.0 <= score <= 1.0:
        raise LLMError(
            f"sentiment_score out of range [-1, 1]: {score}",
            context={"response_body": raw[:500]},
        )

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = float(confidence)
    confidence = max(0.0, min(1.0, confidence))

    themes = data.get("themes", [])
    if not isinstance(themes, list):
        themes = []
    themes = [str(t) for t in themes[:10]]

    tone = str(data.get("tone", "neutral"))
    if tone not in ("positive", "negative", "neutral", "mixed"):
        tone = "neutral"

    reasoning = str(data.get("reasoning", ""))

    return LLMResponse(
        sentiment_score=score,
        confidence=confidence,
        themes=themes,
        tone=tone,
        reasoning=reasoning,
    )


# --- LLM Analyzer ---


class LLMAnalyzer:
    """LLM-based sentiment analyzer.

    Sends filing section text to an LLM and parses the structured
    sentiment response. Supports configurable providers and retry logic.
    """

    def __init__(self, config: LLMAnalyzerConfig) -> None:
        self._config = config
        self._provider = create_provider(config)
        self._max_retries = 3
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

    @property
    def name(self) -> str:
        return "llm"

    @property
    def analyzer_type(self) -> AnalyzerType:
        return AnalyzerType.LLM

    def analyze(
        self,
        section: FilingSection,
        prior_section: FilingSection | None = None,
    ) -> SentimentResult:
        """Analyze a single section using the LLM.

        Synchronous wrapper around the async implementation.
        """
        return asyncio.run(self._analyze_async(section))

    async def _analyze_async(self, section: FilingSection) -> SentimentResult:
        """Async implementation of single-section analysis."""
        text = section.raw_text.strip()
        if not text:
            raise AnalysisError(
                "Empty section text",
                context={
                    "analyzer_name": self.name,
                    "filing_id": section.filing_id,
                    "section_name": section.section_name,
                },
            )

        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH] + TRUNCATION_NOTE

        prompt = SENTIMENT_PROMPT_TEMPLATE.format(
            section_name=section.section_name,
            form_type="10-K/10-Q",
            text=text,
        )

        response = await self._query_with_retries(prompt, section)

        return SentimentResult(
            filing_id=section.filing_id,
            section_name=section.section_name,
            analyzer_name=self.name,
            sentiment_score=response.sentiment_score,
            confidence=response.confidence,
            metadata={
                "themes": response.themes,
                "tone": response.tone,
                "reasoning": response.reasoning,
                "provider": self._provider.name,
                "model": self._config.model,
                "text_length": len(text),
                "truncated": len(section.raw_text) > MAX_TEXT_LENGTH,
            },
            analyzed_at=datetime.now(timezone.utc),
        )

    async def _query_with_retries(
        self,
        prompt: str,
        section: FilingSection,
    ) -> LLMResponse:
        """Query the LLM with exponential backoff retries."""
        last_error: LLMError | None = None
        for attempt in range(self._max_retries):
            try:
                raw_response = await self._provider.query(prompt)
                return parse_llm_response(raw_response)
            except LLMError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    backoff = 2 ** (attempt + 1)
                    logger.warning(
                        "LLM attempt %d/%d failed for %s/%s: %s. "
                        "Retrying in %ds.",
                        attempt + 1, self._max_retries,
                        section.filing_id, section.section_name,
                        e, backoff,
                    )
                    await asyncio.sleep(backoff)

        raise LLMError(
            f"LLM analysis failed after {self._max_retries} attempts: "
            f"{last_error}",
            context={
                "provider": self._provider.name,
                "filing_id": section.filing_id,
                "section_name": section.section_name,
                "attempts": self._max_retries,
            },
        )

    def analyze_batch(
        self,
        sections: list[tuple[FilingSection, FilingSection | None]],
    ) -> list[SentimentResult]:
        """Analyze multiple sections with concurrent LLM requests."""
        return asyncio.run(self._analyze_batch_async(sections))

    async def _analyze_batch_async(
        self,
        sections: list[tuple[FilingSection, FilingSection | None]],
    ) -> list[SentimentResult]:
        """Async batch implementation with concurrency control."""
        tasks = [
            self._analyze_with_semaphore(section)
            for section, _ in sections
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[SentimentResult] = []
        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                section = sections[i][0]
                logger.warning(
                    "LLM analysis failed for %s/%s: %s",
                    section.filing_id, section.section_name, result,
                )
            else:
                results.append(result)
        return results

    async def _analyze_with_semaphore(
        self, section: FilingSection
    ) -> SentimentResult:
        """Rate-limited async analysis."""
        async with self._semaphore:
            return await self._analyze_async(section)
