"""Loughran-McDonald dictionary-based sentiment scorer."""

from __future__ import annotations

import csv
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from edgar_sentinel.core import (
    AnalysisError,
    AnalyzerType,
    FilingSection,
    SentimentResult,
)
from edgar_sentinel.core.config import DictionaryAnalyzerConfig

logger = logging.getLogger("edgar_sentinel.analyzers.dictionary")

# Category columns in the LM Master Dictionary CSV
LM_CATEGORIES = [
    "negative",
    "positive",
    "uncertainty",
    "litigious",
    "constraining",
    "strong_modal",
    "weak_modal",
]


class WordCategory:
    """Pre-compiled word->category lookup."""

    __slots__ = ("word", "categories")

    def __init__(self, word: str, categories: frozenset[str]) -> None:
        self.word = word
        self.categories = categories


class LMDictionary:
    """Loaded Loughran-McDonald dictionary.

    Immutable after construction. Thread-safe for concurrent reads.
    """

    def __init__(self, words: dict[str, frozenset[str]]) -> None:
        self._words = words  # word (uppercase) -> frozenset of category names

    @classmethod
    def from_csv(cls, path: str | Path) -> LMDictionary:
        """Load from the LM Master Dictionary CSV file."""
        path = Path(path)
        if not path.exists():
            raise AnalysisError(
                f"LM dictionary file not found: {path}",
                context={"path": str(path)},
            )

        words: dict[str, frozenset[str]] = {}
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                col_map = _build_column_map(reader.fieldnames or [])
                for row in reader:
                    word = row.get("Word", "").upper().strip()
                    if not word:
                        continue
                    cats: set[str] = set()
                    for category, col_name in col_map.items():
                        value = row.get(col_name, "0")
                        if value and value != "0":
                            cats.add(category)
                    if cats:
                        words[word] = frozenset(cats)
        except (OSError, csv.Error) as e:
            raise AnalysisError(
                f"Failed to load LM dictionary: {e}",
                context={"path": str(path)},
            ) from e

        logger.info("Loaded LM dictionary: %d classified words", len(words))
        return cls(words)

    def lookup(self, word: str) -> frozenset[str]:
        """Return categories for a word, or empty frozenset if not in dictionary."""
        return self._words.get(word.upper(), frozenset())

    @property
    def word_count(self) -> int:
        return len(self._words)


def _build_column_map(fieldnames: list[str]) -> dict[str, str]:
    """Map our category names to the actual CSV column names."""
    col_map: dict[str, str] = {}
    lower_fields = {f.lower().replace(" ", "_"): f for f in fieldnames}
    for category in LM_CATEGORIES:
        if category in lower_fields:
            col_map[category] = lower_fields[category]
        else:
            for lf, original in lower_fields.items():
                if category in lf:
                    col_map[category] = original
                    break
    return col_map


class DictionaryAnalyzer:
    """Loughran-McDonald dictionary-based sentiment scorer."""

    def __init__(self, config: DictionaryAnalyzerConfig) -> None:
        self._config = config
        self._dictionary = LMDictionary.from_csv(config.dictionary_path)

    @property
    def name(self) -> str:
        return "dictionary"

    @property
    def analyzer_type(self) -> AnalyzerType:
        return AnalyzerType.DICTIONARY

    def analyze(
        self,
        section: FilingSection,
        prior_section: FilingSection | None = None,
    ) -> SentimentResult:
        """Score a filing section using the LM dictionary."""
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

        tokens = _tokenize(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            raise AnalysisError(
                "No valid tokens after tokenization",
                context={
                    "analyzer_name": self.name,
                    "filing_id": section.filing_id,
                    "section_name": section.section_name,
                },
            )

        counts = _count_categories(tokens, self._dictionary)

        positive = counts.get("positive", 0)
        negative = counts.get("negative", 0)

        net_sentiment = (positive - negative) / total_tokens
        net_sentiment = max(-1.0, min(1.0, net_sentiment))

        sentiment_tokens = positive + negative
        confidence = min(1.0, sentiment_tokens / total_tokens)

        return SentimentResult(
            filing_id=section.filing_id,
            section_name=section.section_name,
            analyzer_name=self.name,
            sentiment_score=net_sentiment,
            confidence=confidence,
            metadata={
                "total_tokens": total_tokens,
                "category_counts": counts,
                "positive_ratio": positive / total_tokens,
                "negative_ratio": negative / total_tokens,
            },
            analyzed_at=datetime.now(timezone.utc),
        )

    def analyze_batch(
        self,
        sections: list[tuple[FilingSection, FilingSection | None]],
    ) -> list[SentimentResult]:
        """Analyze multiple sections sequentially."""
        results: list[SentimentResult] = []
        for section, prior in sections:
            try:
                results.append(self.analyze(section, prior))
            except AnalysisError as e:
                logger.warning(
                    "Dictionary analysis failed for %s/%s: %s",
                    section.filing_id,
                    section.section_name,
                    e,
                )
        return results


# --- Internal helper functions ---

_TOKEN_PATTERN = re.compile(r"[A-Z]{2,}")


def _tokenize(text: str) -> list[str]:
    """Tokenize text into uppercase words (min 2 chars)."""
    return _TOKEN_PATTERN.findall(text.upper())


def _count_categories(
    tokens: list[str],
    dictionary: LMDictionary,
) -> dict[str, int]:
    """Count token matches per sentiment category."""
    counts: dict[str, int] = {}
    for token in tokens:
        categories = dictionary.lookup(token)
        for cat in categories:
            counts[cat] = counts.get(cat, 0) + 1
    return counts
