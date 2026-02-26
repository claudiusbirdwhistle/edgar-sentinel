"""Tests for the Loughran-McDonald dictionary analyzer."""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from edgar_sentinel.core import AnalysisError, AnalyzerType, FilingSection, SentimentResult
from edgar_sentinel.core.config import DictionaryAnalyzerConfig
from edgar_sentinel.analyzers.base import Analyzer
from edgar_sentinel.analyzers.dictionary import (
    DictionaryAnalyzer,
    LMDictionary,
    LM_CATEGORIES,
    _build_column_map,
    _count_categories,
    _tokenize,
)


NOW = datetime.now(timezone.utc)

# --- Test CSV Data ---

TEST_DICTIONARY_ROWS = [
    # word, Negative, Positive, Uncertainty, Litigious, Constraining, Strong_Modal, Weak_Modal
    ("ABANDON", "2009", "0", "0", "0", "0", "0", "0"),
    ("ABLE", "0", "0", "0", "0", "0", "0", "0"),  # no categories
    ("ACCOMPLISH", "0", "2009", "0", "0", "0", "0", "0"),
    ("ADVANTAGE", "0", "2009", "0", "0", "0", "0", "0"),
    ("ADVERSE", "2009", "0", "0", "0", "0", "0", "0"),
    ("APPROXIMATE", "0", "0", "2009", "0", "0", "0", "0"),
    ("BENEFIT", "0", "2009", "0", "0", "0", "0", "0"),
    ("COMMIT", "0", "0", "0", "0", "2009", "0", "0"),
    ("COULD", "0", "0", "0", "0", "0", "0", "2009"),
    ("DECLINE", "2009", "0", "0", "0", "0", "0", "0"),
    ("GAIN", "0", "2009", "0", "0", "0", "0", "0"),
    ("IMPAIRMENT", "2009", "0", "0", "0", "0", "0", "0"),
    ("IMPROVEMENT", "0", "2009", "0", "0", "0", "0", "0"),
    ("LAWSUIT", "0", "0", "0", "2009", "0", "0", "0"),
    ("LITIGATION", "0", "0", "0", "2009", "0", "0", "0"),
    ("LOSS", "2009", "0", "0", "0", "0", "0", "0"),
    ("MAY", "0", "0", "0", "0", "0", "0", "2009"),
    ("MUST", "0", "0", "0", "0", "0", "2009", "0"),
    ("UNCERTAIN", "0", "0", "2009", "0", "0", "0", "0"),
    ("WILL", "0", "0", "0", "0", "0", "2009", "0"),
    # Multi-category word
    ("OBLIGATE", "2009", "0", "0", "2009", "2009", "0", "0"),
]

CSV_HEADER = ["Word", "Negative", "Positive", "Uncertainty", "Litigious",
              "Constraining", "Strong_Modal", "Weak_Modal"]


@pytest.fixture
def dict_csv(tmp_path):
    """Create a test LM dictionary CSV file."""
    csv_path = tmp_path / "lm_test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in TEST_DICTIONARY_ROWS:
            writer.writerow(row)
    return csv_path


@pytest.fixture
def lm_dict(dict_csv):
    """Load a test LMDictionary."""
    return LMDictionary.from_csv(dict_csv)


@pytest.fixture
def analyzer(dict_csv):
    """Create a DictionaryAnalyzer with test dictionary."""
    config = DictionaryAnalyzerConfig(dictionary_path=str(dict_csv))
    return DictionaryAnalyzer(config)


def make_section(text="hello world", filing_id="0000320193-23-000106", section_name="mda"):
    return FilingSection(
        filing_id=filing_id,
        section_name=section_name,
        raw_text=text,
        word_count=len(text.split()),
        extracted_at=NOW,
    )


# --- _build_column_map Tests ---


class TestBuildColumnMap:
    def test_standard_columns(self):
        cols = CSV_HEADER
        result = _build_column_map(cols)
        assert result["negative"] == "Negative"
        assert result["positive"] == "Positive"
        assert result["uncertainty"] == "Uncertainty"
        assert result["litigious"] == "Litigious"
        assert result["constraining"] == "Constraining"
        assert result["strong_modal"] == "Strong_Modal"
        assert result["weak_modal"] == "Weak_Modal"

    def test_lowercase_columns(self):
        cols = ["Word", "negative", "positive", "uncertainty",
                "litigious", "constraining", "strong_modal", "weak_modal"]
        result = _build_column_map(cols)
        assert len(result) == 7

    def test_empty_fieldnames(self):
        result = _build_column_map([])
        assert result == {}

    def test_partial_columns(self):
        cols = ["Word", "Negative", "Positive"]
        result = _build_column_map(cols)
        assert "negative" in result
        assert "positive" in result
        assert "uncertainty" not in result


# --- _tokenize Tests ---


class TestTokenize:
    def test_basic_text(self):
        tokens = _tokenize("The company reported a significant loss")
        assert "THE" in tokens
        assert "COMPANY" in tokens
        assert "LOSS" in tokens

    def test_single_char_filtered(self):
        tokens = _tokenize("I am a test x y z")
        # Single chars should be filtered out
        assert "I" not in tokens
        assert "AM" in tokens

    def test_numbers_excluded(self):
        tokens = _tokenize("Revenue was 1234 million in 2023")
        assert "REVENUE" in tokens
        assert "1234" not in tokens
        assert "2023" not in tokens

    def test_punctuation_splits(self):
        tokens = _tokenize("loss/gain; decline, improvement.")
        assert "LOSS" in tokens
        assert "GAIN" in tokens
        assert "DECLINE" in tokens
        assert "IMPROVEMENT" in tokens

    def test_empty_text(self):
        assert _tokenize("") == []

    def test_only_numbers_and_symbols(self):
        assert _tokenize("123 456 @#$") == []

    def test_case_insensitive(self):
        tokens = _tokenize("Loss LOSS loss")
        assert tokens == ["LOSS", "LOSS", "LOSS"]

    def test_hyphenated_words(self):
        tokens = _tokenize("year-over-year")
        assert "YEAR" in tokens
        assert "OVER" in tokens


# --- LMDictionary Tests ---


class TestLMDictionary:
    def test_load_from_csv(self, lm_dict):
        # "ABLE" has no categories, so shouldn't be in dictionary
        assert lm_dict.word_count == len(TEST_DICTIONARY_ROWS) - 1  # minus ABLE

    def test_lookup_negative_word(self, lm_dict):
        cats = lm_dict.lookup("LOSS")
        assert "negative" in cats

    def test_lookup_positive_word(self, lm_dict):
        cats = lm_dict.lookup("BENEFIT")
        assert "positive" in cats

    def test_lookup_multi_category(self, lm_dict):
        cats = lm_dict.lookup("OBLIGATE")
        assert "negative" in cats
        assert "litigious" in cats
        assert "constraining" in cats

    def test_lookup_unknown_word(self, lm_dict):
        cats = lm_dict.lookup("XYZZY")
        assert cats == frozenset()

    def test_lookup_case_insensitive(self, lm_dict):
        assert lm_dict.lookup("loss") == lm_dict.lookup("LOSS")

    def test_file_not_found(self, tmp_path):
        with pytest.raises(AnalysisError, match="not found"):
            LMDictionary.from_csv(tmp_path / "nonexistent.csv")

    def test_invalid_csv(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_bytes(b"\x00\x01\x02")
        # Should raise or produce empty dictionary depending on csv module behavior
        # The CSV module may handle binary differently — just verify no crash
        try:
            d = LMDictionary.from_csv(bad_csv)
            # If it loads, it should have 0 words
            assert d.word_count == 0
        except AnalysisError:
            pass  # Expected

    def test_empty_csv(self, tmp_path):
        empty_csv = tmp_path / "empty.csv"
        with open(empty_csv, "w") as f:
            f.write("Word,Negative,Positive\n")
        d = LMDictionary.from_csv(empty_csv)
        assert d.word_count == 0


# --- _count_categories Tests ---


class TestCountCategories:
    def test_count_basic(self, lm_dict):
        tokens = ["LOSS", "BENEFIT", "GAIN", "DECLINE"]
        counts = _count_categories(tokens, lm_dict)
        assert counts["negative"] == 2  # LOSS, DECLINE
        assert counts["positive"] == 2  # BENEFIT, GAIN

    def test_count_empty_tokens(self, lm_dict):
        counts = _count_categories([], lm_dict)
        assert counts == {}

    def test_count_no_matches(self, lm_dict):
        counts = _count_categories(["HELLO", "WORLD", "TESTING"], lm_dict)
        assert counts == {}

    def test_count_multi_category(self, lm_dict):
        counts = _count_categories(["OBLIGATE"], lm_dict)
        assert counts["negative"] == 1
        assert counts["litigious"] == 1
        assert counts["constraining"] == 1

    def test_count_repeated_words(self, lm_dict):
        tokens = ["LOSS", "LOSS", "LOSS"]
        counts = _count_categories(tokens, lm_dict)
        assert counts["negative"] == 3


# --- DictionaryAnalyzer Tests ---


class TestDictionaryAnalyzer:
    def test_satisfies_protocol(self, analyzer):
        assert isinstance(analyzer, Analyzer)

    def test_name(self, analyzer):
        assert analyzer.name == "dictionary"

    def test_analyzer_type(self, analyzer):
        assert analyzer.analyzer_type == AnalyzerType.DICTIONARY

    def test_analyze_basic(self, analyzer):
        text = "The company reported a significant improvement and benefit from the acquisition"
        section = make_section(text=text)
        result = analyzer.analyze(section)
        assert isinstance(result, SentimentResult)
        assert result.filing_id == section.filing_id
        assert result.section_name == section.section_name
        assert result.analyzer_name == "dictionary"
        assert -1.0 <= result.sentiment_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_analyze_positive_text(self, analyzer):
        text = "We achieved improvement gain advantage and benefit from this"
        section = make_section(text=text)
        result = analyzer.analyze(section)
        # All sentiment words are positive, net should be > 0
        assert result.sentiment_score > 0

    def test_analyze_negative_text(self, analyzer):
        text = "The loss decline adverse impairment was significant"
        section = make_section(text=text)
        result = analyzer.analyze(section)
        # All sentiment words are negative, net should be < 0
        assert result.sentiment_score < 0

    def test_analyze_neutral_text(self, analyzer):
        """Text with no dictionary words should have 0 sentiment and 0 confidence."""
        text = "hello world testing this random text here today"
        section = make_section(text=text)
        result = analyzer.analyze(section)
        assert result.sentiment_score == 0.0
        assert result.confidence == 0.0

    def test_analyze_metadata(self, analyzer):
        text = "The loss was offset by improvement"
        section = make_section(text=text)
        result = analyzer.analyze(section)
        assert "total_tokens" in result.metadata
        assert "category_counts" in result.metadata
        assert "positive_ratio" in result.metadata
        assert "negative_ratio" in result.metadata

    def test_analyze_metadata_counts(self, analyzer):
        text = "loss decline benefit gain"
        section = make_section(text=text)
        result = analyzer.analyze(section)
        counts = result.metadata["category_counts"]
        assert counts["negative"] == 2
        assert counts["positive"] == 2

    def test_analyze_empty_text_raises(self, analyzer):
        section = make_section(text="")
        with pytest.raises(AnalysisError, match="Empty section text"):
            analyzer.analyze(section)

    def test_analyze_whitespace_only_raises(self, analyzer):
        section = make_section(text="   \n\t  ")
        with pytest.raises(AnalysisError, match="Empty section text"):
            analyzer.analyze(section)

    def test_analyze_numbers_only_raises(self, analyzer):
        section = make_section(text="123 456 789")
        with pytest.raises(AnalysisError, match="No valid tokens"):
            analyzer.analyze(section)

    def test_analyze_ignores_prior_section(self, analyzer):
        text = "The company reported a benefit"
        section = make_section(text=text)
        prior = make_section(text="Some prior text here", filing_id="0000320193-22-000106")
        result = analyzer.analyze(section, prior_section=prior)
        # Should produce same result regardless of prior
        result_no_prior = analyzer.analyze(section)
        assert result.sentiment_score == result_no_prior.sentiment_score

    def test_analyze_confidence_calculation(self, analyzer):
        # 4 tokens: loss, benefit, hello, world
        # 2 sentiment tokens (loss=neg, benefit=pos), 4 total
        # confidence = 2/4 = 0.5
        text = "loss benefit hello world"
        section = make_section(text=text)
        result = analyzer.analyze(section)
        assert result.confidence == 0.5

    def test_analyze_net_sentiment_calculation(self, analyzer):
        # 4 tokens: loss, benefit, hello, world
        # positive=1, negative=1 -> net = (1-1)/4 = 0.0
        text = "loss benefit hello world"
        section = make_section(text=text)
        result = analyzer.analyze(section)
        assert result.sentiment_score == 0.0

    def test_analyze_uncertainty_tracked(self, analyzer):
        text = "The approximate value is uncertain"
        section = make_section(text=text)
        result = analyzer.analyze(section)
        counts = result.metadata["category_counts"]
        assert counts.get("uncertainty", 0) == 2  # APPROXIMATE, UNCERTAIN

    def test_analyze_modals_tracked(self, analyzer):
        text = "The company must do this but could also consider may"
        section = make_section(text=text)
        result = analyzer.analyze(section)
        counts = result.metadata["category_counts"]
        assert counts.get("strong_modal", 0) == 1  # MUST
        assert counts.get("weak_modal", 0) == 2  # COULD, MAY

    def test_config_not_found_raises(self, tmp_path):
        config = DictionaryAnalyzerConfig(dictionary_path=str(tmp_path / "nonexistent.csv"))
        with pytest.raises(AnalysisError, match="not found"):
            DictionaryAnalyzer(config)


# --- Batch Analysis Tests ---


class TestDictionaryBatch:
    def test_batch_multiple_sections(self, analyzer):
        s1 = make_section(text="The loss was significant")
        s2 = make_section(text="The benefit was substantial", section_name="risk_factors")
        results = analyzer.analyze_batch([(s1, None), (s2, None)])
        assert len(results) == 2
        assert results[0].sentiment_score < 0  # negative
        assert results[1].sentiment_score > 0  # positive

    def test_batch_empty_list(self, analyzer):
        results = analyzer.analyze_batch([])
        assert results == []

    def test_batch_skips_failures(self, analyzer):
        good = make_section(text="The benefit was substantial")
        bad = make_section(text="123 456")  # no valid tokens
        results = analyzer.analyze_batch([(good, None), (bad, None)])
        # Only good section should produce a result
        assert len(results) == 1
        assert results[0].sentiment_score > 0

    def test_batch_all_failures(self, analyzer):
        bad1 = make_section(text="123")
        bad2 = make_section(text="456")
        results = analyzer.analyze_batch([(bad1, None), (bad2, None)])
        assert results == []
