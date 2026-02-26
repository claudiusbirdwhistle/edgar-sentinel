"""TF-IDF cosine similarity analyzer for consecutive filings."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from edgar_sentinel.core import (
    AnalysisError,
    AnalyzerType,
    FilingSection,
    SimilarityResult,
)
from edgar_sentinel.core.config import SimilarityAnalyzerConfig

logger = logging.getLogger("edgar_sentinel.analyzers.similarity")

# TF-IDF vectorizer parameters tuned for SEC filing text
TFIDF_PARAMS: dict[str, object] = {
    "max_features": 10_000,
    "min_df": 1,
    "max_df": 1.0,
    "sublinear_tf": True,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "strip_accents": "unicode",
    "lowercase": True,
    "token_pattern": r"(?u)\b[a-zA-Z]{2,}\b",
}


class SimilarityAnalyzer:
    """TF-IDF cosine similarity analyzer for consecutive filings.

    Compares the TF-IDF vector of a filing section against the same
    section from the company's previous filing. Outputs similarity_score
    (0.0-1.0) and change_score (1 - similarity).

    Requires prior_section -- returns nothing for first filings.
    """

    def __init__(self, config: SimilarityAnalyzerConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "similarity"

    @property
    def analyzer_type(self) -> AnalyzerType:
        return AnalyzerType.SIMILARITY

    def analyze(
        self,
        section: FilingSection,
        prior_section: FilingSection | None = None,
    ) -> SimilarityResult:
        """Compute cosine similarity between current and prior filing sections.

        Args:
            section: Current filing section.
            prior_section: Same section from previous filing. Required.

        Returns:
            SimilarityResult with similarity_score and change_score.

        Raises:
            AnalysisError: If prior_section is None, texts are empty,
                or TF-IDF vectorization fails.
        """
        if prior_section is None:
            raise AnalysisError(
                "Similarity analyzer requires prior_section",
                context={
                    "analyzer_name": self.name,
                    "filing_id": section.filing_id,
                    "section_name": section.section_name,
                },
            )

        current_text = section.raw_text.strip()
        prior_text = prior_section.raw_text.strip()

        if not current_text or not prior_text:
            raise AnalysisError(
                "Empty section text for similarity comparison",
                context={
                    "analyzer_name": self.name,
                    "filing_id": section.filing_id,
                    "prior_filing_id": prior_section.filing_id,
                    "section_name": section.section_name,
                },
            )

        similarity = _compute_cosine_similarity(current_text, prior_text)

        return SimilarityResult(
            filing_id=section.filing_id,
            prior_filing_id=prior_section.filing_id,
            section_name=section.section_name,
            similarity_score=similarity,
            change_score=round(1.0 - similarity, 6),
            analyzed_at=datetime.now(timezone.utc),
        )

    def analyze_batch(
        self,
        sections: list[tuple[FilingSection, FilingSection | None]],
    ) -> list[SimilarityResult]:
        """Analyze multiple section pairs.

        Sequential -- each pair gets its own TF-IDF fit.
        Pairs without prior_section are silently skipped.
        """
        results: list[SimilarityResult] = []
        for section, prior in sections:
            if prior is None:
                logger.debug(
                    "Skipping similarity for %s/%s (no prior filing)",
                    section.filing_id,
                    section.section_name,
                )
                continue
            try:
                results.append(self.analyze(section, prior))
            except AnalysisError as e:
                logger.warning(
                    "Similarity analysis failed for %s/%s: %s",
                    section.filing_id,
                    section.section_name,
                    e,
                )
        return results


def _compute_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using TF-IDF.

    Fits a TF-IDF vectorizer on the pair, computes cosine similarity
    of the resulting vectors.

    Returns:
        Float in [0.0, 1.0]. Identical texts -> 1.0; completely
        disjoint vocabularies -> 0.0.
    """
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    try:
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
    except ValueError as e:
        raise AnalysisError(
            f"TF-IDF vectorization failed: {e}",
            context={"reason": str(e)},
        ) from e

    similarity_matrix = sklearn_cosine(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity = float(similarity_matrix[0][0])

    # Clamp to [0.0, 1.0] (numerical precision)
    return max(0.0, min(1.0, round(similarity, 6)))
