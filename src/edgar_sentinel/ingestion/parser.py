"""Filing HTML/text parser for extracting structured sections from SEC filings."""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone
from typing import ClassVar

from bs4 import BeautifulSoup

from edgar_sentinel.core.models import (
    FilingSection,
    FormType,
    SectionName,
    SectionType,
)

logger = logging.getLogger(__name__)


class FilingParser:
    """Extracts named sections from SEC filing documents.

    Supports 10-K and 10-Q filings. Section extraction uses a ranked list
    of regex patterns per section type, with fallbacks for common formatting
    variations across different filers and years.

    The parser is intentionally conservative: if a section cannot be
    confidently identified, it returns None rather than guessing.
    """

    # Section header patterns (priority-ordered per section type)
    SECTION_PATTERNS: ClassVar[dict[SectionType, list[re.Pattern[str]]]] = {
        SectionType.MDA: [
            re.compile(
                r"(?:Item\s+7[\.\s]*[-—]?\s*)"
                r"Management[''\u2019]?s?\s+Discussion\s+and\s+Analysis",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:Item\s+2[\.\s]*[-—]?\s*)"
                r"Management[''\u2019]?s?\s+Discussion\s+and\s+Analysis",
                re.IGNORECASE,
            ),
            re.compile(
                r"Management[''\u2019]?s?\s+Discussion\s+and\s+Analysis",
                re.IGNORECASE,
            ),
            re.compile(
                r"MANAGEMENT[''\u2019]?S?\s+DISCUSSION\s+AND\s+ANALYSIS",
            ),
            re.compile(
                r"Item\s+7[^A-Z]*$",
                re.MULTILINE | re.IGNORECASE,
            ),
            re.compile(
                r"MD\s*&\s*A",
                re.IGNORECASE,
            ),
        ],
        SectionType.RISK_FACTORS: [
            re.compile(
                r"(?:Item\s+1A[\.\s]*[-—]?\s*)?Risk\s+Factors",
                re.IGNORECASE,
            ),
            re.compile(
                r"Item\s+1A[^A-Z]*$",
                re.MULTILINE | re.IGNORECASE,
            ),
        ],
        SectionType.BUSINESS: [
            re.compile(
                r"(?:Item\s+1[\.\s]*[-—]?\s*)?Business\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"Item\s+1[^A-Z0-9]*$",
                re.MULTILINE | re.IGNORECASE,
            ),
        ],
        SectionType.FINANCIAL: [
            re.compile(
                r"(?:Item\s+8[\.\s]*[-—]?\s*)"
                r"Financial\s+Statements\s+and\s+Supplementary\s+Data",
                re.IGNORECASE,
            ),
            re.compile(
                r"Notes?\s+to\s+(?:Consolidated\s+)?Financial\s+Statements",
                re.IGNORECASE,
            ),
        ],
    }

    # Section order in filings — used to determine where a section ends
    SECTION_ORDER_10K: ClassVar[list[SectionType]] = [
        SectionType.BUSINESS,       # Item 1
        SectionType.RISK_FACTORS,   # Item 1A
        SectionType.MDA,            # Item 7
        SectionType.FINANCIAL,      # Item 8
    ]

    SECTION_ORDER_10Q: ClassVar[list[SectionType]] = [
        SectionType.FINANCIAL,      # Part I, Item 1
        SectionType.MDA,            # Part I, Item 2
        SectionType.RISK_FACTORS,   # Part II, Item 1A
    ]

    # Which sections apply to each form type
    FORM_SECTIONS: ClassVar[dict[FormType, list[SectionType]]] = {
        FormType.FORM_10K: [
            SectionType.MDA,
            SectionType.RISK_FACTORS,
            SectionType.BUSINESS,
            SectionType.FINANCIAL,
        ],
        FormType.FORM_10Q: [
            SectionType.MDA,
            SectionType.RISK_FACTORS,
            SectionType.FINANCIAL,
        ],
        FormType.FORM_10K_A: [
            SectionType.MDA,
            SectionType.RISK_FACTORS,
            SectionType.BUSINESS,
            SectionType.FINANCIAL,
        ],
        FormType.FORM_10Q_A: [
            SectionType.MDA,
            SectionType.RISK_FACTORS,
            SectionType.FINANCIAL,
        ],
    }

    # All section start patterns (used to find the end of an extracted section)
    _ALL_SECTION_HEADERS: ClassVar[list[re.Pattern[str]]] = [
        # Item numbers that mark section boundaries
        re.compile(r"Item\s+1A[\.\s]*[-—]?\s*Risk\s+Factors", re.IGNORECASE),
        re.compile(r"Item\s+1B[\.\s]*[-—]?\s*Unresolved\s+Staff", re.IGNORECASE),
        re.compile(r"Item\s+1[\.\s]*[-—]?\s*Business\b", re.IGNORECASE),
        re.compile(r"Item\s+2[\.\s]*[-—]?\s*Management[''\u2019]?s?\s+Discussion", re.IGNORECASE),
        re.compile(r"Item\s+2[\.\s]*[-—]?\s*Properties", re.IGNORECASE),
        re.compile(r"Item\s+3[\.\s]*[-—]?\s*Legal\s+Proceedings", re.IGNORECASE),
        re.compile(r"Item\s+4[\.\s]*[-—]?\s*Mine\s+Safety", re.IGNORECASE),
        re.compile(r"Item\s+5[\.\s]*[-—]?\s*Market\s+for", re.IGNORECASE),
        re.compile(r"Item\s+6[\.\s]*[-—]?\s*(?:Selected|Reserved)", re.IGNORECASE),
        re.compile(
            r"Item\s+7A[\.\s]*[-—]?\s*Quantitative\s+and\s+Qualitative",
            re.IGNORECASE,
        ),
        re.compile(
            r"Item\s+7[\.\s]*[-—]?\s*Management[''\u2019]?s?\s+Discussion",
            re.IGNORECASE,
        ),
        re.compile(
            r"Item\s+8[\.\s]*[-—]?\s*Financial\s+Statements",
            re.IGNORECASE,
        ),
        re.compile(r"Item\s+9[\.\s]*[-—]?\s*Changes\s+in", re.IGNORECASE),
        re.compile(r"Item\s+10[\.\s]*[-—]?\s*Directors", re.IGNORECASE),
        re.compile(r"Item\s+11[\.\s]*[-—]?\s*Executive\s+Compensation", re.IGNORECASE),
        re.compile(r"Item\s+12[\.\s]*[-—]?\s*Security\s+Ownership", re.IGNORECASE),
        re.compile(r"Item\s+13[\.\s]*[-—]?\s*Certain\s+Relationships", re.IGNORECASE),
        re.compile(r"Item\s+14[\.\s]*[-—]?\s*Principal\s+Account", re.IGNORECASE),
        re.compile(r"Item\s+15[\.\s]*[-—]?\s*Exhibits", re.IGNORECASE),
        re.compile(r"Part\s+(?:I|II|III|IV)\b", re.IGNORECASE),
        re.compile(r"SIGNATURES?\s*$", re.MULTILINE | re.IGNORECASE),
    ]

    def __init__(
        self,
        strip_tables: bool = True,
        min_section_words: int = 50,
    ) -> None:
        """
        Args:
            strip_tables: If True, remove HTML tables from extracted text.
            min_section_words: Minimum word count for a section to be valid.
        """
        self.strip_tables = strip_tables
        self.min_section_words = min_section_words

    def parse(
        self,
        raw_html: str,
        form_type: FormType,
        accession_number: str,
    ) -> dict[SectionName, FilingSection]:
        """Parse a filing document and extract all available sections.

        Args:
            raw_html: Raw filing document content (HTML or plain text).
            form_type: Filing form type (determines which sections to extract).
            accession_number: Filing accession number (used in FilingSection.filing_id).

        Returns:
            dict mapping section names to FilingSection objects.
            Only successfully extracted sections are included.
        """
        clean = self.clean_text(raw_html)
        applicable = self.FORM_SECTIONS.get(form_type, [])
        now = datetime.now(timezone.utc)
        result: dict[SectionName, FilingSection] = {}

        for section_type in applicable:
            text = self.extract_section(clean, section_type, form_type)
            if text is None:
                logger.debug(
                    "Section %s not found in %s (%s)",
                    section_type.value,
                    accession_number,
                    form_type.value,
                )
                continue

            word_count = len(text.split())
            result[section_type.value] = FilingSection(
                filing_id=accession_number,
                section_name=section_type.value,
                raw_text=text,
                word_count=word_count,
                extracted_at=now,
            )

        return result

    def extract_section(
        self,
        clean_text: str,
        section: SectionType,
        form_type: FormType,
    ) -> str | None:
        """Extract a single named section from cleaned filing text.

        Args:
            clean_text: Pre-cleaned plain text (output of clean_text()).
            section: The section type to extract.
            form_type: Form type (affects section ordering).

        Returns:
            Extracted section text, or None if not found or too short.
        """
        patterns = self.SECTION_PATTERNS.get(section, [])
        if not patterns:
            return None

        # Find section start using ranked patterns
        start_match = None
        for pattern in patterns:
            match = pattern.search(clean_text)
            if match:
                start_match = match
                break

        if start_match is None:
            return None

        # Content starts after the header line
        section_start = start_match.end()

        # Find section end: look for the next section header after our start
        section_end = self._find_section_end(clean_text, section_start, section, form_type)

        extracted = clean_text[section_start:section_end].strip()

        # Validate minimum length
        if len(extracted.split()) < self.min_section_words:
            logger.debug(
                "Section %s too short (%d words < %d minimum)",
                section.value,
                len(extracted.split()),
                self.min_section_words,
            )
            return None

        # Normalize whitespace in final output
        extracted = re.sub(r"\n{3,}", "\n\n", extracted)
        return extracted

    def _find_section_end(
        self,
        text: str,
        start_pos: int,
        current_section: SectionType,
        form_type: FormType,
    ) -> int:
        """Find where a section ends by looking for the next section header.

        Searches for any known section header pattern that starts after start_pos,
        excluding patterns that match the current section itself.
        """
        # Get the section order for this form type
        if form_type in (FormType.FORM_10K, FormType.FORM_10K_A):
            section_order = self.SECTION_ORDER_10K
        else:
            section_order = self.SECTION_ORDER_10Q

        # Find the position of the current section in the order
        try:
            current_idx = section_order.index(current_section)
        except ValueError:
            current_idx = -1

        # Look for any subsequent section header
        earliest_end = len(text)
        current_patterns = set()
        for p in self.SECTION_PATTERNS.get(current_section, []):
            current_patterns.add(p.pattern)

        for header_pattern in self._ALL_SECTION_HEADERS:
            # Skip patterns that would match our current section
            if header_pattern.pattern in current_patterns:
                continue

            match = header_pattern.search(text, start_pos)
            if match and match.start() < earliest_end:
                # Only use this as an end marker if it's sufficiently far from start
                # (at least 100 chars) to avoid matching within the header itself
                if match.start() - start_pos > 100:
                    earliest_end = match.start()

        return earliest_end

    def clean_text(self, raw_html: str) -> str:
        """Convert raw filing HTML to clean plain text.

        Args:
            raw_html: Raw document content (HTML or plain text).

        Returns:
            Cleaned plain text.
        """
        if not raw_html or not raw_html.strip():
            return ""

        if self._is_html(raw_html):
            soup = BeautifulSoup(raw_html, "lxml")

            # Remove script and style tags
            for tag in soup.find_all(["script", "style"]):
                tag.decompose()

            # Remove XBRL tags (preserve their text content)
            self._remove_xbrl(soup)

            # Optionally strip tables
            if self.strip_tables:
                for table in soup.find_all("table"):
                    table.decompose()

            text = soup.get_text(separator="\n")
        else:
            text = raw_html

        # Decode HTML entities
        text = html.unescape(text)

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces/tabs → single space
        text = re.sub(r"\n{3,}", "\n\n", text)  # 3+ newlines → 2

        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Remove boilerplate
        text = self._strip_boilerplate(text)

        return text.strip()

    def _is_html(self, content: str) -> bool:
        """Heuristic: does this content appear to be HTML?

        Checks for presence of common HTML tags. A simple tag count
        threshold avoids false positives on text containing '<'.
        """
        html_tags = re.findall(r"<(?:html|body|div|p|table|span|br|head)\b", content[:5000], re.IGNORECASE)
        return len(html_tags) >= 2

    def _remove_xbrl(self, soup: BeautifulSoup) -> None:
        """Remove XBRL inline tags while preserving their text content.

        XBRL wraps financial figures in tags like <ix:nonFraction> that
        clutter extraction but contain useful text.
        """
        for tag in soup.find_all(re.compile(r"^(?:ix|xbrli|xbrl):", re.IGNORECASE)):
            tag.unwrap()

    def _strip_boilerplate(self, text: str) -> str:
        """Remove common boilerplate patterns from cleaned text.

        Removes:
        - "Table of Contents" lines
        - Page number lines like "- 42 -" or "Page 42 of 150"
        - Form feed characters
        """
        # Table of contents references
        text = re.sub(r"^\s*Table\s+of\s+Contents\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

        # Page numbers: "- 42 -" or "Page 42 of 150" or standalone numbers
        text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

        # Form feed characters
        text = text.replace("\f", "\n")

        return text
