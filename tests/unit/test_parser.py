"""Tests for FilingParser — section extraction from SEC filing HTML/text."""

from __future__ import annotations

from edgar_sentinel.core.models import FormType, SectionType
from edgar_sentinel.ingestion.parser import FilingParser


# --- Test fixtures: sample filing HTML fragments ---

SAMPLE_10K_HTML = """
<html>
<head><title>10-K Filing</title></head>
<body>
<div>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p>Washington, D.C. 20549</p>
<p>FORM 10-K</p>

<p>ACME CORPORATION</p>

<p><b>Item 1. Business</b></p>
<p>Acme Corporation is a global leader in innovative products. The company
was founded in 1985 and has grown to serve customers in over 50 countries.
Our primary business segments include consumer electronics, industrial
equipment, and software services. We employ approximately 15,000 people
worldwide and operate manufacturing facilities on three continents. Revenue
for the fiscal year was approximately $4.2 billion, representing growth
of 8% year over year. Our strategy focuses on organic growth supplemented
by targeted acquisitions in adjacent markets.</p>

<p><b>Item 1A. Risk Factors</b></p>
<p>Investing in our securities involves significant risks. The following
risk factors should be carefully considered. Our business faces competition
from both domestic and international companies, many of which have greater
financial resources. Market conditions may adversely affect our operating
results. Changes in government regulations could increase our costs of
compliance. Supply chain disruptions may impact our ability to deliver
products on time. Currency fluctuations affect our international revenue
streams. Cybersecurity threats pose risks to our operations and data.
Climate change and environmental regulations may require significant
capital expenditures.</p>

<p><b>Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations</b></p>
<p>The following discussion should be read in conjunction with our
consolidated financial statements. Revenue increased 8% to $4.2 billion
driven by strong demand in our consumer electronics segment. Gross margin
improved 150 basis points to 42.3% due to favorable product mix and
manufacturing efficiencies. Operating expenses increased 5% reflecting
higher research and development spending. Net income was $380 million
or $2.45 per diluted share compared to $340 million or $2.18 per diluted
share in the prior year. Cash flow from operations was $520 million.
We returned $200 million to shareholders through dividends and share
repurchases. We expect continued growth in the upcoming fiscal year.</p>

<p><b>Item 8. Financial Statements and Supplementary Data</b></p>
<table>
<tr><td>Revenue</td><td>$4,200M</td></tr>
<tr><td>Net Income</td><td>$380M</td></tr>
</table>
<p>The accompanying notes are an integral part of these consolidated
financial statements. Our financial statements have been prepared in
accordance with generally accepted accounting principles in the United
States. The audit committee has reviewed and approved these financial
statements. Independent auditors have expressed an unqualified opinion
on our consolidated financial statements for the year ended December 31.</p>

<p><b>SIGNATURES</b></p>
<p>Pursuant to the requirements of Section 13 of the Securities Exchange Act.</p>
</div>
</body>
</html>
"""

SAMPLE_10K_PLAIN_TEXT = """
FORM 10-K

ACME CORPORATION

Item 1. Business

Acme Corporation is a global leader in innovative products. The company
was founded in 1985 and has grown to serve customers in over 50 countries.
Our primary business segments include consumer electronics, industrial
equipment, and software services. We employ approximately 15,000 people
worldwide and operate manufacturing facilities on three continents. Revenue
for the fiscal year was approximately $4.2 billion, representing growth
of 8% year over year. Our strategy focuses on organic growth supplemented
by targeted acquisitions in adjacent markets.

Item 1A. Risk Factors

Investing in our securities involves significant risks. The following
risk factors should be carefully considered. Our business faces competition
from both domestic and international companies, many of which have greater
financial resources. Market conditions may adversely affect our operating
results. Changes in government regulations could increase our costs of
compliance. Supply chain disruptions may impact our ability to deliver
products on time. Currency fluctuations affect our international revenue
streams. Cybersecurity threats pose risks to our operations and data.
Climate change and environmental regulations may require significant
capital expenditures.

Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations

The following discussion should be read in conjunction with our
consolidated financial statements. Revenue increased 8% to $4.2 billion
driven by strong demand in our consumer electronics segment. Gross margin
improved 150 basis points to 42.3% due to favorable product mix and
manufacturing efficiencies. Operating expenses increased 5% reflecting
higher research and development spending. Net income was $380 million
or $2.45 per diluted share compared to $340 million or $2.18 per diluted
share in the prior year. Cash flow from operations was $520 million.
We returned $200 million to shareholders through dividends and share
repurchases. We expect continued growth in the upcoming fiscal year.

Item 8. Financial Statements and Supplementary Data

The accompanying notes are an integral part of these consolidated
financial statements. Our financial statements have been prepared in
accordance with generally accepted accounting principles in the United
States. Revenue was $4.2 billion and net income was $380 million.
The audit committee has reviewed and approved these financial statements.

SIGNATURES

Pursuant to the requirements of Section 13 of the Securities Exchange Act.
"""

SAMPLE_10Q_HTML = """
<html>
<body>
<p>FORM 10-Q</p>
<p>ACME CORPORATION</p>

<p><b>Part I - Financial Information</b></p>
<p><b>Item 1. Financial Statements</b></p>
<p>The unaudited condensed consolidated financial statements included
herein have been prepared in accordance with generally accepted accounting
principles. Revenue for the quarter was $1.1 billion representing a 6%
increase over the prior year quarter. Gross margin was 41.8% compared
to 40.5% in the prior year quarter. Operating income increased 12% to
$145 million driven by revenue growth and cost discipline. These statements
should be read in conjunction with our annual report on Form 10-K.</p>

<p><b>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</b></p>
<p>Overview of quarterly results. Revenue increased 6% to $1.1 billion.
The growth was primarily driven by strength in our consumer electronics
segment which grew 10% year over year. Industrial equipment revenue was
flat as customers delayed capital expenditure decisions. Software services
revenue grew 15% driven by subscription renewals and new customer wins.
Gross margin expanded 130 basis points to 41.8%. Operating expenses were
well controlled growing only 3% year over year. Net income was $98 million
or $0.63 per diluted share. We continue to expect full year revenue growth
of 7-9% and operating margin expansion of 50-100 basis points.</p>

<p><b>Part II - Other Information</b></p>
<p><b>Item 1A. Risk Factors</b></p>
<p>In addition to the risk factors previously disclosed in our annual
report on Form 10-K, the following additional risk factors should be
considered. Trade tensions between major economies could impact our
supply chain and increase tariff costs. The competitive landscape in
our software services segment has intensified with new market entrants.
Rising interest rates may increase our borrowing costs and impact customer
purchasing decisions. Semiconductor supply constraints continue to pose
risks to our manufacturing timelines and product delivery schedules.
These factors could materially impact our financial results.</p>

<p><b>SIGNATURES</b></p>
<p>Pursuant to the requirements of the Securities Exchange Act.</p>
</body>
</html>
"""

ACCESSION = "0001234567-24-000001"


# --- Tests: FilingParser initialization ---


class TestFilingParserInit:
    """Tests for parser construction and defaults."""

    def test_default_options(self):
        parser = FilingParser()
        assert parser.strip_tables is True
        assert parser.min_section_words == 50

    def test_custom_options(self):
        parser = FilingParser(strip_tables=False, min_section_words=100)
        assert parser.strip_tables is False
        assert parser.min_section_words == 100


# --- Tests: clean_text ---


class TestCleanText:
    """Tests for HTML cleaning and text normalization."""

    def setup_method(self):
        self.parser = FilingParser()

    def test_html_to_plain_text(self):
        html_input = "<html><body><p>Hello</p><p>World</p></body></html>"
        result = self.parser.clean_text(html_input)
        assert "Hello" in result
        assert "World" in result
        assert "<p>" not in result

    def test_strips_script_and_style_tags(self):
        html_input = """
        <html><body>
        <script>var x = 1;</script>
        <style>.foo { color: red; }</style>
        <p>Actual content here</p>
        </body></html>
        """
        result = self.parser.clean_text(html_input)
        assert "var x = 1" not in result
        assert "color: red" not in result
        assert "Actual content here" in result

    def test_strips_tables_when_enabled(self):
        html_input = """
        <html><body>
        <p>Before table</p>
        <table><tr><td>Revenue</td><td>$100M</td></tr></table>
        <p>After table</p>
        </body></html>
        """
        parser = FilingParser(strip_tables=True)
        result = parser.clean_text(html_input)
        assert "Before table" in result
        assert "After table" in result
        assert "Revenue" not in result

    def test_preserves_tables_when_disabled(self):
        html_input = """
        <html><body>
        <p>Before table</p>
        <table><tr><td>Revenue</td><td>$100M</td></tr></table>
        <p>After table</p>
        </body></html>
        """
        parser = FilingParser(strip_tables=False)
        result = parser.clean_text(html_input)
        assert "Revenue" in result
        assert "$100M" in result

    def test_plain_text_passthrough(self):
        text = "This is plain text with no HTML tags at all."
        result = self.parser.clean_text(text)
        assert result == text

    def test_whitespace_normalization(self):
        text = "Hello     world\n\n\n\n\nParagraph two"
        result = self.parser.clean_text(text)
        assert "Hello world" in result
        # 3+ newlines collapsed to 2
        assert "\n\n\n" not in result

    def test_html_entity_decoding(self):
        html_input = "<html><body><p>AT&amp;T &lt;corp&gt; &#160; value</p></body></html>"
        result = self.parser.clean_text(html_input)
        assert "AT&T" in result
        assert "<corp>" in result

    def test_removes_table_of_contents(self):
        text = "Some content\nTable of Contents\nMore content"
        result = self.parser.clean_text(text)
        assert "Some content" in result
        assert "Table of Contents" not in result
        assert "More content" in result

    def test_removes_page_numbers(self):
        text = "Some content\n- 42 -\nMore content\nPage 10 of 150\nEnd"
        result = self.parser.clean_text(text)
        assert "- 42 -" not in result
        assert "Page 10 of 150" not in result
        assert "Some content" in result
        assert "More content" in result

    def test_empty_input(self):
        assert self.parser.clean_text("") == ""
        assert self.parser.clean_text("   ") == ""

    def test_xbrl_tags_unwrapped(self):
        html_input = """
        <html><body>
        <p>Revenue was <ix:nonFraction>4200000000</ix:nonFraction> dollars</p>
        </body></html>
        """
        result = self.parser.clean_text(html_input)
        assert "4200000000" in result
        assert "ix:nonFraction" not in result


# --- Tests: _is_html ---


class TestIsHtml:
    """Tests for the HTML detection heuristic."""

    def setup_method(self):
        self.parser = FilingParser()

    def test_detects_html_document(self):
        assert self.parser._is_html("<html><body><p>Hello</p></body></html>") is True

    def test_plain_text_not_html(self):
        assert self.parser._is_html("Just some regular text.") is False

    def test_text_with_single_angle_bracket(self):
        assert self.parser._is_html("Revenue < $100M and profit > $10M") is False

    def test_minimal_html(self):
        assert self.parser._is_html("<div>text</div><p>more</p>") is True


# --- Tests: extract_section ---


class TestExtractSection:
    """Tests for individual section extraction."""

    def setup_method(self):
        self.parser = FilingParser(min_section_words=10)

    def test_extract_mda_from_10k(self):
        clean = self.parser.clean_text(SAMPLE_10K_PLAIN_TEXT)
        result = self.parser.extract_section(clean, SectionType.MDA, FormType.FORM_10K)
        assert result is not None
        assert "Revenue increased 8%" in result
        assert "consolidated financial statements" in result

    def test_extract_risk_factors_from_10k(self):
        clean = self.parser.clean_text(SAMPLE_10K_PLAIN_TEXT)
        result = self.parser.extract_section(clean, SectionType.RISK_FACTORS, FormType.FORM_10K)
        assert result is not None
        assert "significant risks" in result

    def test_extract_business_from_10k(self):
        clean = self.parser.clean_text(SAMPLE_10K_PLAIN_TEXT)
        result = self.parser.extract_section(clean, SectionType.BUSINESS, FormType.FORM_10K)
        assert result is not None
        assert "global leader" in result

    def test_extract_financial_from_10k(self):
        clean = self.parser.clean_text(SAMPLE_10K_PLAIN_TEXT)
        result = self.parser.extract_section(clean, SectionType.FINANCIAL, FormType.FORM_10K)
        assert result is not None
        assert "financial statements" in result.lower()

    def test_section_not_found_returns_none(self):
        text = "This document has no recognizable section headers at all."
        result = self.parser.extract_section(text, SectionType.MDA, FormType.FORM_10K)
        assert result is None

    def test_section_too_short_returns_none(self):
        parser = FilingParser(min_section_words=1000)
        text = "Item 7. Management's Discussion and Analysis\nShort text here.\nItem 8. Financial Statements and Supplementary Data\nMore text."
        result = parser.extract_section(text, SectionType.MDA, FormType.FORM_10K)
        assert result is None

    def test_sections_dont_overlap(self):
        """Extracted sections should not contain other section headers."""
        clean = self.parser.clean_text(SAMPLE_10K_PLAIN_TEXT)
        business = self.parser.extract_section(clean, SectionType.BUSINESS, FormType.FORM_10K)
        assert business is not None
        # Business section should end before Risk Factors
        assert "Risk Factors" not in business

    def test_mda_upper_case_header(self):
        text = (
            "Some preamble text.\n"
            "MANAGEMENT'S DISCUSSION AND ANALYSIS\n"
            + "This is the MD&A section content. " * 20
            + "\nSIGNATURES\nEnd of document."
        )
        result = self.parser.extract_section(text, SectionType.MDA, FormType.FORM_10K)
        assert result is not None
        assert "MD&A section content" in result

    def test_item_7_standalone_header(self):
        text = (
            "Some preamble text.\n"
            "Item 7\n"
            + "The company's quarterly results showed improvement. " * 20
            + "\nItem 8. Financial Statements and Supplementary Data\n"
            "Financial data follows."
        )
        result = self.parser.extract_section(text, SectionType.MDA, FormType.FORM_10K)
        assert result is not None
        assert "quarterly results" in result


# --- Tests: parse (full pipeline) ---


class TestParse:
    """Tests for the full parse pipeline."""

    def setup_method(self):
        self.parser = FilingParser(min_section_words=10)

    def test_parse_10k_html(self):
        result = self.parser.parse(SAMPLE_10K_HTML, FormType.FORM_10K, ACCESSION)
        assert isinstance(result, dict)
        # Should find at least some sections
        assert len(result) > 0
        # All values should be FilingSection instances
        for section in result.values():
            assert section.filing_id == ACCESSION
            assert section.word_count > 0

    def test_parse_10k_extracts_mda(self):
        result = self.parser.parse(SAMPLE_10K_HTML, FormType.FORM_10K, ACCESSION)
        assert "mda" in result
        assert "Revenue increased 8%" in result["mda"].raw_text

    def test_parse_10k_extracts_risk_factors(self):
        result = self.parser.parse(SAMPLE_10K_HTML, FormType.FORM_10K, ACCESSION)
        assert "risk_factors" in result
        assert "significant risks" in result["risk_factors"].raw_text

    def test_parse_10k_extracts_business(self):
        result = self.parser.parse(SAMPLE_10K_HTML, FormType.FORM_10K, ACCESSION)
        assert "business" in result
        assert "global leader" in result["business"].raw_text

    def test_parse_10q_html(self):
        result = self.parser.parse(SAMPLE_10Q_HTML, FormType.FORM_10Q, ACCESSION)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_parse_10q_extracts_mda(self):
        result = self.parser.parse(SAMPLE_10Q_HTML, FormType.FORM_10Q, ACCESSION)
        assert "mda" in result
        assert "Revenue increased 6%" in result["mda"].raw_text

    def test_parse_10q_extracts_risk_factors(self):
        result = self.parser.parse(SAMPLE_10Q_HTML, FormType.FORM_10Q, ACCESSION)
        assert "risk_factors" in result
        assert "Trade tensions" in result["risk_factors"].raw_text

    def test_parse_10q_no_business_section(self):
        """10-Q filings don't have a Business section."""
        result = self.parser.parse(SAMPLE_10Q_HTML, FormType.FORM_10Q, ACCESSION)
        assert "business" not in result

    def test_parse_10k_amendment(self):
        """10-K/A should extract same sections as 10-K."""
        result = self.parser.parse(SAMPLE_10K_HTML, FormType.FORM_10K_A, ACCESSION)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_parse_plain_text(self):
        result = self.parser.parse(SAMPLE_10K_PLAIN_TEXT, FormType.FORM_10K, ACCESSION)
        assert len(result) > 0
        assert "mda" in result

    def test_filing_section_has_correct_metadata(self):
        result = self.parser.parse(SAMPLE_10K_HTML, FormType.FORM_10K, ACCESSION)
        for name, section in result.items():
            assert section.filing_id == ACCESSION
            assert section.section_name == name
            assert section.word_count == len(section.raw_text.split())
            assert section.extracted_at is not None

    def test_parse_empty_html(self):
        result = self.parser.parse("", FormType.FORM_10K, ACCESSION)
        assert result == {}

    def test_parse_no_sections_found(self):
        result = self.parser.parse(
            "<html><body><p>This has no sections.</p></body></html>",
            FormType.FORM_10K,
            ACCESSION,
        )
        assert result == {}

    def test_tables_stripped_from_html(self):
        """Tables should be stripped by default, so financial table data is gone."""
        result = self.parser.parse(SAMPLE_10K_HTML, FormType.FORM_10K, ACCESSION)
        if "financial" in result:
            assert "$4,200M" not in result["financial"].raw_text

    def test_tables_preserved_when_disabled(self):
        parser = FilingParser(strip_tables=False, min_section_words=10)
        result = parser.parse(SAMPLE_10K_HTML, FormType.FORM_10K, ACCESSION)
        if "financial" in result:
            assert "$4,200M" in result["financial"].raw_text


# --- Tests: edge cases ---


class TestEdgeCases:
    """Edge case and robustness tests."""

    def setup_method(self):
        self.parser = FilingParser(min_section_words=10)

    def test_smart_quotes_in_header(self):
        """Parser should handle smart quotes in 'Management\u2019s'."""
        text = (
            "Item 7. Management\u2019s Discussion and Analysis\n"
            + "Revenue and profitability discussion content here. " * 20
            + "\nItem 8. Financial Statements and Supplementary Data\n"
            "Financial data."
        )
        result = self.parser.extract_section(text, SectionType.MDA, FormType.FORM_10K)
        assert result is not None

    def test_dotted_item_numbers(self):
        """Handle 'Item 1A.' with trailing dot."""
        text = (
            "Preamble.\n"
            "Item 1A. Risk Factors\n"
            + "Risk discussion content for the company. " * 20
            + "\nItem 2. Properties\n"
            "Property information."
        )
        result = self.parser.extract_section(text, SectionType.RISK_FACTORS, FormType.FORM_10K)
        assert result is not None
        assert "Risk discussion" in result

    def test_em_dash_separator(self):
        """Handle 'Item 7 \u2014 MD&A' with em dash."""
        text = (
            "Preamble.\n"
            "Item 7 \u2014 Management's Discussion and Analysis\n"
            + "Quarterly results discussion in detail. " * 20
            + "\nItem 8 \u2014 Financial Statements and Supplementary Data\n"
            "Financial data."
        )
        result = self.parser.extract_section(text, SectionType.MDA, FormType.FORM_10K)
        assert result is not None

    def test_class_variables_are_populated(self):
        """Verify class-level pattern dicts have expected keys."""
        assert SectionType.MDA in FilingParser.SECTION_PATTERNS
        assert SectionType.RISK_FACTORS in FilingParser.SECTION_PATTERNS
        assert SectionType.BUSINESS in FilingParser.SECTION_PATTERNS
        assert SectionType.FINANCIAL in FilingParser.SECTION_PATTERNS

    def test_form_sections_mapping(self):
        """Verify form → sections mapping."""
        assert SectionType.BUSINESS in FilingParser.FORM_SECTIONS[FormType.FORM_10K]
        assert SectionType.BUSINESS not in FilingParser.FORM_SECTIONS[FormType.FORM_10Q]
        assert SectionType.MDA in FilingParser.FORM_SECTIONS[FormType.FORM_10Q]

    def test_multiple_parses_same_instance(self):
        """Parser is stateless — same instance can parse multiple filings."""
        r1 = self.parser.parse(SAMPLE_10K_HTML, FormType.FORM_10K, "0001234567-24-000001")
        r2 = self.parser.parse(SAMPLE_10Q_HTML, FormType.FORM_10Q, "0001234567-24-000002")
        # Different filings produce independent results
        assert r1 != r2
        for s in r1.values():
            assert s.filing_id == "0001234567-24-000001"
        for s in r2.values():
            assert s.filing_id == "0001234567-24-000002"

    def test_deeply_nested_html(self):
        """Parser handles deeply nested HTML structures."""
        nested = "<html><body>" + "<div>" * 20 + "<p>Deep content</p>" + "</div>" * 20 + "</body></html>"
        result = self.parser.clean_text(nested)
        assert "Deep content" in result
