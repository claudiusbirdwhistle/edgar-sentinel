"""Tests for edgar_sentinel.ingestion.client (EdgarClient)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from edgar_sentinel.core.config import EdgarConfig
from edgar_sentinel.core.exceptions import IngestionError, RateLimitError
from edgar_sentinel.core.models import FilingMetadata, FormType
from edgar_sentinel.ingestion.client import EdgarClient


# --- Fixtures ---


@pytest.fixture
def edgar_config(tmp_path: Path) -> EdgarConfig:
    return EdgarConfig(
        user_agent="TestAgent test@example.com",
        rate_limit=10,
        cache_dir=str(tmp_path / "cache"),
        request_timeout=5,
    )


@pytest.fixture
async def client(edgar_config: EdgarConfig) -> EdgarClient:
    async with EdgarClient(edgar_config) as c:
        yield c


@pytest.fixture
def company_tickers_json() -> dict:
    """Mock SEC company_tickers.json response."""
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
        "2": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
    }


@pytest.fixture
def submissions_json() -> dict:
    """Mock EDGAR submissions response for AAPL."""
    return {
        "cik": "0000320193",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-23-000106",
                    "0000320193-23-000077",
                    "0000320193-22-000108",
                ],
                "filingDate": ["2023-11-03", "2023-08-04", "2022-10-28"],
                "form": ["10-K", "10-Q", "10-K"],
                "primaryDocument": ["aapl-20230930.htm", "aapl-20230701.htm", "aapl-20220924.htm"],
            },
            "files": [],
        },
    }


@pytest.fixture
def index_content() -> str:
    """Mock EDGAR company.idx content."""
    return (
        "Company Name|Form Type|CIK|Date Filed|Filename\n"
        "---\n"
        "Apple Inc.|10-K|320193|2023-11-03|edgar/data/320193/000032019323000106/aapl-20230930.htm\n"
        "Microsoft Corp|10-Q|789019|2023-10-25|edgar/data/789019/000078901923000045/msft-20230930.htm\n"
        "Some Other Co|8-K|999999|2023-10-20|edgar/data/999999/000099999923000001/doc.htm\n"
    )


# --- get_company_tickers ---


class TestGetCompanyTickers:
    @respx.mock
    async def test_fetches_and_parses(self, client: EdgarClient, company_tickers_json: dict):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )

        result = await client.get_company_tickers()
        assert result["AAPL"] == "0000320193"
        assert result["MSFT"] == "0000789019"
        assert result["GOOGL"] == "0001652044"

    @respx.mock
    async def test_caches_in_memory(self, client: EdgarClient, company_tickers_json: dict):
        route = respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )

        await client.get_company_tickers()
        await client.get_company_tickers()  # second call should use cache
        assert route.call_count == 1

    @respx.mock
    async def test_caches_to_disk(
        self, client: EdgarClient, edgar_config: EdgarConfig, company_tickers_json: dict
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )

        await client.get_company_tickers()
        cache_file = Path(edgar_config.cache_dir) / "company_tickers.json"
        assert cache_file.exists()

    @respx.mock
    async def test_falls_back_to_disk_cache(
        self, edgar_config: EdgarConfig, company_tickers_json: dict
    ):
        # Pre-populate disk cache
        cache_dir = Path(edgar_config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "company_tickers.json"
        cache_file.write_text(json.dumps(company_tickers_json))

        # Make network fail
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(500)
        )

        async with EdgarClient(edgar_config) as client:
            result = await client.get_company_tickers()
            assert result["AAPL"] == "0000320193"

    @respx.mock
    async def test_raises_on_failure_no_cache(self, client: EdgarClient):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(404)
        )

        with pytest.raises(IngestionError, match="HTTP 404"):
            await client.get_company_tickers()


# --- resolve_ticker ---


class TestResolveTicker:
    @respx.mock
    async def test_resolves_known_ticker(
        self, client: EdgarClient, company_tickers_json: dict
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )

        cik = await client.resolve_ticker("aapl")
        assert cik == "0000320193"

    @respx.mock
    async def test_case_insensitive(
        self, client: EdgarClient, company_tickers_json: dict
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )

        assert await client.resolve_ticker("Msft") == "0000789019"

    @respx.mock
    async def test_unknown_ticker_raises(
        self, client: EdgarClient, company_tickers_json: dict
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )

        with pytest.raises(IngestionError, match="Ticker not found"):
            await client.resolve_ticker("ZZZZ")


# --- resolve_cik ---


class TestResolveCIK:
    @respx.mock
    async def test_resolves_known_cik(
        self, client: EdgarClient, company_tickers_json: dict
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )

        ticker = await client.resolve_cik("320193")
        assert ticker == "AAPL"

    @respx.mock
    async def test_resolves_zero_padded_cik(
        self, client: EdgarClient, company_tickers_json: dict
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )

        ticker = await client.resolve_cik("0000320193")
        assert ticker == "AAPL"

    @respx.mock
    async def test_unknown_cik_returns_none(
        self, client: EdgarClient, company_tickers_json: dict
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )

        result = await client.resolve_cik("9999999999")
        assert result is None


# --- get_submissions ---


class TestGetSubmissions:
    @respx.mock
    async def test_fetches_submissions(
        self, client: EdgarClient, submissions_json: dict
    ):
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
            return_value=httpx.Response(200, json=submissions_json)
        )

        result = await client.get_submissions("0000320193")
        assert result["name"] == "Apple Inc."
        assert len(result["filings"]["recent"]["accessionNumber"]) == 3

    @respx.mock
    async def test_handles_pagination(self, client: EdgarClient):
        page1 = {
            "cik": "0000320193",
            "name": "Apple Inc.",
            "filings": {
                "recent": {
                    "accessionNumber": ["0000320193-23-000106"],
                    "filingDate": ["2023-11-03"],
                    "form": ["10-K"],
                    "primaryDocument": ["aapl-20230930.htm"],
                },
                "files": [{"name": "CIK0000320193-submissions-001.json"}],
            },
        }
        page2 = {
            "accessionNumber": ["0000320193-22-000108"],
            "filingDate": ["2022-10-28"],
            "form": ["10-K"],
            "primaryDocument": ["aapl-20220924.htm"],
        }

        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
            return_value=httpx.Response(200, json=page1)
        )
        respx.get(
            "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json"
        ).mock(return_value=httpx.Response(200, json=page2))

        result = await client.get_submissions("0000320193")
        accessions = result["filings"]["recent"]["accessionNumber"]
        assert len(accessions) == 2
        assert "0000320193-22-000108" in accessions


# --- get_filings_for_ticker ---


class TestGetFilingsForTicker:
    @respx.mock
    async def test_returns_filtered_filings(
        self,
        client: EdgarClient,
        company_tickers_json: dict,
        submissions_json: dict,
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
            return_value=httpx.Response(200, json=submissions_json)
        )

        filings = await client.get_filings_for_ticker(
            "AAPL", form_types=[FormType.FORM_10K]
        )
        assert len(filings) == 2
        assert all(f.form_type == FormType.FORM_10K for f in filings)

    @respx.mock
    async def test_date_filtering(
        self,
        client: EdgarClient,
        company_tickers_json: dict,
        submissions_json: dict,
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
            return_value=httpx.Response(200, json=submissions_json)
        )

        filings = await client.get_filings_for_ticker(
            "AAPL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert len(filings) == 2  # 10-K and 10-Q from 2023
        assert all(f.filed_date.year == 2023 for f in filings)

    @respx.mock
    async def test_sorted_descending(
        self,
        client: EdgarClient,
        company_tickers_json: dict,
        submissions_json: dict,
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
            return_value=httpx.Response(200, json=submissions_json)
        )

        filings = await client.get_filings_for_ticker("AAPL")
        dates = [f.filed_date for f in filings]
        assert dates == sorted(dates, reverse=True)

    @respx.mock
    async def test_default_form_types(
        self,
        client: EdgarClient,
        company_tickers_json: dict,
        submissions_json: dict,
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )
        respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
            return_value=httpx.Response(200, json=submissions_json)
        )

        filings = await client.get_filings_for_ticker("AAPL")
        assert len(filings) == 3  # All 3 are 10-K or 10-Q


# --- get_filing_document ---


class TestGetFilingDocument:
    @respx.mock
    async def test_downloads_document(self, client: EdgarClient):
        url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
        respx.get(url).mock(
            return_value=httpx.Response(200, text="<html>filing content</html>")
        )

        content = await client.get_filing_document(url)
        assert content == "<html>filing content</html>"

    @respx.mock
    async def test_caches_document(self, client: EdgarClient, edgar_config: EdgarConfig):
        url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
        route = respx.get(url).mock(
            return_value=httpx.Response(200, text="<html>filing</html>")
        )

        await client.get_filing_document(url)
        await client.get_filing_document(url)  # second call uses cache
        assert route.call_count == 1

    @respx.mock
    async def test_returns_cached_content(
        self, edgar_config: EdgarConfig
    ):
        url = "https://www.sec.gov/Archives/edgar/data/320193/test.htm"
        # Pre-populate cache
        cache_path = Path(edgar_config.cache_dir) / "Archives/edgar/data/320193/test.htm"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("cached content")

        async with EdgarClient(edgar_config) as client:
            content = await client.get_filing_document(url)
            assert content == "cached content"


# --- get_filing_url ---


class TestGetFilingURL:
    async def test_returns_metadata_url(self, client: EdgarClient, sample_filing_metadata):
        url = await client.get_filing_url(sample_filing_metadata)
        assert url == sample_filing_metadata.url


# --- bulk_discover ---


class TestBulkDiscover:
    @respx.mock
    async def test_discovers_filings(
        self,
        client: EdgarClient,
        company_tickers_json: dict,
        index_content: str,
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )
        respx.get(
            "https://www.sec.gov/Archives/edgar/full-index/2023/QTR4/company.idx"
        ).mock(return_value=httpx.Response(200, text=index_content))
        # Mock other quarters as 404 (will be skipped)
        for q in range(1, 4):
            respx.get(
                f"https://www.sec.gov/Archives/edgar/full-index/2023/QTR{q}/company.idx"
            ).mock(return_value=httpx.Response(404))

        results = await client.bulk_discover(
            form_types=[FormType.FORM_10K],
            start_year=2023,
            end_year=2023,
        )
        assert len(results) == 1
        assert results[0].form_type == FormType.FORM_10K
        assert results[0].cik == "0000320193"

    @respx.mock
    async def test_filters_by_ticker(
        self,
        client: EdgarClient,
        company_tickers_json: dict,
        index_content: str,
    ):
        respx.get("https://www.sec.gov/files/company_tickers.json").mock(
            return_value=httpx.Response(200, json=company_tickers_json)
        )
        for q in range(1, 5):
            respx.get(
                f"https://www.sec.gov/Archives/edgar/full-index/2023/QTR{q}/company.idx"
            ).mock(return_value=httpx.Response(200, text=index_content))

        results = await client.bulk_discover(
            form_types=[FormType.FORM_10K, FormType.FORM_10Q],
            start_year=2023,
            end_year=2023,
            tickers=["MSFT"],
        )
        assert len(results) == 4  # 1 10-Q per quarter
        assert all(r.cik == "0000789019" for r in results)


# --- _parse_index_line ---


class TestParseIndexLine:
    def test_parses_valid_line(self, client: EdgarClient):
        line = "Apple Inc.|10-K|320193|2023-11-03|edgar/data/320193/000032019323000106/aapl.htm"
        result = client._parse_index_line(line, {"10-K"}, None)
        assert result is not None
        assert result.company_name == "Apple Inc."
        assert result.form_type == FormType.FORM_10K
        assert result.cik == "0000320193"
        assert result.filed_date == date(2023, 11, 3)

    def test_skips_header_lines(self, client: EdgarClient):
        assert client._parse_index_line("Company Name|Form Type|CIK|Date Filed|Filename", {"10-K"}, None) is None
        assert client._parse_index_line("---", {"10-K"}, None) is None

    def test_filters_by_form_type(self, client: EdgarClient):
        line = "Company|8-K|320193|2023-11-03|edgar/data/320193/000032019323000106/doc.htm"
        result = client._parse_index_line(line, {"10-K"}, None)
        assert result is None

    def test_filters_by_cik(self, client: EdgarClient):
        line = "Apple Inc.|10-K|320193|2023-11-03|edgar/data/320193/000032019323000106/aapl.htm"
        result = client._parse_index_line(line, {"10-K"}, {"0000789019"})
        assert result is None

    def test_accepts_matching_cik(self, client: EdgarClient):
        line = "Apple Inc.|10-K|320193|2023-11-03|edgar/data/320193/000032019323000106/aapl.htm"
        result = client._parse_index_line(line, {"10-K"}, {"0000320193"})
        assert result is not None

    def test_handles_bad_date(self, client: EdgarClient):
        line = "Company|10-K|320193|not-a-date|edgar/data/320193/000032019323000106/aapl.htm"
        result = client._parse_index_line(line, {"10-K"}, None)
        assert result is None


# --- Rate Limiting & Retry ---


class TestRateLimitedRequest:
    @respx.mock
    async def test_basic_get(self, client: EdgarClient):
        respx.get("https://example.com/test").mock(
            return_value=httpx.Response(200, text="ok")
        )

        response = await client._rate_limited_request("GET", "https://example.com/test")
        assert response.status_code == 200

    @respx.mock
    async def test_raises_on_404(self, client: EdgarClient):
        respx.get("https://example.com/missing").mock(
            return_value=httpx.Response(404)
        )

        with pytest.raises(IngestionError, match="HTTP 404"):
            await client._rate_limited_request("GET", "https://example.com/missing")

    @respx.mock
    async def test_retries_on_429(self, client: EdgarClient):
        route = respx.get("https://example.com/rate").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "0"}),
                httpx.Response(200, text="ok"),
            ]
        )

        response = await client._rate_limited_request("GET", "https://example.com/rate")
        assert response.status_code == 200
        assert route.call_count == 2

    @respx.mock
    async def test_raises_rate_limit_after_retries(self, client: EdgarClient):
        respx.get("https://example.com/rate").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "0"})
        )

        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await client._rate_limited_request("GET", "https://example.com/rate")

    @respx.mock
    async def test_retries_on_server_error(self, client: EdgarClient):
        route = respx.get("https://example.com/server").mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(200, text="ok"),
            ]
        )

        response = await client._rate_limited_request("GET", "https://example.com/server")
        assert response.status_code == 200
        assert route.call_count == 2


# --- File Cache ---


class TestFileCache:
    def test_cache_path_mapping(self, client: EdgarClient):
        url = "https://www.sec.gov/Archives/edgar/data/320193/file.htm"
        path = client._cache_path(url)
        assert str(path).endswith("Archives/edgar/data/320193/file.htm")

    async def test_get_cached_returns_none_for_missing(self, client: EdgarClient):
        result = await client._get_cached("https://example.com/nonexistent")
        assert result is None

    async def test_write_and_read_cache(self, client: EdgarClient):
        url = "https://example.com/test/file.txt"
        await client._write_cache(url, "cached data")
        result = await client._get_cached(url)
        assert result == "cached data"


# --- Context Manager ---


class TestContextManager:
    async def test_async_context_manager(self, edgar_config: EdgarConfig):
        async with EdgarClient(edgar_config) as client:
            assert client is not None
        # Client should be closed after exiting
        assert client._client.is_closed
