"""Rate-limited async HTTP client for SEC EDGAR APIs."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx
from aiolimiter import AsyncLimiter

from edgar_sentinel.core.config import EdgarConfig
from edgar_sentinel.core.exceptions import IngestionError, RateLimitError
from edgar_sentinel.core.models import FilingMetadata, FormType

logger = logging.getLogger(__name__)

# EDGAR API base URLs
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_FULL_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/company.idx"
)
_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# Retry configuration
_MAX_RETRIES_429 = 3
_DEFAULT_RETRY_AFTER = 12
_MAX_RETRIES_SERVER = 3
_MAX_RETRIES_CONNECTION = 2
_CONNECTION_RETRY_DELAY = 2.0

# Map EDGAR form type strings to FormType enum
_FORM_TYPE_MAP: dict[str, FormType] = {ft.value: ft for ft in FormType}


class EdgarClient:
    """Rate-limited async client for SEC EDGAR APIs.

    SEC EDGAR fair-access policy:
    - Max 10 requests/second
    - User-Agent MUST include company/person name + email
    - No scraping outside business hours discouraged (but not enforced)

    All methods are async. Use via `async with EdgarClient(...) as client:`.
    """

    def __init__(self, config: EdgarConfig) -> None:
        self._config = config
        self._limiter = AsyncLimiter(max_rate=config.rate_limit, time_period=1.0)
        self._client = httpx.AsyncClient(
            headers={"User-Agent": config.user_agent},
            timeout=httpx.Timeout(config.request_timeout),
            follow_redirects=True,
        )
        self._cache_dir = Path(config.cache_dir)
        self._tickers_cache: dict[str, str] | None = None
        self._cik_to_ticker: dict[str, str] | None = None

    async def __aenter__(self) -> EdgarClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client. Called automatically by __aexit__."""
        await self._client.aclose()

    # --- CIK / Ticker Resolution ---

    async def get_company_tickers(self) -> dict[str, str]:
        """Fetch SEC company_tickers.json and return {ticker: CIK} mapping.

        Returns:
            dict mapping uppercase ticker symbols to zero-padded 10-digit CIK strings.

        Raises:
            IngestionError: If the endpoint returns non-200 or unparseable JSON.
        """
        if self._tickers_cache is not None:
            return self._tickers_cache

        cache_file = self._cache_dir / "company_tickers.json"

        try:
            response = await self._rate_limited_request("GET", _TICKERS_URL)
            raw = response.json()

            # SEC format: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
            result: dict[str, str] = {}
            reverse: dict[str, str] = {}
            for entry in raw.values():
                ticker = str(entry["ticker"]).upper()
                cik = str(entry["cik_str"]).zfill(10)
                result[ticker] = cik
                reverse[cik] = ticker

            self._tickers_cache = result
            self._cik_to_ticker = reverse

            # Write to cache file
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(response.text, encoding="utf-8")

            return result

        except (IngestionError, RateLimitError):
            # Try file cache fallback
            if cache_file.exists():
                logger.warning("EDGAR tickers endpoint failed, using cached file")
                return self._load_tickers_from_cache(cache_file)
            raise

    def _load_tickers_from_cache(self, cache_file: Path) -> dict[str, str]:
        """Load tickers from local cache file."""
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        result: dict[str, str] = {}
        reverse: dict[str, str] = {}
        for entry in raw.values():
            ticker = str(entry["ticker"]).upper()
            cik = str(entry["cik_str"]).zfill(10)
            result[ticker] = cik
            reverse[cik] = ticker
        self._tickers_cache = result
        self._cik_to_ticker = reverse
        return result

    async def resolve_ticker(self, ticker: str) -> str:
        """Resolve a stock ticker to its SEC CIK.

        Args:
            ticker: Stock symbol (case-insensitive).

        Returns:
            10-digit zero-padded CIK string.

        Raises:
            IngestionError: If ticker is not found in SEC database.
        """
        tickers = await self.get_company_tickers()
        upper = ticker.upper()
        if upper not in tickers:
            raise IngestionError(
                f"Ticker not found: {ticker!r}",
                context={"ticker": ticker},
            )
        return tickers[upper]

    async def resolve_cik(self, cik: str) -> str | None:
        """Resolve a CIK to its primary ticker, if available.

        Args:
            cik: SEC Central Index Key (any format — will be normalized).

        Returns:
            Primary ticker symbol (uppercase), or None if no ticker is mapped.
        """
        await self.get_company_tickers()
        normalized = cik.strip().zfill(10)
        if self._cik_to_ticker is None:
            return None
        return self._cik_to_ticker.get(normalized)

    # --- Submission Metadata ---

    async def get_submissions(self, cik: str) -> dict:
        """Fetch all submission metadata for a company.

        Handles pagination — EDGAR splits large submission histories across
        multiple JSON files.

        Args:
            cik: 10-digit zero-padded CIK.

        Returns:
            Complete submissions JSON as a dict.

        Raises:
            IngestionError: Network or parse error.
            RateLimitError: If SEC returns 429 after retry exhaustion.
        """
        url = _SUBMISSIONS_URL.format(cik=cik)
        response = await self._rate_limited_request("GET", url)
        data = response.json()

        # Handle pagination: merge additional filing files
        files = data.get("filings", {}).get("files", [])
        if files:
            recent = data["filings"]["recent"]
            for file_ref in files:
                page_url = f"https://data.sec.gov/submissions/{file_ref['name']}"
                page_response = await self._rate_limited_request("GET", page_url)
                page_data = page_response.json()
                # Merge parallel arrays
                for key in recent:
                    if key in page_data:
                        recent[key].extend(page_data[key])

        return data

    async def get_filings_for_ticker(
        self,
        ticker: str,
        form_types: list[FormType] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[FilingMetadata]:
        """Resolve ticker -> fetch submissions -> filter -> return metadata.

        Args:
            ticker: Stock symbol (e.g., "AAPL").
            form_types: Filter by form type. Default: [10-K, 10-Q, 10-K/A, 10-Q/A].
            start_date: Earliest filing date (inclusive).
            end_date: Latest filing date (inclusive).

        Returns:
            List of FilingMetadata, sorted by filed_date descending.

        Raises:
            IngestionError: If ticker resolution or submission fetch fails.
        """
        if form_types is None:
            form_types = list(FormType)

        cik = await self.resolve_ticker(ticker)
        submissions = await self.get_submissions(cik)

        form_type_values = {ft.value for ft in form_types}
        recent = submissions.get("filings", {}).get("recent", {})
        company_name = submissions.get("name", "")

        results: list[FilingMetadata] = []
        accessions = recent.get("accessionNumber", [])
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        for i in range(len(accessions)):
            form = forms[i] if i < len(forms) else ""
            if form not in form_type_values:
                continue

            filed = date.fromisoformat(dates[i]) if i < len(dates) else None
            if filed is None:
                continue
            if start_date and filed < start_date:
                continue
            if end_date and filed > end_date:
                continue

            accession = accessions[i]
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""
            accession_no_dashes = accession.replace("-", "")
            doc_url = f"{_ARCHIVES_BASE}/{cik}/{accession_no_dashes}/{primary_doc}"

            results.append(
                FilingMetadata(
                    cik=cik,
                    ticker=ticker.upper(),
                    company_name=company_name,
                    form_type=FormType(form),
                    filed_date=filed,
                    accession_number=accession,
                    url=doc_url,
                )
            )

        results.sort(key=lambda m: m.filed_date, reverse=True)
        return results

    # --- Filing Document Retrieval ---

    async def get_filing_document(self, url: str) -> str:
        """Download a single filing document.

        Args:
            url: Full URL to the filing document on EDGAR.

        Returns:
            Raw document content as string (usually HTML).

        Raises:
            IngestionError: Network error or non-200 status.
            RateLimitError: HTTP 429 after retry exhaustion.
        """
        cached = await self._get_cached(url)
        if cached is not None:
            return cached

        response = await self._rate_limited_request("GET", url)
        content = response.text
        await self._write_cache(url, content)
        return content

    async def get_filing_url(self, metadata: FilingMetadata) -> str:
        """Construct the full document URL for a filing.

        Args:
            metadata: Filing metadata (must have cik, accession_number, url fields).

        Returns:
            Full HTTPS URL string.
        """
        return metadata.url

    # --- Bulk Discovery ---

    async def get_full_index(self, year: int, quarter: int) -> str:
        """Fetch EDGAR full-index file for a specific quarter.

        Args:
            year: Calendar year (e.g., 2024).
            quarter: Quarter number (1-4).

        Returns:
            Raw index file content as string.

        Raises:
            IngestionError: Network error or non-200 response.
        """
        url = _FULL_INDEX_URL.format(year=year, quarter=quarter)
        cached = await self._get_cached(url)
        if cached is not None:
            return cached

        response = await self._rate_limited_request("GET", url)
        content = response.text
        await self._write_cache(url, content)
        return content

    async def bulk_discover(
        self,
        form_types: list[FormType],
        start_year: int,
        end_year: int,
        tickers: list[str] | None = None,
    ) -> list[FilingMetadata]:
        """Discover filings in bulk via EDGAR full-index files.

        Args:
            form_types: Form types to include.
            start_year: First year to scan (inclusive).
            end_year: Last year to scan (inclusive).
            tickers: If provided, filter to these tickers only.

        Returns:
            List of FilingMetadata for all matching filings.

        Raises:
            IngestionError: If index download or parsing fails for any quarter.
        """
        # Resolve tickers to CIKs if filtering
        cik_filter: set[str] | None = None
        if tickers:
            cik_filter = set()
            for t in tickers:
                cik = await self.resolve_ticker(t)
                cik_filter.add(cik)

        form_type_values = {ft.value for ft in form_types}
        results: list[FilingMetadata] = []

        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                try:
                    index_content = await self.get_full_index(year, quarter)
                except IngestionError:
                    logger.warning("Failed to fetch index for %d/Q%d, skipping", year, quarter)
                    continue

                for line in index_content.splitlines():
                    meta = self._parse_index_line(line, form_type_values, cik_filter)
                    if meta is not None:
                        results.append(meta)

        return results

    def _parse_index_line(
        self,
        line: str,
        form_type_values: set[str],
        cik_filter: set[str] | None = None,
    ) -> FilingMetadata | None:
        """Parse a single line from EDGAR company.idx file.

        Expected format (pipe-delimited):
            Company Name|Form Type|CIK|Date Filed|Filename

        Returns:
            FilingMetadata if parsing succeeds, None if the line is a header or
            contains an unsupported form type.
        """
        parts = line.split("|")
        if len(parts) != 5:
            return None

        company_name, form_type, cik_str, date_str, filename = parts
        form_type = form_type.strip()

        if form_type not in form_type_values:
            return None

        cik = cik_str.strip().zfill(10)
        if cik_filter is not None and cik not in cik_filter:
            return None

        try:
            filed = date.fromisoformat(date_str.strip())
        except ValueError:
            return None

        # Extract accession number from filename path
        # Format: edgar/data/{cik}/{accession-no-dashes}/{filename}
        filename = filename.strip()
        filename_parts = filename.split("/")
        if len(filename_parts) < 4:
            return None

        accession_raw = filename_parts[-2]
        # Convert 18-digit number to dashed format
        if len(accession_raw) == 18 and accession_raw.isdigit():
            accession = f"{accession_raw[:10]}-{accession_raw[10:12]}-{accession_raw[12:]}"
        elif "-" in accession_raw:
            accession = accession_raw
        else:
            return None

        doc_name = filename_parts[-1]
        url = f"https://www.sec.gov/Archives/{filename}"

        # Look up ticker from reverse map
        ticker = None
        if self._cik_to_ticker:
            ticker = self._cik_to_ticker.get(cik)

        try:
            return FilingMetadata(
                cik=cik,
                ticker=ticker,
                company_name=company_name.strip(),
                form_type=FormType(form_type),
                filed_date=filed,
                accession_number=accession,
                url=url,
            )
        except (ValueError, KeyError):
            return None

    # --- Rate Limiting & Retry ---

    async def _rate_limited_request(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> httpx.Response:
        """Execute an HTTP request with rate limiting and retry logic.

        Rate limiting:
            Uses aiolimiter.AsyncLimiter as a token bucket. Each request
            acquires one token before sending.

        Retry policy:
            - HTTP 429: Wait for Retry-After header value (or 12s default),
              then retry up to 3 times.
            - HTTP 500/502/503: Retry up to 3 times with exponential backoff.
            - Other HTTP errors: Raise immediately (no retry).
            - Connection errors: Retry up to 2 times with 2s delay.

        Returns:
            httpx.Response with status 200.

        Raises:
            RateLimitError: If retries exhausted on 429 responses.
            IngestionError: If retries exhausted on server errors.
        """
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES_429 + 1):
            try:
                await self._limiter.acquire()
                response = await self._client.request(method, url, **kwargs)

                if response.status_code == 200:
                    return response

                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("Retry-After", _DEFAULT_RETRY_AFTER)
                    )
                    if attempt < _MAX_RETRIES_429:
                        logger.warning(
                            "Rate limited (429) on %s, waiting %ds (attempt %d/%d)",
                            url, retry_after, attempt + 1, _MAX_RETRIES_429,
                        )
                        await asyncio.sleep(retry_after)
                        continue
                    raise RateLimitError(
                        f"Rate limit exceeded after {_MAX_RETRIES_429} retries: {url}",
                        context={"url": url, "retry_after": retry_after},
                    )

                if response.status_code in (500, 502, 503):
                    if attempt < _MAX_RETRIES_SERVER:
                        delay = 2**attempt
                        logger.warning(
                            "Server error %d on %s, retrying in %ds (attempt %d/%d)",
                            response.status_code, url, delay,
                            attempt + 1, _MAX_RETRIES_SERVER,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise IngestionError(
                        f"Server error {response.status_code} after retries: {url}",
                        context={"url": url, "status_code": response.status_code},
                    )

                # Non-retryable HTTP error
                raise IngestionError(
                    f"HTTP {response.status_code} from {url}",
                    context={"url": url, "status_code": response.status_code},
                )

            except httpx.ConnectError as e:
                last_exc = e
                if attempt < _MAX_RETRIES_CONNECTION:
                    logger.warning(
                        "Connection error on %s, retrying in %ds (attempt %d/%d)",
                        url, _CONNECTION_RETRY_DELAY,
                        attempt + 1, _MAX_RETRIES_CONNECTION,
                    )
                    await asyncio.sleep(_CONNECTION_RETRY_DELAY)
                    continue
                raise IngestionError(
                    f"Connection failed after retries: {url}",
                    context={"url": url, "error": str(e)},
                ) from e

        # Should not reach here, but just in case
        raise IngestionError(
            f"Request failed after all retries: {url}",
            context={"url": url},
        )

    # --- File Cache ---

    def _cache_path(self, url: str) -> Path:
        """Determine local cache path for a URL.

        The URL path after the domain becomes the filesystem path under cache_dir.
        """
        parsed = urlparse(url)
        # Strip leading slash from path
        rel_path = parsed.path.lstrip("/")
        return self._cache_dir / rel_path

    async def _get_cached(self, url: str) -> str | None:
        """Return cached content for a URL, or None if not cached."""
        path = self._cache_path(url)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    async def _write_cache(self, url: str, content: str) -> None:
        """Write content to the file cache. Creates parent directories as needed."""
        path = self._cache_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
