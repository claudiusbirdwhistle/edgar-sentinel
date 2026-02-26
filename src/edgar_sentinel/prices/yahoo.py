"""Yahoo Finance price provider — direct HTTP implementation.

Uses the unauthenticated ``/v8/finance/chart/`` endpoint via httpx.
No external dependencies beyond httpx (already a core project dep).

The chart endpoint provides OHLCV data without authentication for daily,
weekly, and monthly intervals with full history available.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timezone
from typing import Any

import httpx

from edgar_sentinel.prices.models import (
    PriceBar,
    PriceInterval,
    PriceMeta,
    PriceQuery,
)

logger = logging.getLogger(__name__)

# Yahoo Finance chart API base
_BASE_URL = "https://query2.finance.yahoo.com"
_CHART_PATH = "/v8/finance/chart"
_USER_AGENT = "Mozilla/5.0 (compatible; edgar-sentinel/0.1)"

# Map our intervals to Yahoo Finance interval strings
_INTERVAL_MAP: dict[PriceInterval, str] = {
    PriceInterval.DAILY: "1d",
    PriceInterval.WEEKLY: "1wk",
    PriceInterval.MONTHLY: "1mo",
}


class YahooFinanceAdapter:
    """Transforms raw Yahoo Finance chart JSON into PriceBar records.

    This adapter understands the ``/v8/finance/chart/`` response format
    and converts it to the canonical PriceBar model.
    """

    def adapt(self, raw_data: Any, ticker: str) -> list[PriceBar]:
        """Parse Yahoo Finance chart response into PriceBar list.

        Parameters
        ----------
        raw_data : dict
            The ``chart.result[0]`` object from a Yahoo Finance chart response.
        ticker : str
            The ticker symbol.

        Returns
        -------
        list[PriceBar]
            Sorted by date ascending. Bars with null values are skipped.
        """
        timestamps: list[int] = raw_data.get("timestamp", [])
        if not timestamps:
            return []

        quotes = raw_data.get("indicators", {}).get("quote", [{}])[0]
        adjclose_data = raw_data.get("indicators", {}).get("adjclose", [{}])
        adj_closes: list[float | None] = (
            adjclose_data[0].get("adjclose", []) if adjclose_data else []
        )

        opens: list[float | None] = quotes.get("open", [])
        highs: list[float | None] = quotes.get("high", [])
        lows: list[float | None] = quotes.get("low", [])
        closes: list[float | None] = quotes.get("close", [])
        volumes: list[int | None] = quotes.get("volume", [])

        bars: list[PriceBar] = []
        for i, ts in enumerate(timestamps):
            o = opens[i] if i < len(opens) else None
            h = highs[i] if i < len(highs) else None
            lo = lows[i] if i < len(lows) else None
            c = closes[i] if i < len(closes) else None
            v = volumes[i] if i < len(volumes) else None
            ac = adj_closes[i] if i < len(adj_closes) else None

            # Skip bars with null OHLC values (holidays, missing data)
            if any(x is None for x in (o, h, lo, c)):
                continue

            bar_date = date.fromtimestamp(ts)
            bars.append(
                PriceBar(
                    ticker=ticker,
                    date=bar_date,
                    open=float(o),
                    high=float(h),
                    low=float(lo),
                    close=float(c),
                    volume=int(v) if v is not None else 0,
                    adj_close=float(ac) if ac is not None else None,
                    source="yahoo_finance",
                )
            )

        return sorted(bars, key=lambda b: b.date)


class YahooFinancePriceProvider:
    """Fetches price data from Yahoo Finance's chart API.

    Uses the unauthenticated ``/v8/finance/chart/`` endpoint. Rate-limits
    requests with configurable delay between calls.

    Parameters
    ----------
    request_delay : float
        Minimum seconds between HTTP requests. Default: 0.5.
    timeout : float
        HTTP request timeout in seconds. Default: 15.0.
    base_url : str
        Override base URL (useful for testing).
    adapter : YahooFinanceAdapter | None
        Custom adapter instance. Uses default if None.
    """

    def __init__(
        self,
        request_delay: float = 0.5,
        timeout: float = 15.0,
        base_url: str = _BASE_URL,
        adapter: YahooFinanceAdapter | None = None,
    ) -> None:
        self._delay = request_delay
        self._timeout = timeout
        self._base_url = base_url
        self._adapter = adapter or YahooFinanceAdapter()
        self._last_request_time: float = 0.0

    async def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._delay:
            await asyncio.sleep(self._delay - elapsed)
        self._last_request_time = time.monotonic()

    async def _fetch_chart(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str,
    ) -> dict | None:
        """Fetch raw chart data for a single ticker.

        Returns the ``chart.result[0]`` object, or None on error.
        """
        await self._rate_limit()

        # Convert dates to Unix timestamps
        period1 = int(datetime.combine(start, datetime.min.time()).timestamp())
        period2 = int(datetime.combine(end, datetime.min.time()).timestamp())

        url = f"{self._base_url}{_CHART_PATH}/{ticker}"
        params = {
            "interval": interval,
            "period1": str(period1),
            "period2": str(period2),
            "events": "div,split",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    url,
                    params=params,
                    headers={"User-Agent": _USER_AGENT},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "Yahoo Finance HTTP error for %s: %s %s",
                ticker,
                e.response.status_code,
                e.response.text[:200],
            )
            return None
        except httpx.RequestError as e:
            logger.error("Yahoo Finance request error for %s: %s", ticker, e)
            return None

        # Check for API-level errors
        chart = data.get("chart", {})
        if chart.get("error"):
            err = chart["error"]
            logger.error(
                "Yahoo Finance API error for %s: %s — %s",
                ticker,
                err.get("code"),
                err.get("description"),
            )
            return None

        results = chart.get("result")
        if not results:
            logger.warning("Yahoo Finance returned no results for %s", ticker)
            return None

        return results[0]

    async def get_prices(self, query: PriceQuery) -> dict[str, list[PriceBar]]:
        """Fetch price bars for the requested tickers.

        Makes one HTTP request per ticker with rate limiting.
        Tickers that fail or return no data are omitted from results.
        """
        interval = _INTERVAL_MAP[query.interval]
        result: dict[str, list[PriceBar]] = {}

        for ticker in query.tickers:
            raw = await self._fetch_chart(ticker, query.start, query.end, interval)
            if raw is None:
                continue

            bars = self._adapter.adapt(raw, ticker)
            if bars:
                # Filter to exact date range (API may return extra buffer)
                bars = [b for b in bars if query.start <= b.date <= query.end]
                if bars:
                    result[ticker] = bars

        return result

    async def get_metadata(self, ticker: str) -> PriceMeta | None:
        """Fetch metadata for a ticker from the chart endpoint's meta field."""
        raw = await self._fetch_chart(
            ticker,
            date.today(),
            date.today(),
            "1d",
        )
        if raw is None:
            return None

        meta = raw.get("meta", {})
        return PriceMeta(
            ticker=ticker,
            currency=meta.get("currency"),
            exchange=meta.get("fullExchangeName") or meta.get("exchangeName"),
            instrument_type=meta.get("instrumentType"),
            source="yahoo_finance",
            fetched_at=datetime.now(timezone.utc),
        )
