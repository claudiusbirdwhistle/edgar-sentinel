"""Price provider and adapter protocols — the source-agnostic interface layer.

Architecture
------------
The price system uses an adapter pattern to decouple data sources from
consumers:

    RawSource → PriceAdapter → list[PriceBar] → PriceProvider → Consumer

- **PriceProvider** is the consumer-facing protocol. Any code that needs
  prices depends only on this interface.

- **PriceAdapter** transforms raw data (JSON, CSV rows, DataFrame) from
  any source into the canonical ``PriceBar`` format. To add a new price
  source, implement an adapter — no changes to consumers needed.

This design means:
1. The price source and the price consumer are fully decoupled.
2. Adding a new source = writing one adapter class.
3. Raw data in any format can be ingested if an adapter exists.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from edgar_sentinel.prices.models import PriceBar, PriceMeta, PriceQuery


@runtime_checkable
class PriceAdapter(Protocol):
    """Transforms raw data from any source into PriceBar records.

    Each data source (Yahoo Finance, CSV, Polygon, etc.) provides a
    corresponding adapter that knows how to parse its raw format.

    Parameters
    ----------
    raw_data : Any
        The raw response from the data source (JSON dict, CSV rows,
        DataFrame, etc.). The adapter knows the expected shape.
    ticker : str
        The ticker symbol the data belongs to.

    Returns
    -------
    list[PriceBar]
        Canonical price records sorted by date ascending.
    """

    def adapt(self, raw_data: Any, ticker: str) -> list[PriceBar]: ...


@runtime_checkable
class PriceProvider(Protocol):
    """Consumer-facing interface for fetching price data.

    All code that needs prices should depend on this protocol, never
    on a concrete implementation. This ensures source-agnosticism.
    """

    async def get_prices(self, query: PriceQuery) -> dict[str, list[PriceBar]]:
        """Fetch price bars for the requested tickers and date range.

        Returns
        -------
        dict[str, list[PriceBar]]
            Mapping of ticker → sorted list of PriceBar records.
            Tickers with no data are omitted (not empty lists).
        """
        ...

    async def get_metadata(self, ticker: str) -> PriceMeta | None:
        """Fetch metadata for a ticker, if available.

        Returns None if the provider doesn't support metadata or the
        ticker is unknown.
        """
        ...
