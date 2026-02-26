"""Source-agnostic price ingestion system.

Architecture
------------
Uses the adapter pattern to decouple price data sources from consumers:

    DataSource → PriceAdapter → list[PriceBar] → PriceProvider → Consumer

Key abstractions:

- ``PriceBar``: Canonical OHLCV price record (the universal schema).
- ``PriceAdapter``: Transforms raw source data into ``PriceBar`` records.
- ``PriceProvider``: Consumer-facing async interface for fetching prices.
- ``PriceStore``: Persistence protocol for caching fetched prices.

Built-in implementations:

- ``YahooFinancePriceProvider``: Fetches from Yahoo Finance chart API.
- ``YahooFinanceAdapter``: Parses Yahoo Finance JSON into PriceBars.
- ``CSVPriceAdapter``: Parses CSV files into PriceBars.
- ``SqlitePriceStore``: SQLite-backed persistent price cache.

Adding a new price source:
1. Write an adapter that implements ``PriceAdapter.adapt(raw_data, ticker)``.
2. Optionally write a provider that implements ``PriceProvider``.
3. That's it — no changes to consumers needed.
"""

from edgar_sentinel.prices.csv_adapter import CSVPriceAdapter, load_csv_prices
from edgar_sentinel.prices.models import PriceBar, PriceInterval, PriceMeta, PriceQuery
from edgar_sentinel.prices.provider import PriceAdapter, PriceProvider
from edgar_sentinel.prices.store import PriceStore, SqlitePriceStore
from edgar_sentinel.prices.yahoo import YahooFinanceAdapter, YahooFinancePriceProvider

__all__ = [
    # Models
    "PriceBar",
    "PriceInterval",
    "PriceMeta",
    "PriceQuery",
    # Protocols
    "PriceAdapter",
    "PriceProvider",
    "PriceStore",
    # Yahoo Finance
    "YahooFinanceAdapter",
    "YahooFinancePriceProvider",
    # CSV
    "CSVPriceAdapter",
    "load_csv_prices",
    # SQLite
    "SqlitePriceStore",
]
