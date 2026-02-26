"""Price data models for the source-agnostic price ingestion system."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class PriceInterval(StrEnum):
    """Supported price data intervals."""

    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"


class PriceBar(BaseModel):
    """A single OHLCV price bar — the canonical price record.

    All price providers and adapters must produce data in this format.
    This is the system's universal price representation.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float | None = None
    source: str = "unknown"

    @field_validator("volume")
    @classmethod
    def volume_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"volume must be >= 0, got {v}")
        return v

    @model_validator(mode="after")
    def high_gte_low(self) -> PriceBar:
        if self.high < self.low:
            raise ValueError(
                f"high ({self.high}) must be >= low ({self.low})"
            )
        return self

    @property
    def adjusted_close(self) -> float:
        """Return adj_close if available, otherwise close."""
        return self.adj_close if self.adj_close is not None else self.close


class PriceQuery(BaseModel):
    """Parameters for a price data request."""

    model_config = ConfigDict(frozen=True)

    tickers: list[str]
    start: date
    end: date
    interval: PriceInterval = PriceInterval.DAILY

    @field_validator("tickers")
    @classmethod
    def tickers_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("tickers must not be empty")
        return v

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError(f"end ({v}) must be after start ({start})")
        return v


class PriceMeta(BaseModel):
    """Metadata returned alongside price data."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    currency: str | None = None
    exchange: str | None = None
    instrument_type: str | None = None
    source: str = "unknown"
    fetched_at: datetime | None = None
