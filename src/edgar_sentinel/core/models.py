"""Pydantic data models — the system's type contracts."""

from __future__ import annotations

import re
import warnings
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# --- Type Aliases ---

AccessionNumber = str
CIK = str
Ticker = str
SectionName = str
AnalyzerName = str
SignalName = str

# --- Enumerations ---


class FormType(StrEnum):
    """SEC filing form types supported by edgar-sentinel."""

    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_10K_A = "10-K/A"
    FORM_10Q_A = "10-Q/A"


class SectionType(StrEnum):
    """Standard section identifiers for extracted filing sections."""

    MDA = "mda"
    RISK_FACTORS = "risk_factors"
    BUSINESS = "business"
    FINANCIAL = "financial"


class AnalyzerType(StrEnum):
    """Built-in analyzer identifiers."""

    DICTIONARY = "dictionary"
    SIMILARITY = "similarity"
    LLM = "llm"


class RebalanceFrequency(StrEnum):
    """Backtest rebalance periods."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class StorageBackend(StrEnum):
    """Supported storage backends."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class LLMProvider(StrEnum):
    """Supported LLM providers for the LLM analyzer."""

    CLAUDE_CLI = "claude_cli"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class CompositeMethod(StrEnum):
    """Signal combination strategies."""

    EQUAL = "equal"
    IC_WEIGHTED = "ic_weighted"


# --- Filing Models ---


class FilingMetadata(BaseModel):
    """Metadata for a single SEC filing."""

    model_config = ConfigDict(frozen=True)

    cik: CIK
    ticker: Ticker | None = None
    company_name: str
    form_type: FormType
    filed_date: date
    accession_number: AccessionNumber
    fiscal_year_end: date | None = None
    url: str

    @field_validator("cik")
    @classmethod
    def cik_must_be_numeric(cls, v: str) -> str:
        if not v.strip().isdigit():
            raise ValueError(f"CIK must be numeric, got: {v!r}")
        return v.strip().zfill(10)

    @field_validator("accession_number")
    @classmethod
    def accession_format(cls, v: str) -> str:
        """Validate EDGAR accession number format: XXXXXXXXXX-YY-ZZZZZZ."""
        normalized = v.replace("-", "")
        if not re.match(r"^\d{18}$", normalized):
            raise ValueError(f"Invalid accession number format: {v!r}")
        return f"{normalized[:10]}-{normalized[10:12]}-{normalized[12:]}"

    @field_validator("url")
    @classmethod
    def url_must_be_https(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("URL must use HTTPS")
        return v

    @property
    def filing_id(self) -> AccessionNumber:
        """Alias for accession_number."""
        return self.accession_number


class FilingSection(BaseModel):
    """A single extracted section from a filing."""

    model_config = ConfigDict(frozen=True)

    filing_id: AccessionNumber
    section_name: SectionName
    raw_text: str
    word_count: int
    extracted_at: datetime

    @field_validator("word_count")
    @classmethod
    def word_count_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("word_count cannot be negative")
        return v

    @model_validator(mode="after")
    def word_count_matches_text(self) -> FilingSection:
        """Warn if word_count diverges significantly from actual text."""
        actual = len(self.raw_text.split())
        if actual > 0 and abs(actual - self.word_count) / actual > 0.1:
            warnings.warn(
                f"word_count ({self.word_count}) differs from actual "
                f"({actual}) by >10% for {self.filing_id}/{self.section_name}",
                stacklevel=2,
            )
        return self


class Filing(BaseModel):
    """A complete filing: metadata + extracted sections."""

    model_config = ConfigDict(frozen=True)

    metadata: FilingMetadata
    sections: dict[SectionName, FilingSection]

    @property
    def filing_id(self) -> AccessionNumber:
        return self.metadata.accession_number

    @property
    def ticker(self) -> Ticker | None:
        return self.metadata.ticker

    def get_section(self, name: SectionName) -> FilingSection | None:
        """Return section by name, or None if not extracted."""
        return self.sections.get(name)

    def section_names(self) -> list[SectionName]:
        """Return list of available section names."""
        return list(self.sections.keys())


# --- Analysis Result Models ---


class SentimentResult(BaseModel):
    """Output of any analyzer for a single filing section."""

    model_config = ConfigDict(frozen=True)

    filing_id: AccessionNumber
    section_name: SectionName
    analyzer_name: AnalyzerName
    sentiment_score: float
    confidence: float
    metadata: dict[str, Any] = {}
    analyzed_at: datetime

    @field_validator("sentiment_score")
    @classmethod
    def score_in_range(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"sentiment_score must be in [-1.0, 1.0], got {v}")
        return round(v, 6)

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {v}")
        return round(v, 6)


class SimilarityResult(BaseModel):
    """Output of the cosine similarity analyzer (filing-over-filing comparison)."""

    model_config = ConfigDict(frozen=True)

    filing_id: AccessionNumber
    prior_filing_id: AccessionNumber
    section_name: SectionName
    similarity_score: float
    change_score: float
    analyzed_at: datetime

    @field_validator("similarity_score")
    @classmethod
    def similarity_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"similarity_score must be in [0.0, 1.0], got {v}")
        return round(v, 6)

    @model_validator(mode="after")
    def change_is_complement(self) -> SimilarityResult:
        expected = round(1.0 - self.similarity_score, 6)
        if abs(self.change_score - expected) > 1e-5:
            raise ValueError(
                f"change_score ({self.change_score}) must equal "
                f"1 - similarity_score ({expected})"
            )
        return self


# --- Signal Models ---


class Signal(BaseModel):
    """A single trading signal for one ticker at one point in time."""

    model_config = ConfigDict(frozen=True)

    ticker: Ticker
    signal_date: date
    signal_name: SignalName
    raw_value: float
    z_score: float | None = None
    percentile: float | None = None
    decay_weight: float = 1.0

    @field_validator("percentile")
    @classmethod
    def percentile_in_range(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 100.0:
            raise ValueError(f"percentile must be in [0.0, 100.0], got {v}")
        return v

    @field_validator("decay_weight")
    @classmethod
    def decay_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"decay_weight must be in [0.0, 1.0], got {v}")
        return round(v, 6)


class CompositeSignal(BaseModel):
    """Combined signal from multiple analyzers for one ticker at one date."""

    model_config = ConfigDict(frozen=True)

    ticker: Ticker
    signal_date: date
    composite_score: float
    components: dict[SignalName, float]
    rank: int | None = None

    @field_validator("rank")
    @classmethod
    def rank_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"rank must be >= 1, got {v}")
        return v


# --- Backtest Models ---


class BacktestConfig(BaseModel):
    """Configuration for a backtest run."""

    model_config = ConfigDict(frozen=True)

    start_date: date
    end_date: date
    universe: list[Ticker]
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.QUARTERLY
    num_quantiles: int = 5
    signal_buffer_days: int = 2
    long_quantile: int = 1
    short_quantile: int | None = None
    transaction_cost_bps: int = 10

    @field_validator("universe")
    @classmethod
    def universe_not_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("universe must contain at least one ticker")
        return v

    @model_validator(mode="after")
    def dates_ordered(self) -> BacktestConfig:
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self

    @model_validator(mode="after")
    def quantiles_valid(self) -> BacktestConfig:
        if self.long_quantile < 1 or self.long_quantile > self.num_quantiles:
            raise ValueError(
                f"long_quantile ({self.long_quantile}) must be in "
                f"[1, {self.num_quantiles}]"
            )
        if self.short_quantile is not None:
            if self.short_quantile < 1 or self.short_quantile > self.num_quantiles:
                raise ValueError(
                    f"short_quantile ({self.short_quantile}) must be in "
                    f"[1, {self.num_quantiles}]"
                )
        return self


class MonthlyReturn(BaseModel):
    """A single month's return data from a backtest."""

    model_config = ConfigDict(frozen=True)

    period_end: date
    long_return: float
    short_return: float | None = None
    long_short_return: float | None = None
    benchmark_return: float | None = None


class BacktestResult(BaseModel):
    """Complete results from a backtest run."""

    model_config = ConfigDict(frozen=True)

    config: BacktestConfig
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    information_ratio: float | None = None
    monthly_returns: list[MonthlyReturn]
    factor_exposures: dict[str, float] | None = None
    turnover: float
