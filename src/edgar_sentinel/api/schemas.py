"""API-specific request/response schemas (Pydantic v2)."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


# -- Pagination --


class PaginatedResponse(BaseModel):
    """Wrapper for paginated list responses."""

    total: int
    offset: int
    limit: int


# -- Error --


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: str | None = None


# -- Signals --


class SignalResponse(BaseModel):
    """Single composite signal in API response format."""

    ticker: str
    signal_date: date
    composite_score: float
    rank: int | None = None
    components: dict[str, float]


class SignalListResponse(PaginatedResponse):
    """Paginated list of composite signals."""

    items: list[SignalResponse]


# -- Filings --


class FilingResponse(BaseModel):
    """Filing metadata in API response format."""

    accession_number: str
    ticker: str | None
    company_name: str
    form_type: str
    filed_date: date
    sections: list[str]
    analysis_status: dict[str, bool]


class FilingListResponse(PaginatedResponse):
    """Paginated list of filings."""

    items: list[FilingResponse]


class AnalysisResultResponse(BaseModel):
    """Analysis result in API response format."""

    analyzer_name: str
    section_name: str
    sentiment_score: float | None = None
    similarity_score: float | None = None
    confidence: float | None = None
    analyzed_at: datetime


class FilingDetailResponse(FilingResponse):
    """Filing with full section text and analysis results."""

    section_texts: dict[str, str]
    analysis_results: list[AnalysisResultResponse]


# -- Pipeline Requests --


class IngestRequest(BaseModel):
    """Request body for POST /api/pipeline/ingest."""

    tickers: list[str] = Field(..., min_length=1, max_length=500)
    form_types: list[str] = Field(default=["10-K", "10-Q"])
    start_year: int = Field(..., ge=1993, le=2030)
    end_year: int = Field(..., ge=1993, le=2030)
    force: bool = False


class AnalyzeRequest(BaseModel):
    """Request body for POST /api/pipeline/analyze."""

    analyzers: list[str] = Field(default=["dictionary", "similarity"])
    tickers: list[str] | None = Field(default=None, description="None = all ingested")
    force: bool = False


class BacktestRequest(BaseModel):
    """Request body for POST /api/pipeline/backtest."""

    start_date: date
    end_date: date
    rebalance_frequency: str = "quarterly"
    num_quantiles: int = Field(default=5, ge=2, le=20)
    long_only: bool = False
    transaction_cost_bps: int = Field(default=10, ge=0, le=100)


# -- Jobs --


class JobResponse(BaseModel):
    """Response for async pipeline job submission."""

    job_id: str
    status: str
    created_at: datetime
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status polling."""

    job_id: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None
    result: dict | None = None
    error: str | None = None


# -- Health --


class HealthResponse(BaseModel):
    """Response for GET /api/health."""

    status: str = "ok"
    version: str
    storage_backend: str
    total_filings: int
    total_signals: int
