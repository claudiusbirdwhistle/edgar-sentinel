"""FastAPI route definitions for the EDGAR Sentinel API."""

from __future__ import annotations

from datetime import UTC as _UTC, date, datetime
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

import edgar_sentinel
from edgar_sentinel.api.deps import AppState, JobStatus, get_app_state, get_config, get_store
from edgar_sentinel.api.schemas import (
    AnalysisResultResponse,
    AnalyzeRequest,
    BacktestRequest,
    FilingDetailResponse,
    FilingListResponse,
    FilingResponse,
    HealthResponse,
    IngestRequest,
    JobResponse,
    JobStatusResponse,
    SignalListResponse,
    SignalResponse,
)
from edgar_sentinel.ingestion.store import SqliteStore

router = APIRouter()


# -- Health --


@router.get("/health", response_model=HealthResponse)
async def health_check(
    store: SqliteStore = Depends(get_store),
    config=Depends(get_config),
):
    """System health and basic statistics."""
    stats = await store.get_statistics()
    return HealthResponse(
        status="ok",
        version=edgar_sentinel.__version__,
        storage_backend=str(config.storage.backend.value),
        total_filings=stats["total_filings"],
        total_signals=stats["composite_signals"],
    )


# -- Signals --


@router.get("/signals", response_model=SignalListResponse)
async def list_signals(
    ticker: str | None = Query(None, description="Filter by ticker"),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    min_score: float | None = Query(None, ge=-1.0, le=1.0),
    max_score: float | None = Query(None, ge=-1.0, le=1.0),
    sort_by: str = Query(
        "composite_score", pattern="^(composite_score|signal_date|ticker)$"
    ),
    sort_desc: bool = Query(True),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    store: SqliteStore = Depends(get_store),
):
    """List composite signals with filtering and pagination."""
    signals = await store.get_composites(
        ticker=ticker.upper() if ticker else None,
        start_date=start_date,
        end_date=end_date,
    )

    # Apply score filters in Python (not in SQL)
    if min_score is not None:
        signals = [s for s in signals if s.composite_score >= min_score]
    if max_score is not None:
        signals = [s for s in signals if s.composite_score <= max_score]

    # Sort
    if sort_by == "composite_score":
        signals.sort(key=lambda s: s.composite_score, reverse=sort_desc)
    elif sort_by == "signal_date":
        signals.sort(key=lambda s: s.signal_date, reverse=sort_desc)
    elif sort_by == "ticker":
        signals.sort(key=lambda s: s.ticker, reverse=sort_desc)

    total = len(signals)
    page = signals[offset : offset + limit]

    return SignalListResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=[
            SignalResponse(
                ticker=s.ticker,
                signal_date=s.signal_date,
                composite_score=s.composite_score,
                rank=s.rank,
                components=s.components,
            )
            for s in page
        ],
    )


@router.get("/signals/{ticker}", response_model=list[SignalResponse])
async def get_ticker_signals(
    ticker: str,
    limit: int = Query(10, ge=1, le=100),
    store: SqliteStore = Depends(get_store),
):
    """Get the latest signals for a specific ticker."""
    signals = await store.get_composites(ticker=ticker.upper())

    if not signals:
        raise HTTPException(
            status_code=404,
            detail=f"No signals found for ticker '{ticker.upper()}'",
        )

    signals.sort(key=lambda s: s.signal_date, reverse=True)
    return [
        SignalResponse(
            ticker=s.ticker,
            signal_date=s.signal_date,
            composite_score=s.composite_score,
            rank=s.rank,
            components=s.components,
        )
        for s in signals[:limit]
    ]


# -- Filings --


@router.get("/filings", response_model=FilingListResponse)
async def list_filings(
    ticker: str | None = Query(None),
    form_type: str | None = Query(None),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    store: SqliteStore = Depends(get_store),
):
    """List filing metadata with filtering and pagination."""
    from edgar_sentinel.core.models import FormType

    ft = FormType(form_type) if form_type else None
    filings = await store.list_filings(
        ticker=ticker.upper() if ticker else None,
        form_type=ft,
        start_date=start_date,
        end_date=end_date,
    )

    total = len(filings)
    page = filings[offset : offset + limit]

    items = []
    for meta in page:
        section_names = await store.get_section_names(meta.accession_number)
        analysis_status = await _get_analysis_status(store, meta.accession_number)
        items.append(
            FilingResponse(
                accession_number=meta.accession_number,
                ticker=meta.ticker,
                company_name=meta.company_name,
                form_type=str(meta.form_type),
                filed_date=meta.filed_date,
                sections=section_names,
                analysis_status=analysis_status,
            )
        )

    return FilingListResponse(total=total, offset=offset, limit=limit, items=items)


@router.get("/filings/{ticker}", response_model=list[FilingResponse])
async def get_ticker_filings(
    ticker: str,
    limit: int = Query(20, ge=1, le=100),
    store: SqliteStore = Depends(get_store),
):
    """Get filing history for a specific ticker."""
    filings = await store.list_filings(ticker=ticker.upper(), limit=limit)

    if not filings:
        raise HTTPException(
            status_code=404,
            detail=f"No filings found for ticker '{ticker.upper()}'",
        )

    items = []
    for meta in filings:
        section_names = await store.get_section_names(meta.accession_number)
        analysis_status = await _get_analysis_status(store, meta.accession_number)
        items.append(
            FilingResponse(
                accession_number=meta.accession_number,
                ticker=meta.ticker,
                company_name=meta.company_name,
                form_type=str(meta.form_type),
                filed_date=meta.filed_date,
                sections=section_names,
                analysis_status=analysis_status,
            )
        )

    return items


@router.get("/filings/detail/{accession_number}", response_model=FilingDetailResponse)
async def get_filing_detail(
    accession_number: str,
    store: SqliteStore = Depends(get_store),
):
    """Get full filing detail including section text and analysis results."""
    filing = await store.get_filing(accession_number)
    if not filing:
        raise HTTPException(status_code=404, detail="Filing not found")

    sentiments = await store.get_sentiments(filing_id=accession_number)
    similarities = await store.get_similarity(filing_id=accession_number)
    analysis_status = await _get_analysis_status(store, accession_number)

    analysis_results = []
    for s in sentiments:
        analysis_results.append(
            AnalysisResultResponse(
                analyzer_name=s.analyzer_name,
                section_name=s.section_name,
                sentiment_score=s.sentiment_score,
                confidence=s.confidence,
                analyzed_at=s.analyzed_at,
            )
        )
    for s in similarities:
        analysis_results.append(
            AnalysisResultResponse(
                analyzer_name="similarity",
                section_name=s.section_name,
                similarity_score=s.similarity_score,
                analyzed_at=s.analyzed_at,
            )
        )

    return FilingDetailResponse(
        accession_number=filing.metadata.accession_number,
        ticker=filing.metadata.ticker,
        company_name=filing.metadata.company_name,
        form_type=str(filing.metadata.form_type),
        filed_date=filing.metadata.filed_date,
        sections=filing.section_names(),
        analysis_status=analysis_status,
        section_texts={name: sec.raw_text for name, sec in filing.sections.items()},
        analysis_results=analysis_results,
    )


# -- Pipeline Triggers --


@router.post("/pipeline/ingest", response_model=JobResponse, status_code=202)
async def trigger_ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    state: AppState = Depends(get_app_state),
):
    """Trigger filing ingestion (async job)."""
    job_id = f"ingest-{uuid4().hex[:8]}"
    now = datetime.now(tz=_UTC).isoformat()
    job = JobStatus(job_id=job_id, status="pending", created_at=now)
    state.jobs[job_id] = job

    background_tasks.add_task(_run_ingest_job, state, job, request)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=datetime.fromisoformat(now),
        message=f"Ingestion job queued for {len(request.tickers)} tickers",
    )


@router.post("/pipeline/analyze", response_model=JobResponse, status_code=202)
async def trigger_analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    state: AppState = Depends(get_app_state),
):
    """Trigger analysis pipeline (async job)."""
    job_id = f"analyze-{uuid4().hex[:8]}"
    now = datetime.now(tz=_UTC).isoformat()
    job = JobStatus(job_id=job_id, status="pending", created_at=now)
    state.jobs[job_id] = job

    background_tasks.add_task(_run_analyze_job, state, job, request)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=datetime.fromisoformat(now),
        message=f"Analysis job queued with analyzers: {', '.join(request.analyzers)}",
    )


@router.post("/pipeline/backtest", response_model=JobResponse, status_code=202)
async def trigger_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    state: AppState = Depends(get_app_state),
):
    """Trigger backtest run (async job)."""
    job_id = f"backtest-{uuid4().hex[:8]}"
    now = datetime.now(tz=_UTC).isoformat()
    job = JobStatus(job_id=job_id, status="pending", created_at=now)
    state.jobs[job_id] = job

    background_tasks.add_task(_run_backtest_job, state, job, request)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=datetime.fromisoformat(now),
        message=f"Backtest job queued: {request.start_date} to {request.end_date}",
    )


# -- Jobs --


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    state: AppState = Depends(get_app_state),
):
    """Poll job status."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=datetime.fromisoformat(job.created_at),
        completed_at=(datetime.fromisoformat(job.completed_at) if job.completed_at else None),
        result=job.result,
        error=job.error,
    )


# -- Helpers --


async def _get_analysis_status(store: SqliteStore, accession_number: str) -> dict[str, bool]:
    """Build analyzer -> has_results mapping."""
    sentiments = await store.get_sentiments(filing_id=accession_number)
    similarities = await store.get_similarity(filing_id=accession_number)

    found_analyzers = {s.analyzer_name for s in sentiments}
    if similarities:
        found_analyzers.add("similarity")

    return {
        "dictionary": "dictionary" in found_analyzers,
        "similarity": "similarity" in found_analyzers,
        "llm": "llm" in found_analyzers,
    }


async def _run_ingest_job(
    state: AppState, job: JobStatus, request: IngestRequest
) -> None:
    """Execute ingestion in background."""
    job.status = "running"
    try:
        from edgar_sentinel.core.models import FormType
        from edgar_sentinel.ingestion import EdgarClient, FilingParser

        client = EdgarClient(state.config.edgar)
        parser = FilingParser()
        store = state.store

        form_types = [FormType(ft) for ft in request.form_types]
        ingested = 0

        for ticker in request.tickers:
            filings = await client.get_filings(
                ticker=ticker,
                form_types=form_types,
                start_year=request.start_year,
                end_year=request.end_year,
            )
            for filing_data in filings:
                parsed = parser.parse(filing_data)
                await store.save_filing(parsed)
                ingested += 1

        job.status = "completed"
        job.completed_at = datetime.now(tz=_UTC).isoformat()
        job.result = {"filings_ingested": ingested, "tickers": len(request.tickers)}

    except Exception as e:
        job.status = "failed"
        job.completed_at = datetime.now(tz=_UTC).isoformat()
        job.error = str(e)


async def _run_analyze_job(
    state: AppState, job: JobStatus, request: AnalyzeRequest
) -> None:
    """Execute analysis in background."""
    job.status = "running"
    try:
        store = state.store
        tickers = request.tickers or await store.get_all_tickers()
        total_results = 0

        for ticker in tickers:
            filings_meta = await store.list_filings(ticker=ticker)
            for meta in filings_meta:
                filing = await store.get_filing(meta.accession_number)
                if filing is None:
                    continue
                # Run requested analyzers
                for analyzer_name in request.analyzers:
                    if analyzer_name == "dictionary":
                        from edgar_sentinel.analyzers import DictionaryAnalyzer

                        analyzer = DictionaryAnalyzer(state.config.analyzers.dictionary)
                        results = await analyzer.analyze(filing)
                        for r in results:
                            await store.save_sentiment(r)
                        total_results += len(results)
                    elif analyzer_name == "similarity":
                        from edgar_sentinel.analyzers import SimilarityAnalyzer

                        analyzer = SimilarityAnalyzer(state.config.analyzers.similarity)
                        results = await analyzer.analyze(filing, store)
                        for r in results:
                            await store.save_similarity(r)
                        total_results += len(results)

        job.status = "completed"
        job.completed_at = datetime.now(tz=_UTC).isoformat()
        job.result = {"results_generated": total_results}

    except Exception as e:
        job.status = "failed"
        job.completed_at = datetime.now(tz=_UTC).isoformat()
        job.error = str(e)


async def _run_backtest_job(
    state: AppState, job: JobStatus, request: BacktestRequest
) -> None:
    """Execute backtest in background."""
    job.status = "running"
    try:
        from edgar_sentinel.backtest import run_backtest
        from edgar_sentinel.core.models import BacktestConfig, RebalanceFrequency

        store = state.store
        tickers = await store.get_all_tickers()

        config = BacktestConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            universe=tickers,
            rebalance_frequency=RebalanceFrequency(request.rebalance_frequency),
            num_quantiles=request.num_quantiles,
            short_quantile=None if request.long_only else request.num_quantiles,
            transaction_cost_bps=request.transaction_cost_bps,
        )

        signals = await store.get_composites()
        result = run_backtest(config=config, signals=signals)

        from edgar_sentinel.backtest.metrics import MetricsCalculator

        metrics = MetricsCalculator().compute_all(result)

        job.status = "completed"
        job.completed_at = datetime.now(tz=_UTC).isoformat()
        job.result = {
            "sharpe_ratio": result.sharpe_ratio,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "max_drawdown": result.max_drawdown,
            "metrics": metrics,
        }

    except Exception as e:
        job.status = "failed"
        job.completed_at = datetime.now(tz=_UTC).isoformat()
        job.error = str(e)
