# EDGAR Sentinel

Trading signal generator from SEC EDGAR filing sentiment analysis.

EDGAR Sentinel ingests 10-K and 10-Q filings from the SEC EDGAR database,
extracts key sections (MD&A, Risk Factors), runs sentiment and similarity
analyzers, and produces cross-sectionally normalized trading signals with
optional backtesting.

## Features

- **EDGAR Ingestion** — rate-limited client with CIK/ticker resolution and filing parsing
- **Dictionary Analyzer** — Loughran-McDonald financial sentiment scoring
- **Similarity Analyzer** — TF-IDF cosine similarity between consecutive filings
- **LLM Analyzer** — optional Claude/OpenAI integration for nuanced sentiment
- **Signal Builder** — z-score normalization, percentile ranking, time decay
- **Composite Signals** — equal-weight and IC-weighted ensemble scoring
- **Backtesting** — long/short portfolio simulation with Sharpe, IC, and factor analysis
- **REST API** — FastAPI with async endpoints for signals, filings, and pipeline triggers
- **CLI** — `edgar-sentinel` command with ingest, analyze, signals, backtest, serve, status

## Quick Start

### Installation

```bash
# Core (CLI + analyzers)
pip install edgar-sentinel

# With API server
pip install edgar-sentinel[api]

# With backtesting (yfinance + statsmodels)
pip install edgar-sentinel[backtest]

# Everything
pip install edgar-sentinel[all]

# Development (includes tests, linting, type checking)
pip install -e ".[dev]"
```

### Configuration

EDGAR Sentinel uses a YAML config file or environment variables.

**Environment variables** (prefix `EDGAR_SENTINEL_`):

```bash
export EDGAR_SENTINEL_EDGAR__USER_AGENT="YourName your@email.com"
export EDGAR_SENTINEL_STORAGE__DATABASE_URL="sqlite:///data/sentinel.db"
```

**Config file** (`edgar-sentinel.yml`):

```yaml
edgar:
  user_agent: "YourName your@email.com"
  rate_limit: 10  # requests per second (SEC limit)

storage:
  backend: sqlite
  sqlite_path: data/sentinel.db

analyzers:
  dictionary:
    enabled: true
  similarity:
    enabled: true
  llm:
    enabled: false
    provider: claude
```

Pass the config file via `--config` flag or `EDGAR_SENTINEL_CONFIG` env var.

### Usage

```bash
# 1. Ingest filings
edgar-sentinel ingest --tickers AAPL,MSFT,GOOGL --form-type 10-K --start 2020 --end 2025

# 2. Run analyzers
edgar-sentinel analyze --analyzer dictionary --analyzer similarity

# 3. View signals
edgar-sentinel signals --format table
edgar-sentinel signals --ticker AAPL --format json

# 4. Backtest
edgar-sentinel backtest --start 2020-01-01 --end 2025-12-31 --quantiles 5

# 5. Start API server
edgar-sentinel serve --port 8000

# 6. Check status
edgar-sentinel status
```

### API

When running the API server, interactive docs are at `http://localhost:8000/docs`.

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check and statistics |
| GET | `/api/signals` | List composite signals (filterable) |
| GET | `/api/signals/{ticker}` | Signals for a specific ticker |
| GET | `/api/filings` | List ingested filings |
| GET | `/api/filings/{accession}` | Filing detail with sections |
| POST | `/api/pipeline/ingest` | Trigger ingestion job |
| POST | `/api/pipeline/analyze` | Trigger analysis job |
| POST | `/api/pipeline/backtest` | Trigger backtest job |
| GET | `/api/jobs/{job_id}` | Check job status |

## Docker

### Production

```bash
# Set required env vars
export EDGAR_USER_AGENT="YourName your@email.com"

# Start API server
docker compose -f docker/docker-compose.yml up -d

# Run an ingestion job
docker compose -f docker/docker-compose.yml run --rm edgar-sentinel \
    ingest --tickers AAPL,MSFT --form-type 10-K --start 2020 --end 2025
```

### Development

```bash
# Run tests
docker compose -f docker/docker-compose.dev.yml run --rm edgar-sentinel-dev

# API with hot-reload
docker compose -f docker/docker-compose.dev.yml up edgar-sentinel-api
```

## Architecture

```
EDGAR API ──> EdgarClient ──> FilingParser ──> SqliteStore
                                                   │
                                    ┌──────────────┼──────────────┐
                                    ▼              ▼              ▼
                              Dictionary     Similarity        LLM
                              Analyzer       Analyzer        Analyzer
                                    │              │              │
                                    └──────────────┼──────────────┘
                                                   ▼
                                            SignalBuilder
                                                   │
                                                   ▼
                                           SignalComposite
                                                   │
                                    ┌──────────────┼──────────────┐
                                    ▼              ▼              ▼
                                  CLI           REST API      Backtest
```

All modules are async-first with SQLite (WAL mode) for storage.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (excludes e2e network tests)
pytest

# Run with coverage
pytest --cov=edgar_sentinel --cov-report=term-missing

# Run e2e tests (requires network)
pytest tests/e2e/ -m e2e -v

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/edgar_sentinel/
```

## License

MIT
