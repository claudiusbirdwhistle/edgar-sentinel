# edgar-sentinel

Trading signal generator from SEC EDGAR filing sentiment analysis.

## Overview

edgar-sentinel is a Python library that generates quantitative trading signals by analyzing the sentiment and textual changes in SEC EDGAR filings (10-K and 10-Q). It combines dictionary-based sentiment scoring, document similarity analysis, and optional LLM-powered interpretation to produce actionable signals with built-in backtesting.

## Installation

```bash
pip install edgar-sentinel
```

With optional extras:
```bash
pip install edgar-sentinel[api]       # FastAPI server
pip install edgar-sentinel[llm]       # LLM analyzer support
pip install edgar-sentinel[backtest]  # Backtesting with yfinance
pip install edgar-sentinel[all]       # Everything
```

## Quick Start

```python
from edgar_sentinel import Pipeline

pipeline = Pipeline(
    universe=["AAPL", "MSFT", "GOOGL"],
    start_year=2020,
    end_year=2025,
)

pipeline.ingest()
signals = pipeline.analyze()
df = pipeline.to_dataframe()
```

## License

MIT
