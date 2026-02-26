"""Backtest module: returns providers, portfolio construction, engine, metrics."""

from edgar_sentinel.backtest.engine import BacktestEngine, run_backtest
from edgar_sentinel.backtest.metrics import MetricsCalculator
from edgar_sentinel.backtest.portfolio import (
    PortfolioConstructor,
    PortfolioHistory,
    PortfolioSnapshot,
    Position,
    generate_rebalance_dates,
)
from edgar_sentinel.backtest.returns import (
    CSVProvider,
    ReturnsProvider,
    YFinanceProvider,
)

__all__ = [
    "BacktestEngine",
    "CSVProvider",
    "MetricsCalculator",
    "PortfolioConstructor",
    "PortfolioHistory",
    "PortfolioSnapshot",
    "Position",
    "ReturnsProvider",
    "YFinanceProvider",
    "generate_rebalance_dates",
    "run_backtest",
]
