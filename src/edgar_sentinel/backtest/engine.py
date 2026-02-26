"""Core backtest simulation engine."""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from edgar_sentinel.backtest.metrics import MetricsCalculator
from edgar_sentinel.backtest.portfolio import (
    PortfolioConstructor,
    PortfolioHistory,
    PortfolioSnapshot,
    generate_rebalance_dates,
)
from edgar_sentinel.backtest.returns import ReturnsProvider
from edgar_sentinel.core.models import (
    BacktestConfig,
    BacktestResult,
    CompositeSignal,
    MonthlyReturn,
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Core backtest simulation engine.

    Parameters
    ----------
    config : BacktestConfig
        Full backtest configuration.
    returns_provider : ReturnsProvider
        Data source for historical returns.
    metrics_calculator : MetricsCalculator | None
        Custom metrics calculator. If None, uses default with
        risk_free_rate=0.0.
    """

    def __init__(
        self,
        config: BacktestConfig,
        returns_provider: ReturnsProvider,
        metrics_calculator: MetricsCalculator | None = None,
    ) -> None:
        self._config = config
        self._returns = returns_provider
        self._metrics = metrics_calculator or MetricsCalculator()
        self.portfolio_history = PortfolioHistory()

    def run(self, signals: list[CompositeSignal]) -> BacktestResult:
        """Execute the full backtest simulation.

        Algorithm
        ---------
        1. Generate rebalance dates from config.
        2. Fetch returns for the full universe over the backtest window.
        3. For each rebalance date:
           a. Construct portfolio from available signals.
           b. Compute portfolio return over the holding period.
           c. Deduct transaction costs.
        4. Aggregate into monthly return series.
        5. Compute metrics.
        6. Package into BacktestResult.
        """
        # Step 1: Generate rebalance dates
        rebalance_dates = generate_rebalance_dates(
            self._config.start_date,
            self._config.end_date,
            self._config.rebalance_frequency.value,
        )

        if not rebalance_dates:
            raise ValueError(
                f"No rebalance dates between {self._config.start_date} "
                f"and {self._config.end_date}"
            )

        logger.info(
            f"Running backtest: {len(rebalance_dates)} rebalance dates, "
            f"{len(self._config.universe)} tickers"
        )

        # Step 2: Fetch returns
        returns_df = self._returns.get_returns(
            tickers=list(self._config.universe),
            start=self._config.start_date,
            end=self._config.end_date,
            frequency="monthly",
        )

        if returns_df.empty:
            raise ValueError("Returns provider returned no data")

        # Step 3: Simulate
        constructor = PortfolioConstructor(self._config)
        monthly_returns: list[MonthlyReturn] = []
        portfolio_returns: list[float] = []

        for i, rebalance_date in enumerate(rebalance_dates):
            # 3a: Construct portfolio
            try:
                snapshot = constructor.construct(signals, rebalance_date)
            except ValueError as e:
                logger.warning(f"Skipping rebalance {rebalance_date}: {e}")
                continue

            self.portfolio_history.snapshots.append(snapshot)

            # 3b: Compute return over holding period
            if i + 1 < len(rebalance_dates):
                period_end = rebalance_dates[i + 1]
            else:
                period_end = self._config.end_date

            period_return = self._compute_period_return(
                snapshot, returns_df, rebalance_date, period_end
            )

            # 3c: Deduct transaction costs
            net_return = period_return - snapshot.transaction_cost
            portfolio_returns.append(net_return)

            # Decompose into long/short
            long_ret = self._compute_leg_return(
                snapshot.long_positions, returns_df, rebalance_date, period_end
            )
            short_ret = (
                self._compute_leg_return(
                    snapshot.short_positions, returns_df, rebalance_date, period_end
                )
                if snapshot.short_positions
                else None
            )

            ls_ret = net_return if short_ret is not None else None

            monthly_returns.append(
                MonthlyReturn(
                    period_end=period_end,
                    long_return=long_ret,
                    short_return=short_ret,
                    long_short_return=ls_ret,
                )
            )

        if not portfolio_returns:
            raise ValueError("No valid rebalance periods produced returns")

        # Step 5: Compute metrics
        returns_series = pd.Series(
            portfolio_returns,
            index=pd.DatetimeIndex([mr.period_end for mr in monthly_returns]),
        )

        metrics = self._metrics.compute_all(returns_series, signals, returns_df)

        # Step 6: Package result
        return BacktestResult(
            config=self._config,
            total_return=float(np.prod(1 + returns_series) - 1),
            annualized_return=self._annualize(returns_series),
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            information_ratio=metrics.get("information_ratio"),
            monthly_returns=monthly_returns,
            factor_exposures=metrics.get("factor_exposures"),
            turnover=self.portfolio_history.average_turnover,
        )

    def _compute_period_return(
        self,
        snapshot: PortfolioSnapshot,
        returns_df: pd.DataFrame,
        start: date,
        end: date,
    ) -> float:
        """Compute weighted portfolio return over a holding period."""
        total = 0.0
        for pos in snapshot.positions:
            ticker_ret = self._ticker_period_return(pos.ticker, returns_df, start, end)
            total += pos.weight * ticker_ret
        return total

    def _compute_leg_return(
        self,
        positions: list[Position],
        returns_df: pd.DataFrame,
        start: date,
        end: date,
    ) -> float:
        """Compute return for one leg (long or short) of the portfolio."""
        if not positions:
            return 0.0
        total = 0.0
        for pos in positions:
            ticker_ret = self._ticker_period_return(pos.ticker, returns_df, start, end)
            total += abs(pos.weight) * ticker_ret
        return total

    def _ticker_period_return(
        self,
        ticker: str,
        returns_df: pd.DataFrame,
        start: date,
        end: date,
    ) -> float:
        """Cumulative return for a single ticker over (start, end].

        Uses compound returns: prod(1 + r_i) - 1
        """
        if ticker not in returns_df.columns:
            logger.warning(f"No return data for {ticker}, assuming 0.0")
            return 0.0

        mask = (returns_df.index.date > start) & (returns_df.index.date <= end)
        period_rets = returns_df.loc[mask, ticker].dropna()

        if period_rets.empty:
            return 0.0

        return float(np.prod(1 + period_rets) - 1)

    @staticmethod
    def _annualize(monthly_returns: pd.Series) -> float:
        """Annualize a monthly return series.

        annualized = (1 + total_return) ^ (12 / n_months) - 1
        """
        n = len(monthly_returns)
        if n == 0:
            return 0.0
        total = float(np.prod(1 + monthly_returns) - 1)
        return (1 + total) ** (12.0 / n) - 1


def run_backtest(
    config: BacktestConfig,
    signals: list[CompositeSignal],
    returns_provider: ReturnsProvider | None = None,
) -> BacktestResult:
    """One-call convenience function for running a backtest.

    Creates a YFinanceProvider if no provider is given.
    """
    if returns_provider is None:
        from edgar_sentinel.backtest.returns import YFinanceProvider

        returns_provider = YFinanceProvider()

    engine = BacktestEngine(config=config, returns_provider=returns_provider)
    return engine.run(signals)
