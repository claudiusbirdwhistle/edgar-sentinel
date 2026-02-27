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
from edgar_sentinel.backtest.universe import StaticUniverseProvider, UniverseProvider
from edgar_sentinel.core.models import (
    BacktestConfig,
    BacktestResult,
    CompositeSignal,
    EquityCurvePoint,
    MonthlyReturn,
    RebalanceFrequency,
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
        universe_provider: UniverseProvider | None = None,
    ) -> None:
        self._config = config
        self._returns = returns_provider
        # Determine periods per year from rebalance frequency
        self._periods_per_year: float = (
            4.0
            if config.rebalance_frequency == RebalanceFrequency.QUARTERLY
            else 12.0
        )
        if metrics_calculator is None:
            self._metrics = MetricsCalculator(
                annualization_factor=int(self._periods_per_year)
            )
        else:
            self._metrics = metrics_calculator
        self.portfolio_history = PortfolioHistory()
        # Universe provider: enables survivorship-bias-free backtests by
        # filtering signals to only index members on each rebalance date.
        # Defaults to a static provider wrapping config.universe.
        self._universe_provider: UniverseProvider = (
            universe_provider
            if universe_provider is not None
            else StaticUniverseProvider(list(config.universe))
        )

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

        # Step 2: Determine superset of all tickers across all rebalance dates
        # (needed so we can fetch returns for tickers that may only appear on
        # certain dates when using a dynamic universe provider).
        all_universe_tickers: set[str] = set(self._config.universe)
        for rd in rebalance_dates:
            all_universe_tickers.update(self._universe_provider.get_tickers(rd))

        logger.info(
            f"Running backtest: {len(rebalance_dates)} rebalance dates, "
            f"{len(all_universe_tickers)} tickers (superset)"
        )

        # Step 2: Fetch returns for the full superset
        returns_df = self._returns.get_returns(
            tickers=list(all_universe_tickers),
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
            # 3a: Filter signals to the active universe on this rebalance date
            active_tickers = set(self._universe_provider.get_tickers(rebalance_date))
            date_signals = [s for s in signals if s.ticker in active_tickers]

            # 3b: Construct portfolio from filtered signals
            try:
                snapshot = constructor.construct(date_signals, rebalance_date)
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
                    period_start=rebalance_date,
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

        # Step 6: Compute equity curve
        equity_curve = self._compute_equity_curve(
            snapshots=self.portfolio_history.snapshots,
            period_returns=monthly_returns,
        )

        # Step 7: Package result
        return BacktestResult(
            config=self._config,
            total_return=float(np.prod(1 + returns_series) - 1),
            annualized_return=self._annualize(returns_series, self._periods_per_year),
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            information_ratio=metrics.get("information_ratio"),
            monthly_returns=monthly_returns,
            factor_exposures=metrics.get("factor_exposures"),
            turnover=self.portfolio_history.average_turnover,
            equity_curve=equity_curve,
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
    def _annualize(
        monthly_returns: pd.Series, periods_per_year: float = 12.0
    ) -> float:
        """Annualize a return series.

        annualized = (1 + total_return) ^ (periods_per_year / n_periods) - 1

        Parameters
        ----------
        monthly_returns : pd.Series
            Series of per-period returns (one value per rebalance period).
        periods_per_year : float
            Number of rebalance periods per calendar year.
            12 for monthly rebalancing, 4 for quarterly.
        """
        n = len(monthly_returns)
        if n == 0:
            return 0.0
        total = float(np.prod(1 + monthly_returns) - 1)
        return (1 + total) ** (periods_per_year / n) - 1

    def _compute_equity_curve(
        self,
        snapshots: list[PortfolioSnapshot],
        period_returns: list[MonthlyReturn],
        initial_value: float = 10_000.0,
    ) -> list[EquityCurvePoint]:
        """Compute a daily equity curve for portfolio, equal-weight, and SPY benchmarks.

        For each holding period defined by the rebalanced snapshots, fetches
        daily returns and compounds the portfolio value, a uniform equal-weight
        benchmark (all universe tickers), and the SPY ETF.

        Parameters
        ----------
        snapshots : list[PortfolioSnapshot]
            Ordered list of portfolio snapshots from portfolio_history.
        period_returns : list[MonthlyReturn]
            Ordered period return records providing period_start/period_end dates.
        initial_value : float
            Starting notional account value in dollars. Default: $10,000.

        Returns
        -------
        list[EquityCurvePoint]
            One entry per trading day across the full backtest window,
            in chronological order. Empty list if no daily data is available.
        """
        if not snapshots or not period_returns:
            return []

        # Fetch daily returns for universe + SPY
        all_tickers = list(self._config.universe) + ["SPY"]
        try:
            daily_returns = self._returns.get_returns(
                tickers=all_tickers,
                start=self._config.start_date,
                end=self._config.end_date,
                frequency="daily",
            )
        except Exception as exc:
            logger.warning("Could not fetch daily returns for equity curve: %s", exc)
            return []

        if daily_returns is None or daily_returns.empty:
            return []

        ew_tickers = [t for t in self._config.universe if t in daily_returns.columns]
        has_spy = "SPY" in daily_returns.columns

        portfolio_value = initial_value
        spy_value = initial_value
        ew_value = initial_value

        points: list[EquityCurvePoint] = []

        for snapshot, mr in zip(snapshots, period_returns):
            period_start = mr.period_start
            period_end = mr.period_end

            # Select daily rows strictly inside (period_start, period_end]
            mask = (daily_returns.index.date > period_start) & (
                daily_returns.index.date <= period_end
            )
            period_daily = daily_returns.loc[mask]

            for dt_idx, row in period_daily.iterrows():
                dt: date = dt_idx.date() if hasattr(dt_idx, "date") else dt_idx  # type: ignore[assignment]

                # Portfolio: weighted sum of position daily returns
                port_ret = 0.0
                for pos in snapshot.positions:
                    if pos.ticker in row.index:
                        val = row[pos.ticker]
                        if not pd.isna(val):
                            port_ret += pos.weight * float(val)
                portfolio_value *= 1.0 + port_ret

                # Equal-weight benchmark: average of available universe tickers
                if ew_tickers:
                    ew_vals = [
                        float(row[t])
                        for t in ew_tickers
                        if t in row.index and not pd.isna(row[t])
                    ]
                    ew_ret = sum(ew_vals) / len(ew_vals) if ew_vals else 0.0
                    ew_value *= 1.0 + ew_ret

                # SPY benchmark
                cur_spy: float | None = None
                if has_spy and "SPY" in row.index and not pd.isna(row["SPY"]):
                    spy_value *= 1.0 + float(row["SPY"])
                    cur_spy = spy_value

                points.append(
                    EquityCurvePoint(
                        date=dt,
                        portfolio_value=round(portfolio_value, 4),
                        spy_value=round(cur_spy, 4) if cur_spy is not None else None,
                        ew_value=round(ew_value, 4),
                    )
                )

        return points


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
