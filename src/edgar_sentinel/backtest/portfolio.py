"""Portfolio construction: quantile assignment, position sizing, turnover tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from edgar_sentinel.core.models import BacktestConfig, CompositeSignal, Ticker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Position:
    """A single portfolio position at a point in time."""

    ticker: str
    weight: float
    quantile: int
    signal_score: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Complete portfolio state at a single rebalance date."""

    rebalance_date: date
    positions: list[Position]
    long_positions: list[Position]
    short_positions: list[Position]
    turnover: float
    transaction_cost: float

    @property
    def n_long(self) -> int:
        return len(self.long_positions)

    @property
    def n_short(self) -> int:
        return len(self.short_positions)

    @property
    def tickers(self) -> list[str]:
        return [p.ticker for p in self.positions]


@dataclass
class PortfolioHistory:
    """Full sequence of portfolio snapshots across the backtest."""

    snapshots: list[PortfolioSnapshot] = field(default_factory=list)

    @property
    def rebalance_dates(self) -> list[date]:
        return [s.rebalance_date for s in self.snapshots]

    @property
    def average_turnover(self) -> float:
        if not self.snapshots:
            return 0.0
        turnovers = [s.turnover for s in self.snapshots]
        return float(np.mean(turnovers))

    @property
    def total_transaction_costs(self) -> float:
        return sum(s.transaction_cost for s in self.snapshots)


def generate_rebalance_dates(
    start: date,
    end: date,
    frequency: str,
) -> list[date]:
    """Generate rebalance dates between start and end.

    Parameters
    ----------
    frequency : str
        "monthly" -> last business day of each month
        "quarterly" -> last business day of Mar, Jun, Sep, Dec

    Returns a list of dates in ascending order, all within [start, end].
    """
    dates = pd.date_range(start=start, end=end, freq="BME")

    if frequency == "quarterly":
        dates = dates[dates.month.isin([3, 6, 9, 12])]
    elif frequency == "monthly":
        pass  # BME already gives monthly
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

    return [d.date() for d in dates]


class PortfolioConstructor:
    """Builds portfolio snapshots from ranked composite signals.

    Parameters
    ----------
    config : BacktestConfig
        Backtest configuration (num_quantiles, long/short quantiles,
        transaction_cost_bps).
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._prev_snapshot: PortfolioSnapshot | None = None

    def construct(
        self,
        signals: list[CompositeSignal],
        rebalance_date: date,
    ) -> PortfolioSnapshot:
        """Build a portfolio snapshot from signals at a rebalance date.

        Algorithm
        ---------
        1. Filter signals to those with signal_date <= rebalance_date.
           Take the most recent signal per ticker.
        2. Rank tickers by composite_score (descending: highest = best).
        3. Assign quantiles using pd.qcut (equal-frequency bins).
           Quantile 1 = highest scores, quantile N = lowest.
        4. Long positions: all tickers in long_quantile.
           Short positions: all tickers in short_quantile (if configured).
        5. Equal-weight within each leg.
        6. Compute turnover vs. previous snapshot.
        """
        # Step 1: Select most recent signal per ticker
        latest: dict[Ticker, CompositeSignal] = {}
        for sig in signals:
            if sig.signal_date <= rebalance_date:
                existing = latest.get(sig.ticker)
                if existing is None or sig.signal_date > existing.signal_date:
                    latest[sig.ticker] = sig

        if len(latest) < self._config.num_quantiles:
            raise ValueError(
                f"Only {len(latest)} tickers available but "
                f"{self._config.num_quantiles} quantiles requested"
            )

        # Step 2: Rank by composite_score descending
        ranked = sorted(latest.values(), key=lambda s: s.composite_score, reverse=True)

        # Step 3: Assign quantiles
        scores = [s.composite_score for s in ranked]
        tickers = [s.ticker for s in ranked]
        # Reverse labels so quantile 1 = highest scores, quantile N = lowest
        quantile_labels = pd.qcut(
            scores,
            q=self._config.num_quantiles,
            labels=range(self._config.num_quantiles, 0, -1),
            duplicates="drop",
        )

        # Step 4 & 5: Build positions
        positions: list[Position] = []
        long_positions: list[Position] = []
        short_positions: list[Position] = []

        # Group by quantile
        quantile_map: dict[int, list[tuple[str, float]]] = {}
        for ticker, q_label, score in zip(tickers, quantile_labels, scores):
            q = int(q_label)
            quantile_map.setdefault(q, []).append((ticker, score))

        # Long leg
        long_q = self._config.long_quantile
        if long_q in quantile_map:
            n_long = len(quantile_map[long_q])
            weight = 1.0 / n_long
            for ticker, score in quantile_map[long_q]:
                pos = Position(ticker=ticker, weight=weight, quantile=long_q, signal_score=score)
                positions.append(pos)
                long_positions.append(pos)

        # Short leg (optional)
        short_q = self._config.short_quantile
        if short_q is not None and short_q in quantile_map:
            n_short = len(quantile_map[short_q])
            weight = -1.0 / n_short
            for ticker, score in quantile_map[short_q]:
                pos = Position(ticker=ticker, weight=weight, quantile=short_q, signal_score=score)
                positions.append(pos)
                short_positions.append(pos)

        # Step 6: Compute turnover
        turnover = self._compute_turnover(positions)
        tx_cost = turnover * self._config.transaction_cost_bps / 10000.0

        snapshot = PortfolioSnapshot(
            rebalance_date=rebalance_date,
            positions=positions,
            long_positions=long_positions,
            short_positions=short_positions,
            turnover=turnover,
            transaction_cost=tx_cost,
        )

        self._prev_snapshot = snapshot
        return snapshot

    def _compute_turnover(self, new_positions: list[Position]) -> float:
        """Compute turnover as sum of absolute weight changes divided by 2.

        On first rebalance (no previous snapshot), turnover = 1.0.
        """
        if self._prev_snapshot is None:
            return 1.0

        old_weights: dict[str, float] = {
            p.ticker: p.weight for p in self._prev_snapshot.positions
        }
        new_weights: dict[str, float] = {
            p.ticker: p.weight for p in new_positions
        }

        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        abs_changes = sum(
            abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0))
            for t in all_tickers
        )

        return abs_changes / 2.0
