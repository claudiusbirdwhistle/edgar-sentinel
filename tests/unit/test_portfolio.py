"""Tests for backtest.portfolio: positions, snapshots, construction, rebalance dates."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from edgar_sentinel.backtest.portfolio import (
    PortfolioConstructor,
    PortfolioHistory,
    PortfolioSnapshot,
    Position,
    generate_rebalance_dates,
)
from edgar_sentinel.core.models import BacktestConfig, CompositeSignal


# --- Fixtures ---


def _make_signals(
    tickers: list[str],
    signal_date: date,
    scores: list[float] | None = None,
) -> list[CompositeSignal]:
    """Helper to create CompositeSignal list."""
    if scores is None:
        np.random.seed(42)
        scores = list(np.random.normal(0.0, 1.0, len(tickers)))
    return [
        CompositeSignal(
            ticker=t,
            signal_date=signal_date,
            composite_score=s,
            components={"dict": 0.5, "sim": 0.5},
        )
        for t, s in zip(tickers, scores)
    ]


def _make_config(**overrides) -> BacktestConfig:
    defaults = dict(
        start_date=date(2020, 1, 1),
        end_date=date(2022, 12, 31),
        universe=["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                   "NVDA", "TSLA", "JPM", "BAC", "WMT"],
        rebalance_frequency="quarterly",
        num_quantiles=5,
        long_quantile=1,
        short_quantile=5,
        transaction_cost_bps=10,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


# --- Position tests ---


class TestPosition:
    def test_creation(self):
        pos = Position(ticker="AAPL", weight=0.5, quantile=1, signal_score=0.8)
        assert pos.ticker == "AAPL"
        assert pos.weight == 0.5
        assert pos.quantile == 1
        assert pos.signal_score == 0.8

    def test_frozen(self):
        pos = Position(ticker="AAPL", weight=0.5, quantile=1, signal_score=0.8)
        with pytest.raises(AttributeError):
            pos.weight = 0.3


# --- PortfolioSnapshot tests ---


class TestPortfolioSnapshot:
    def test_properties(self):
        long = [Position("AAPL", 0.5, 1, 0.8), Position("MSFT", 0.5, 1, 0.7)]
        short = [Position("BAC", -0.5, 5, -0.8), Position("WMT", -0.5, 5, -0.7)]
        snap = PortfolioSnapshot(
            rebalance_date=date(2020, 3, 31),
            positions=long + short,
            long_positions=long,
            short_positions=short,
            turnover=1.0,
            transaction_cost=0.001,
        )
        assert snap.n_long == 2
        assert snap.n_short == 2
        assert snap.tickers == ["AAPL", "MSFT", "BAC", "WMT"]

    def test_long_only(self):
        long = [Position("AAPL", 1.0, 1, 0.9)]
        snap = PortfolioSnapshot(
            rebalance_date=date(2020, 3, 31),
            positions=long,
            long_positions=long,
            short_positions=[],
            turnover=1.0,
            transaction_cost=0.001,
        )
        assert snap.n_short == 0
        assert snap.n_long == 1


# --- PortfolioHistory tests ---


class TestPortfolioHistory:
    def test_empty(self):
        history = PortfolioHistory()
        assert history.rebalance_dates == []
        assert history.average_turnover == 0.0
        assert history.total_transaction_costs == 0.0

    def test_with_snapshots(self):
        snap1 = PortfolioSnapshot(
            rebalance_date=date(2020, 3, 31),
            positions=[], long_positions=[], short_positions=[],
            turnover=1.0, transaction_cost=0.001,
        )
        snap2 = PortfolioSnapshot(
            rebalance_date=date(2020, 6, 30),
            positions=[], long_positions=[], short_positions=[],
            turnover=0.4, transaction_cost=0.0004,
        )
        history = PortfolioHistory(snapshots=[snap1, snap2])
        assert history.rebalance_dates == [date(2020, 3, 31), date(2020, 6, 30)]
        assert history.average_turnover == pytest.approx(0.7, abs=0.01)
        assert history.total_transaction_costs == pytest.approx(0.0014, abs=1e-6)


# --- generate_rebalance_dates tests ---


class TestRebalanceDates:
    def test_quarterly(self):
        dates = generate_rebalance_dates(
            date(2020, 1, 1), date(2020, 12, 31), "quarterly"
        )
        # Expect 4 quarterly dates in 2020 (Mar, Jun, Sep, Dec)
        assert len(dates) == 4
        assert all(d.month in (3, 6, 9, 12) for d in dates)
        assert all(date(2020, 1, 1) <= d <= date(2020, 12, 31) for d in dates)

    def test_monthly(self):
        dates = generate_rebalance_dates(
            date(2020, 1, 1), date(2020, 6, 30), "monthly"
        )
        assert len(dates) == 6
        assert dates[0].month == 1
        assert dates[-1].month == 6

    def test_ascending_order(self):
        dates = generate_rebalance_dates(
            date(2020, 1, 1), date(2022, 12, 31), "quarterly"
        )
        assert dates == sorted(dates)

    def test_invalid_frequency(self):
        with pytest.raises(ValueError, match="Unsupported frequency"):
            generate_rebalance_dates(date(2020, 1, 1), date(2020, 12, 31), "weekly")

    def test_no_dates_in_range(self):
        # Very short range with no business month end
        dates = generate_rebalance_dates(
            date(2020, 1, 1), date(2020, 1, 15), "quarterly"
        )
        assert len(dates) == 0

    def test_dates_are_business_days(self):
        dates = generate_rebalance_dates(
            date(2020, 1, 1), date(2020, 12, 31), "monthly"
        )
        for d in dates:
            assert d.weekday() < 5  # Mon-Fri


# --- PortfolioConstructor tests ---


class TestPortfolioConstructor:
    def test_basic_long_short(self):
        config = _make_config()
        constructor = PortfolioConstructor(config)
        tickers = config.universe
        # High scores for AAPL/MSFT (quantile 1), low for BAC/WMT (quantile 5)
        scores = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8]
        signals = _make_signals(tickers, date(2020, 3, 1), scores)

        snap = constructor.construct(signals, date(2020, 3, 31))

        # Should have long and short positions
        assert snap.n_long > 0
        assert snap.n_short > 0
        # Long weights positive, short weights negative
        for pos in snap.long_positions:
            assert pos.weight > 0
        for pos in snap.short_positions:
            assert pos.weight < 0

    def test_long_weights_sum_to_one(self):
        config = _make_config()
        constructor = PortfolioConstructor(config)
        signals = _make_signals(config.universe, date(2020, 3, 1))
        snap = constructor.construct(signals, date(2020, 3, 31))

        long_sum = sum(p.weight for p in snap.long_positions)
        assert long_sum == pytest.approx(1.0, abs=0.01)

    def test_short_weights_sum_to_minus_one(self):
        config = _make_config()
        constructor = PortfolioConstructor(config)
        signals = _make_signals(config.universe, date(2020, 3, 1))
        snap = constructor.construct(signals, date(2020, 3, 31))

        short_sum = sum(p.weight for p in snap.short_positions)
        assert short_sum == pytest.approx(-1.0, abs=0.01)

    def test_long_only(self):
        config = _make_config(short_quantile=None)
        constructor = PortfolioConstructor(config)
        signals = _make_signals(config.universe, date(2020, 3, 1))
        snap = constructor.construct(signals, date(2020, 3, 31))

        assert snap.n_long > 0
        assert snap.n_short == 0
        assert len(snap.short_positions) == 0

    def test_first_turnover_is_one(self):
        config = _make_config()
        constructor = PortfolioConstructor(config)
        signals = _make_signals(config.universe, date(2020, 3, 1))
        snap = constructor.construct(signals, date(2020, 3, 31))

        assert snap.turnover == 1.0

    def test_identical_portfolio_zero_turnover(self):
        config = _make_config()
        constructor = PortfolioConstructor(config)
        signals = _make_signals(config.universe, date(2020, 3, 1))

        # First rebalance
        constructor.construct(signals, date(2020, 3, 31))
        # Second rebalance with same signals
        snap2 = constructor.construct(signals, date(2020, 6, 30))

        assert snap2.turnover == pytest.approx(0.0, abs=1e-10)

    def test_transaction_cost(self):
        config = _make_config(transaction_cost_bps=20)
        constructor = PortfolioConstructor(config)
        signals = _make_signals(config.universe, date(2020, 3, 1))
        snap = constructor.construct(signals, date(2020, 3, 31))

        expected_cost = 1.0 * 20 / 10000.0  # turnover=1.0 on first rebalance
        assert snap.transaction_cost == pytest.approx(expected_cost)

    def test_too_few_tickers(self):
        config = _make_config(num_quantiles=5)
        constructor = PortfolioConstructor(config)
        # Only 3 tickers, but need 5 quantiles
        signals = _make_signals(["AAPL", "MSFT", "GOOGL"], date(2020, 3, 1))

        with pytest.raises(ValueError, match="tickers available"):
            constructor.construct(signals, date(2020, 3, 31))

    def test_filters_future_signals(self):
        config = _make_config()
        constructor = PortfolioConstructor(config)
        # Signals dated AFTER the rebalance date should be excluded
        future_signals = _make_signals(config.universe, date(2021, 1, 1))
        past_signals = _make_signals(config.universe, date(2020, 2, 1))
        all_signals = future_signals + past_signals

        snap = constructor.construct(all_signals, date(2020, 3, 31))
        # Only past signals should be used
        assert snap.n_long > 0

    def test_uses_most_recent_signal(self):
        config = _make_config()
        constructor = PortfolioConstructor(config)
        tickers = config.universe
        # Old signals with low scores
        old_signals = _make_signals(tickers, date(2020, 1, 1), [-1.0] * 10)
        # Recent signals with varied scores
        new_scores = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8]
        new_signals = _make_signals(tickers, date(2020, 3, 1), new_scores)

        snap = constructor.construct(old_signals + new_signals, date(2020, 3, 31))
        # AAPL should be in long (highest score in new signals)
        long_tickers = [p.ticker for p in snap.long_positions]
        assert "AAPL" in long_tickers

    def test_quantile_assignment(self):
        config = _make_config()
        constructor = PortfolioConstructor(config)
        signals = _make_signals(config.universe, date(2020, 3, 1))
        snap = constructor.construct(signals, date(2020, 3, 31))

        # All positions should have valid quantiles
        for pos in snap.positions:
            assert 1 <= pos.quantile <= config.num_quantiles

    def test_equal_weight_within_quantile(self):
        config = _make_config()
        constructor = PortfolioConstructor(config)
        signals = _make_signals(config.universe, date(2020, 3, 1))
        snap = constructor.construct(signals, date(2020, 3, 31))

        if len(snap.long_positions) > 1:
            weights = [p.weight for p in snap.long_positions]
            assert all(w == pytest.approx(weights[0]) for w in weights)
