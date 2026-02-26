"""Tests for backtest.engine: BacktestEngine and run_backtest."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from edgar_sentinel.backtest.engine import BacktestEngine, run_backtest
from edgar_sentinel.backtest.metrics import MetricsCalculator
from edgar_sentinel.core.models import BacktestConfig, CompositeSignal, EquityCurvePoint


# --- Mock Returns Provider ---


class MockReturnsProvider:
    """In-memory returns provider for testing."""

    def __init__(self, returns_df: pd.DataFrame):
        self._returns = returns_df

    def get_returns(
        self, tickers: list[str], start: date, end: date, frequency: str = "monthly"
    ) -> pd.DataFrame:
        mask = (self._returns.index.date >= start) & (self._returns.index.date <= end)
        cols = [t for t in tickers if t in self._returns.columns]
        return self._returns.loc[mask, cols]

    def get_prices(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        return self._returns  # Not needed for engine tests


class MockFrequencyReturnsProvider:
    """Mock provider that returns different DataFrames for daily vs monthly frequencies."""

    def __init__(self, monthly_df: pd.DataFrame, daily_df: pd.DataFrame):
        self._monthly = monthly_df
        self._daily = daily_df

    def get_returns(
        self, tickers: list[str], start: date, end: date, frequency: str = "monthly"
    ) -> pd.DataFrame:
        df = self._daily if frequency == "daily" else self._monthly
        mask = (df.index.date >= start) & (df.index.date <= end)
        cols = [t for t in tickers if t in df.columns]
        return df.loc[mask, cols]

    def get_prices(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        return self._daily


# --- Fixtures ---


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


def _make_returns(seed: int = 42) -> pd.DataFrame:
    """Synthetic monthly returns for 10 tickers over 36 months."""
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
               "NVDA", "TSLA", "JPM", "BAC", "WMT"]
    np.random.seed(seed)
    data = np.random.normal(0.01, 0.05, size=(36, 10))
    return pd.DataFrame(data, index=dates, columns=tickers)


def _make_signals(
    tickers: list[str] | None = None,
    dates: list[date] | None = None,
) -> list[CompositeSignal]:
    """Generate quarterly composite signals."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                    "NVDA", "TSLA", "JPM", "BAC", "WMT"]
    if dates is None:
        dates = [
            date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30), date(2020, 12, 31),
            date(2021, 3, 31), date(2021, 6, 30), date(2021, 9, 30), date(2021, 12, 31),
            date(2022, 3, 31), date(2022, 6, 30), date(2022, 9, 30), date(2022, 12, 31),
        ]
    signals = []
    for q_date in dates:
        np.random.seed(hash(str(q_date)) % 2**31)
        for ticker in tickers:
            score = np.random.normal(0.0, 1.0)
            signals.append(
                CompositeSignal(
                    ticker=ticker,
                    signal_date=q_date,
                    composite_score=score,
                    components={"dictionary_mda": 0.5, "similarity_mda": 0.5},
                )
            )
    return signals


# --- BacktestEngine tests ---


class TestBacktestEngine:
    def test_basic_run(self):
        config = _make_config()
        returns_df = _make_returns()
        provider = MockReturnsProvider(returns_df)
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        assert result.config == config
        assert isinstance(result.total_return, float)
        assert isinstance(result.annualized_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.turnover, float)
        assert len(result.monthly_returns) > 0

    def test_portfolio_history_populated(self):
        config = _make_config()
        provider = MockReturnsProvider(_make_returns())
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        engine.run(signals)

        assert len(engine.portfolio_history.snapshots) > 0

    def test_long_only(self):
        config = _make_config(short_quantile=None)
        provider = MockReturnsProvider(_make_returns())
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        # Monthly returns should not have short or long-short returns
        for mr in result.monthly_returns:
            assert mr.short_return is None
            assert mr.long_short_return is None

    def test_long_short(self):
        config = _make_config(short_quantile=5)
        provider = MockReturnsProvider(_make_returns())
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        # Monthly returns should have short and long-short returns
        for mr in result.monthly_returns:
            assert mr.short_return is not None
            assert mr.long_short_return is not None

    def test_deterministic(self):
        config = _make_config()
        returns_df = _make_returns(seed=42)
        signals = _make_signals()

        engine1 = BacktestEngine(config, MockReturnsProvider(returns_df))
        result1 = engine1.run(signals)

        engine2 = BacktestEngine(config, MockReturnsProvider(returns_df))
        result2 = engine2.run(signals)

        assert result1.total_return == result2.total_return
        assert result1.sharpe_ratio == result2.sharpe_ratio
        assert result1.max_drawdown == result2.max_drawdown

    def test_no_lookahead(self):
        config = _make_config()
        provider = MockReturnsProvider(_make_returns())

        # Only provide future signals (dated after all rebalance dates)
        future_signals = _make_signals(dates=[date(2023, 6, 30)])
        # Also provide a few valid signals so at least some rebalances work
        past_signals = _make_signals(dates=[date(2020, 3, 31)])

        engine = BacktestEngine(config, provider)
        # The engine should only use past_signals for the 2020-Q1 rebalance
        result = engine.run(past_signals + future_signals)
        assert len(result.monthly_returns) > 0

    def test_empty_returns_raises(self):
        config = _make_config()
        # Create a properly-typed empty DataFrame with DatetimeIndex
        empty_df = pd.DataFrame(
            index=pd.DatetimeIndex([], name="date"),
            columns=config.universe,
            dtype=float,
        )
        empty_provider = MockReturnsProvider(empty_df)
        signals = _make_signals()

        engine = BacktestEngine(config, empty_provider)
        with pytest.raises(ValueError, match="no data"):
            engine.run(signals)

    def test_no_rebalance_dates_raises(self):
        # Config with very short period
        config = _make_config(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 15),
        )
        provider = MockReturnsProvider(_make_returns())
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        with pytest.raises(ValueError, match="No rebalance dates"):
            engine.run(signals)

    def test_transaction_costs_deducted(self):
        # High transaction costs should reduce returns
        config_high = _make_config(transaction_cost_bps=100)
        config_low = _make_config(transaction_cost_bps=0)
        returns_df = _make_returns()
        signals = _make_signals()

        engine_high = BacktestEngine(config_high, MockReturnsProvider(returns_df))
        engine_low = BacktestEngine(config_low, MockReturnsProvider(returns_df))

        result_high = engine_high.run(signals)
        result_low = engine_low.run(signals)

        # Higher costs should lead to lower total return
        assert result_high.total_return < result_low.total_return

    def test_custom_metrics_calculator(self):
        config = _make_config()
        provider = MockReturnsProvider(_make_returns())
        signals = _make_signals()

        calc = MetricsCalculator(risk_free_rate=0.05)
        engine = BacktestEngine(config, provider, metrics_calculator=calc)
        result = engine.run(signals)

        # Should work with custom calculator
        assert isinstance(result.sharpe_ratio, float)

    def test_monthly_returns_have_dates(self):
        config = _make_config()
        provider = MockReturnsProvider(_make_returns())
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        for mr in result.monthly_returns:
            assert isinstance(mr.period_end, date)
            assert isinstance(mr.long_return, float)


class TestRunBacktest:
    def test_convenience_function(self):
        config = _make_config()
        returns_df = _make_returns()
        provider = MockReturnsProvider(returns_df)
        signals = _make_signals()

        result = run_backtest(config, signals, returns_provider=provider)
        assert isinstance(result.total_return, float)
        assert len(result.monthly_returns) > 0


class TestAnnualize:
    def test_positive_returns(self):
        # 12 months of 1% returns
        returns = pd.Series([0.01] * 12)
        ann = BacktestEngine._annualize(returns)
        expected = (1.01**12) ** (12.0 / 12) - 1
        assert ann == pytest.approx(expected, rel=1e-6)

    def test_zero_returns(self):
        returns = pd.Series([0.0] * 12)
        ann = BacktestEngine._annualize(returns)
        assert ann == pytest.approx(0.0)

    def test_empty_returns(self):
        returns = pd.Series([], dtype=float)
        ann = BacktestEngine._annualize(returns)
        assert ann == 0.0

    def test_partial_year(self):
        # 6 months -> annualized
        returns = pd.Series([0.01] * 6)
        ann = BacktestEngine._annualize(returns)
        total = (1.01**6) - 1
        expected = (1 + total) ** (12.0 / 6) - 1
        assert ann == pytest.approx(expected, rel=1e-6)

    def test_quarterly_factor(self):
        # 4 quarters at 3% each → annualized with periods_per_year=4
        returns = pd.Series([0.03] * 4)
        ann = BacktestEngine._annualize(returns, periods_per_year=4.0)
        total = (1.03 ** 4) - 1
        expected = (1 + total) ** (4.0 / 4) - 1  # same as total for 1 year
        assert ann == pytest.approx(expected, rel=1e-6)

    def test_quarterly_partial_year(self):
        # 2 quarters at 3% each → annualized with periods_per_year=4
        returns = pd.Series([0.03] * 2)
        ann = BacktestEngine._annualize(returns, periods_per_year=4.0)
        total = (1.03 ** 2) - 1
        expected = (1 + total) ** (4.0 / 2) - 1
        assert ann == pytest.approx(expected, rel=1e-6)


class TestQuarterlyAnnualization:
    """Tests verifying the engine correctly annualizes quarterly backtests."""

    def test_quarterly_engine_uses_periods_per_year_4(self):
        """BacktestEngine with quarterly config should use annualization_factor=4."""
        config = _make_config(rebalance_frequency="quarterly")
        provider = MockReturnsProvider(_make_returns())
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        engine.run(signals)

        # The engine should have configured its MetricsCalculator with factor 4
        assert engine._metrics._af == 4

    def test_monthly_engine_uses_periods_per_year_12(self):
        """BacktestEngine with monthly config should use annualization_factor=12."""
        config = _make_config(rebalance_frequency="monthly")
        provider = MockReturnsProvider(_make_returns())
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        engine.run(signals)

        assert engine._metrics._af == 12

    def test_quarterly_annualized_return_correct(self):
        """Quarterly annualized return should use 4 periods/year, not 12."""
        config = _make_config(rebalance_frequency="quarterly")
        provider = MockReturnsProvider(_make_returns())
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        # Re-derive expected: (1 + total_return) ^ (4 / n_periods) - 1
        n = len(result.monthly_returns)
        expected = (1 + result.total_return) ** (4.0 / n) - 1
        assert result.annualized_return == pytest.approx(expected, rel=1e-6)

    def test_monthly_returns_have_period_start(self):
        """MonthlyReturn entries should include a non-None period_start date."""
        config = _make_config(rebalance_frequency="quarterly")
        provider = MockReturnsProvider(_make_returns())
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        for mr in result.monthly_returns:
            assert mr.period_start is not None
            assert isinstance(mr.period_start, date)
            assert mr.period_start < mr.period_end


# --- Equity Curve helpers ---


def _make_daily_returns(
    tickers: list[str] | None = None,
    start: str = "2020-01-02",
    end: str = "2022-12-30",
    seed: int = 99,
    include_spy: bool = True,
) -> pd.DataFrame:
    """Synthetic daily returns: ~252 trading days per year."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                   "NVDA", "TSLA", "JPM", "BAC", "WMT"]
    all_cols = list(tickers) + (["SPY"] if include_spy else [])
    dates = pd.bdate_range(start, end)
    np.random.seed(seed)
    data = np.random.normal(0.0005, 0.012, size=(len(dates), len(all_cols)))
    return pd.DataFrame(data, index=dates, columns=all_cols)


def _make_zero_daily_returns(
    tickers: list[str] | None = None,
    start: str = "2020-01-02",
    end: str = "2022-12-30",
    include_spy: bool = True,
) -> pd.DataFrame:
    """Zero daily returns so values stay constant at initial_value."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                   "NVDA", "TSLA", "JPM", "BAC", "WMT"]
    all_cols = list(tickers) + (["SPY"] if include_spy else [])
    dates = pd.bdate_range(start, end)
    data = np.zeros((len(dates), len(all_cols)))
    return pd.DataFrame(data, index=dates, columns=all_cols)


# --- EquityCurvePoint model tests ---


class TestEquityCurvePoint:
    def test_basic_construction(self):
        pt = EquityCurvePoint(
            date=date(2020, 1, 31),
            portfolio_value=10000.0,
            ew_value=10000.0,
        )
        assert pt.portfolio_value == 10000.0
        assert pt.spy_value is None  # optional

    def test_with_spy(self):
        pt = EquityCurvePoint(
            date=date(2020, 1, 31),
            portfolio_value=10100.0,
            spy_value=10050.0,
            ew_value=10080.0,
        )
        assert pt.spy_value == 10050.0

    def test_immutable(self):
        pt = EquityCurvePoint(
            date=date(2020, 1, 31),
            portfolio_value=10000.0,
            ew_value=10000.0,
        )
        with pytest.raises(Exception):
            pt.portfolio_value = 999.0  # frozen model


# --- Engine equity curve tests ---


class TestEquityCurveComputation:
    def test_equity_curve_present_in_result(self):
        """result.equity_curve is non-empty after a successful backtest."""
        config = _make_config()
        monthly_df = _make_returns()
        daily_df = _make_daily_returns()
        provider = MockFrequencyReturnsProvider(monthly_df, daily_df)
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        assert hasattr(result, "equity_curve")
        assert len(result.equity_curve) > 0

    def test_equity_curve_dates_ascending(self):
        """Equity curve dates must be strictly non-decreasing."""
        config = _make_config()
        provider = MockFrequencyReturnsProvider(_make_returns(), _make_daily_returns())
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        dates = [pt.date for pt in result.equity_curve]
        assert dates == sorted(dates)

    def test_equity_curve_zero_returns_constant_value(self):
        """With zero daily returns, portfolio/EW values stay constant at initial_value."""
        config = _make_config()
        monthly_df = _make_returns()
        daily_df = _make_zero_daily_returns()
        provider = MockFrequencyReturnsProvider(monthly_df, daily_df)
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        assert len(result.equity_curve) > 0
        for pt in result.equity_curve:
            assert pt.portfolio_value == pytest.approx(10_000.0, rel=1e-6)
            assert pt.ew_value == pytest.approx(10_000.0, rel=1e-6)

    def test_equity_curve_spy_none_when_absent(self):
        """spy_value is None for every point when SPY is not in the returns provider."""
        config = _make_config()
        monthly_df = _make_returns()
        daily_df = _make_daily_returns(include_spy=False)
        provider = MockFrequencyReturnsProvider(monthly_df, daily_df)
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        assert len(result.equity_curve) > 0
        for pt in result.equity_curve:
            assert pt.spy_value is None

    def test_equity_curve_spy_populated_when_present(self):
        """spy_value is a float for every point when SPY is in the returns provider."""
        config = _make_config()
        monthly_df = _make_returns()
        daily_df = _make_daily_returns(include_spy=True)
        provider = MockFrequencyReturnsProvider(monthly_df, daily_df)
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        assert len(result.equity_curve) > 0
        for pt in result.equity_curve:
            assert pt.spy_value is not None
            assert isinstance(pt.spy_value, float)

    def test_equity_curve_zero_spy_constant(self):
        """With zero SPY returns, spy_value stays at 10000 throughout."""
        config = _make_config()
        monthly_df = _make_returns()
        daily_df = _make_zero_daily_returns(include_spy=True)
        provider = MockFrequencyReturnsProvider(monthly_df, daily_df)
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        for pt in result.equity_curve:
            assert pt.spy_value == pytest.approx(10_000.0, rel=1e-6)

    def test_equity_curve_empty_when_no_daily_data(self):
        """equity_curve is empty when the daily provider returns no data."""
        config = _make_config()
        monthly_df = _make_returns()
        # Empty daily DataFrame (no rows)
        empty_daily = pd.DataFrame(
            index=pd.DatetimeIndex([], name="date"),
            columns=list(_make_config().universe) + ["SPY"],
            dtype=float,
        )
        provider = MockFrequencyReturnsProvider(monthly_df, empty_daily)
        signals = _make_signals()

        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        assert result.equity_curve == []

    def test_equity_curve_portfolio_tracks_known_return(self):
        """With a single positive return day, portfolio value increases correctly."""
        config = _make_config(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 6, 30),
            universe=["AAPL", "MSFT"],
            rebalance_frequency="quarterly",
            num_quantiles=2,
            long_quantile=1,
            short_quantile=None,
        )
        # 1 quarterly rebalance: Q1 2020
        monthly_dates = pd.date_range("2020-01-31", periods=6, freq="ME")
        monthly_data = np.full((6, 2), 0.01)
        monthly_df = pd.DataFrame(monthly_data, index=monthly_dates, columns=["AAPL", "MSFT"])

        # Daily: one trading day with known returns
        daily_dates = pd.bdate_range("2020-01-02", "2020-06-30")
        # AAPL returns 0.01 per day, MSFT 0.02 per day
        daily_data = np.column_stack([
            np.full(len(daily_dates), 0.01),
            np.full(len(daily_dates), 0.02),
        ])
        daily_df = pd.DataFrame(daily_data, index=daily_dates, columns=["AAPL", "MSFT"])

        signals = _make_signals(
            tickers=["AAPL", "MSFT"],
            dates=[date(2020, 3, 31)],
        )
        provider = MockFrequencyReturnsProvider(monthly_df, daily_df)
        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        assert len(result.equity_curve) > 0
        # Values should be growing (positive daily returns)
        values = [pt.portfolio_value for pt in result.equity_curve]
        assert values[-1] > values[0]

    def test_equity_curve_ew_uses_equal_weights(self):
        """EW curve applies equal weights to all universe tickers."""
        config = _make_config(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 6, 30),
            universe=["AAPL", "MSFT"],
            rebalance_frequency="quarterly",
            num_quantiles=2,
            long_quantile=1,
            short_quantile=None,
        )
        monthly_dates = pd.date_range("2020-01-31", periods=6, freq="ME")
        monthly_df = pd.DataFrame(
            np.full((6, 2), 0.01),
            index=monthly_dates,
            columns=["AAPL", "MSFT"],
        )
        daily_dates = pd.bdate_range("2020-01-02", "2020-06-30")
        n = len(daily_dates)
        # AAPL = 0.0, MSFT = 0.02 → EW average = 0.01 per day
        daily_data = np.column_stack([np.zeros(n), np.full(n, 0.02)])
        daily_df = pd.DataFrame(daily_data, index=daily_dates, columns=["AAPL", "MSFT"])

        signals = _make_signals(tickers=["AAPL", "MSFT"], dates=[date(2020, 3, 31)])
        provider = MockFrequencyReturnsProvider(monthly_df, daily_df)
        engine = BacktestEngine(config, provider)
        result = engine.run(signals)

        assert len(result.equity_curve) > 0
        # EW should grow at 1% per day (average of 0% and 2%)
        # Only holding-period days are in the curve: (2020-03-31, 2020-06-30]
        n_holding = len(result.equity_curve)
        expected_ew = 10_000.0 * (1.01 ** n_holding)
        actual_ew = result.equity_curve[-1].ew_value
        assert actual_ew == pytest.approx(expected_ew, rel=1e-4)
