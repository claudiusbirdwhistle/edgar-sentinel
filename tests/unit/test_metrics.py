"""Tests for backtest.metrics: MetricsCalculator."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from edgar_sentinel.backtest.metrics import MetricsCalculator
from edgar_sentinel.core.models import CompositeSignal


# --- Fixtures ---


def _monthly_index(n: int = 24) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-31", periods=n, freq="ME")


def _constant_returns(value: float, n: int = 24) -> pd.Series:
    return pd.Series([value] * n, index=_monthly_index(n))


def _make_signals(
    tickers: list[str],
    sig_dates: list[date],
    scores_fn=None,
) -> list[CompositeSignal]:
    signals = []
    for sig_date in sig_dates:
        for i, ticker in enumerate(tickers):
            score = scores_fn(i, sig_date) if scores_fn else float(i) / len(tickers)
            signals.append(
                CompositeSignal(
                    ticker=ticker,
                    signal_date=sig_date,
                    composite_score=score,
                    components={"a": 1.0},
                )
            )
    return signals


# --- Sharpe Ratio ---


class TestSharpeRatio:
    def test_zero_returns(self):
        calc = MetricsCalculator()
        returns = _constant_returns(0.0)
        assert calc.sharpe_ratio(returns) == 0.0

    def test_positive_constant_returns(self):
        calc = MetricsCalculator()
        returns = _constant_returns(0.01)
        # Constant returns → 0 std → 0 sharpe
        assert calc.sharpe_ratio(returns) == 0.0

    def test_positive_mean_returns(self):
        calc = MetricsCalculator()
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.02, 0.03, 60), index=_monthly_index(60))
        sharpe = calc.sharpe_ratio(returns)
        # With positive mean and moderate vol, sharpe should be positive
        assert sharpe > 0

    def test_negative_mean_returns(self):
        calc = MetricsCalculator()
        np.random.seed(42)
        returns = pd.Series(np.random.normal(-0.02, 0.03, 60), index=_monthly_index(60))
        sharpe = calc.sharpe_ratio(returns)
        assert sharpe < 0

    def test_single_return(self):
        calc = MetricsCalculator()
        returns = pd.Series([0.05], index=_monthly_index(1))
        assert calc.sharpe_ratio(returns) == 0.0  # Need at least 2

    def test_with_risk_free_rate(self):
        calc = MetricsCalculator(risk_free_rate=0.05)
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.03, 60), index=_monthly_index(60))
        sharpe_rf = calc.sharpe_ratio(returns)
        # Higher risk-free rate → lower Sharpe
        calc_zero = MetricsCalculator(risk_free_rate=0.0)
        sharpe_zero = calc_zero.sharpe_ratio(returns)
        assert sharpe_rf < sharpe_zero

    def test_annualization(self):
        calc = MetricsCalculator()
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.05, 60), index=_monthly_index(60))
        sharpe = calc.sharpe_ratio(returns)
        # Manual calculation
        excess = returns - 0.0
        expected = float(excess.mean() / excess.std() * np.sqrt(12))
        assert sharpe == pytest.approx(expected, rel=1e-6)


# --- Sortino Ratio ---


class TestSortinoRatio:
    def test_all_positive_returns(self):
        calc = MetricsCalculator()
        returns = pd.Series([0.01, 0.02, 0.03, 0.01], index=_monthly_index(4))
        sortino = calc.sortino_ratio(returns)
        assert sortino == float("inf")

    def test_mixed_returns(self):
        calc = MetricsCalculator()
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.05, 60), index=_monthly_index(60))
        sortino = calc.sortino_ratio(returns)
        assert isinstance(sortino, float)
        assert sortino != 0.0

    def test_all_negative_returns(self):
        calc = MetricsCalculator()
        returns = pd.Series([-0.01, -0.02, -0.03, -0.01], index=_monthly_index(4))
        sortino = calc.sortino_ratio(returns)
        assert sortino < 0

    def test_single_return(self):
        calc = MetricsCalculator()
        returns = pd.Series([0.05], index=_monthly_index(1))
        assert calc.sortino_ratio(returns) == 0.0


# --- Max Drawdown ---


class TestMaxDrawdown:
    def test_no_drawdown(self):
        calc = MetricsCalculator()
        returns = pd.Series([0.01, 0.02, 0.03, 0.01], index=_monthly_index(4))
        result = calc.max_drawdown(returns)
        assert result["max_drawdown"] == pytest.approx(0.0)

    def test_known_drawdown(self):
        calc = MetricsCalculator()
        # Up 10%, down 20%, up 5%
        returns = pd.Series([0.10, -0.20, 0.05], index=_monthly_index(3))
        result = calc.max_drawdown(returns)
        # After +10%: cum=1.10, After -20%: cum=0.88, drawdown from peak 1.10 = -0.20
        assert result["max_drawdown"] < 0
        assert result["max_drawdown"] == pytest.approx(-0.20, abs=0.01)

    def test_duration(self):
        calc = MetricsCalculator()
        returns = pd.Series([0.10, -0.20, 0.05], index=_monthly_index(3))
        result = calc.max_drawdown(returns)
        assert result["duration"] >= 1

    def test_empty_returns(self):
        calc = MetricsCalculator()
        returns = pd.Series([], dtype=float)
        result = calc.max_drawdown(returns)
        assert result["max_drawdown"] == 0.0
        assert result["duration"] == 0

    def test_single_negative_return(self):
        calc = MetricsCalculator()
        returns = pd.Series([-0.10], index=_monthly_index(1))
        result = calc.max_drawdown(returns)
        # Single observation: cumulative=0.9, running_max=0.9, drawdown=0
        # No prior peak to draw down from
        assert result["max_drawdown"] == pytest.approx(0.0)

    def test_peak_and_trough_dates(self):
        calc = MetricsCalculator()
        returns = pd.Series([0.10, 0.05, -0.30, -0.10, 0.20], index=_monthly_index(5))
        result = calc.max_drawdown(returns)
        assert result["peak_date"] is not None
        assert result["trough_date"] is not None
        assert result["peak_date"] <= result["trough_date"]


# --- Annualized Volatility ---


class TestAnnualizedVolatility:
    def test_constant_returns(self):
        calc = MetricsCalculator()
        returns = _constant_returns(0.01)
        assert calc.annualized_volatility(returns) == pytest.approx(0.0, abs=1e-10)

    def test_volatile_returns(self):
        calc = MetricsCalculator()
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.05, 60), index=_monthly_index(60))
        vol = calc.annualized_volatility(returns)
        # Should be roughly 0.05 * sqrt(12) ≈ 0.17
        assert 0.10 < vol < 0.25

    def test_single_return(self):
        calc = MetricsCalculator()
        returns = pd.Series([0.05], index=_monthly_index(1))
        assert calc.annualized_volatility(returns) == 0.0


# --- Win Rate ---


class TestWinRate:
    def test_all_positive(self):
        calc = MetricsCalculator()
        returns = pd.Series([0.01, 0.02, 0.03], index=_monthly_index(3))
        assert calc.win_rate(returns) == pytest.approx(1.0)

    def test_all_negative(self):
        calc = MetricsCalculator()
        returns = pd.Series([-0.01, -0.02, -0.03], index=_monthly_index(3))
        assert calc.win_rate(returns) == pytest.approx(0.0)

    def test_mixed(self):
        calc = MetricsCalculator()
        returns = pd.Series([0.01, -0.02, 0.03, -0.01], index=_monthly_index(4))
        assert calc.win_rate(returns) == pytest.approx(0.5)

    def test_empty(self):
        calc = MetricsCalculator()
        returns = pd.Series([], dtype=float)
        assert calc.win_rate(returns) == 0.0


# --- Information Coefficient ---


class TestInformationCoefficient:
    def test_perfect_correlation(self):
        calc = MetricsCalculator()
        tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        # Signals: scores = rank order
        signals = _make_signals(
            tickers,
            [date(2020, 3, 31), date(2020, 6, 30)],
            scores_fn=lambda i, d: float(i),
        )

        # Returns: perfectly correlated with signal scores
        dates_idx = pd.date_range("2020-04-30", periods=3, freq="ME")
        data = {}
        for i, t in enumerate(tickers):
            data[t] = [0.01 * (i + 1)] * 3  # Higher-scored tickers get higher returns
        returns_df = pd.DataFrame(data, index=dates_idx)

        result = calc.information_coefficient(signals, returns_df)
        assert result["mean_ic"] is not None
        assert result["mean_ic"] > 0.5  # Should be high

    def test_no_signals(self):
        calc = MetricsCalculator()
        returns_df = pd.DataFrame(
            {"A": [0.01, 0.02]},
            index=pd.date_range("2020-01-31", periods=2, freq="ME"),
        )
        result = calc.information_coefficient([], returns_df)
        assert result["mean_ic"] is None

    def test_single_date(self):
        calc = MetricsCalculator()
        tickers = ["A", "B", "C", "D", "E"]
        signals = _make_signals(tickers, [date(2020, 3, 31)])
        returns_df = pd.DataFrame(
            {t: [0.01] for t in tickers},
            index=pd.date_range("2020-04-30", periods=1, freq="ME"),
        )
        result = calc.information_coefficient(signals, returns_df)
        # Single date → no forward return pair → None
        assert result["mean_ic"] is None

    def test_t_statistic(self):
        calc = MetricsCalculator()
        tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        sig_dates = [date(2020, m, 28) for m in range(1, 12)]
        signals = _make_signals(
            tickers, sig_dates,
            scores_fn=lambda i, d: float(i) + np.random.normal(0, 0.1),
        )

        np.random.seed(42)
        dates_idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        data = {t: np.random.normal(0.01, 0.05, 12) for t in tickers}
        returns_df = pd.DataFrame(data, index=dates_idx)

        result = calc.information_coefficient(signals, returns_df)
        if result["mean_ic"] is not None:
            assert isinstance(result["t_statistic"], float)
            assert isinstance(result["icir"], float)


# --- Factor Regression ---


class TestFactorRegression:
    def test_with_synthetic_factors(self):
        calc = MetricsCalculator()
        n = 24
        idx = _monthly_index(n)

        np.random.seed(42)
        mkt = np.random.normal(0.01, 0.04, n)
        smb = np.random.normal(0.002, 0.02, n)
        hml = np.random.normal(0.001, 0.02, n)

        # Portfolio returns = alpha + 1.0*Mkt + 0.5*SMB + noise
        alpha = 0.005
        portfolio = alpha + 1.0 * mkt + 0.5 * smb + np.random.normal(0, 0.01, n)

        returns = pd.Series(portfolio, index=idx)
        factor_data = pd.DataFrame(
            {"Mkt-RF": mkt, "SMB": smb, "HML": hml, "RF": [0.0] * n},
            index=idx,
        )

        result = calc.factor_regression(returns, factor_data=factor_data)

        assert isinstance(result["alpha"], float)
        assert isinstance(result["alpha_t_stat"], float)
        assert "Mkt-RF" in result["loadings"]
        assert "SMB" in result["loadings"]
        assert isinstance(result["r_squared"], float)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_too_few_months_raises(self):
        calc = MetricsCalculator()
        idx = _monthly_index(6)  # Only 6 months, need 12
        returns = pd.Series(np.random.normal(0.01, 0.05, 6), index=idx)
        factor_data = pd.DataFrame(
            {"Mkt-RF": np.random.normal(0.01, 0.04, 6), "RF": [0.0] * 6},
            index=idx,
        )

        with pytest.raises(ValueError, match="at least 12"):
            calc.factor_regression(returns, factor_data=factor_data)

    def test_empty_factor_data_raises(self):
        calc = MetricsCalculator()
        returns = pd.Series([0.01] * 24, index=_monthly_index(24))

        with pytest.raises(ValueError, match="No factor data"):
            calc.factor_regression(returns, factor_data=pd.DataFrame())

    def test_alpha_positive_for_positive_excess(self):
        calc = MetricsCalculator()
        n = 36
        idx = _monthly_index(n)
        np.random.seed(42)

        # Market goes nowhere, portfolio has strong alpha
        mkt = np.random.normal(0.0, 0.04, n)
        portfolio = 0.01 + 0.5 * mkt + np.random.normal(0, 0.005, n)

        returns = pd.Series(portfolio, index=idx)
        factor_data = pd.DataFrame(
            {"Mkt-RF": mkt, "RF": [0.0] * n},
            index=idx,
        )

        result = calc.factor_regression(returns, factor_data=factor_data)
        assert result["alpha"] > 0


# --- compute_all ---


class TestComputeAll:
    def test_basic(self):
        calc = MetricsCalculator()
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.05, 36), index=_monthly_index(36))
        result = calc.compute_all(returns)

        assert "sharpe_ratio" in result
        assert "sortino_ratio" in result
        assert "max_drawdown" in result
        assert "win_rate" in result
        assert "best_month" in result
        assert "worst_month" in result
        assert "annualized_volatility" in result

    def test_with_signals(self):
        calc = MetricsCalculator()
        tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        signals = _make_signals(
            tickers,
            [date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30)],
        )

        np.random.seed(42)
        dates_idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        returns_data = {t: np.random.normal(0.01, 0.05, 12) for t in tickers}
        universe_returns = pd.DataFrame(returns_data, index=dates_idx)

        portfolio_returns = pd.Series(np.random.normal(0.01, 0.05, 12), index=dates_idx)

        result = calc.compute_all(portfolio_returns, signals, universe_returns)
        # IC fields should be populated (or None if not enough data)
        assert "information_coefficient" in result
        assert "ic_t_statistic" in result

    def test_without_signals(self):
        calc = MetricsCalculator()
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.05, 36), index=_monthly_index(36))
        result = calc.compute_all(returns)

        assert result["information_coefficient"] is None
        assert result["information_ratio"] is None


# --- Module imports ---


class TestModuleImports:
    def test_package_exports(self):
        from edgar_sentinel.backtest import (
            BacktestEngine,
            MetricsCalculator,
            PortfolioConstructor,
            PortfolioHistory,
            PortfolioSnapshot,
            Position,
            generate_rebalance_dates,
            run_backtest,
        )

        assert BacktestEngine is not None
        assert MetricsCalculator is not None
        assert PortfolioConstructor is not None
        assert run_backtest is not None
