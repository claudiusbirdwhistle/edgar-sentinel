"""Performance metrics: Sharpe, Sortino, drawdown, IC, Fama-French regression."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Computes performance metrics for backtest return series.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate for Sharpe/Sortino computation.
        Default: 0.0 (standard for long-short factor portfolios).
    annualization_factor : int
        Number of return periods per year. Default: 12 (monthly).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 12,
    ) -> None:
        self._rf = risk_free_rate
        self._af = annualization_factor

    def compute_all(
        self,
        returns: pd.Series,
        signals: list | None = None,
        universe_returns: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Compute all metrics and return as a dictionary."""
        result: dict[str, Any] = {}

        # Return metrics
        result["sharpe_ratio"] = self.sharpe_ratio(returns)
        result["sortino_ratio"] = self.sortino_ratio(returns)
        result["annualized_volatility"] = self.annualized_volatility(returns)
        result["win_rate"] = self.win_rate(returns)
        result["best_month"] = float(returns.max()) if len(returns) > 0 else 0.0
        result["worst_month"] = float(returns.min()) if len(returns) > 0 else 0.0

        # Risk metrics
        dd_result = self.max_drawdown(returns)
        result["max_drawdown"] = dd_result["max_drawdown"]
        result["max_drawdown_duration"] = dd_result["duration"]

        # IC metrics
        if signals is not None and universe_returns is not None:
            ic_result = self.information_coefficient(signals, universe_returns)
            result["information_coefficient"] = ic_result["mean_ic"]
            result["ic_t_statistic"] = ic_result["t_statistic"]
            result["information_ratio"] = ic_result["icir"]
        else:
            result["information_coefficient"] = None
            result["ic_t_statistic"] = None
            result["information_ratio"] = None

        # Factor metrics (skip if statsmodels not available)
        try:
            factor_result = self.factor_regression(returns)
            result["factor_exposures"] = factor_result["loadings"]
            result["alpha"] = factor_result["alpha"]
            result["alpha_t_stat"] = factor_result["alpha_t_stat"]
        except Exception as e:
            logger.warning(f"Factor regression failed: {e}")
            result["factor_exposures"] = None
            result["alpha"] = None
            result["alpha_t_stat"] = None

        return result

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Annualized Sharpe ratio.

        Sharpe = (mean(r) - rf_monthly) / std(r) * sqrt(annualization_factor)
        """
        if len(returns) < 2:
            return 0.0

        rf_period = (1 + self._rf) ** (1 / self._af) - 1
        excess = returns - rf_period

        if excess.std() == 0:
            return 0.0

        return float(excess.mean() / excess.std() * np.sqrt(self._af))

    def sortino_ratio(self, returns: pd.Series) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0

        rf_period = (1 + self._rf) ** (1 / self._af) - 1
        excess = returns - rf_period
        downside = excess[excess < 0]

        if len(downside) == 0 or downside.std() == 0:
            return float("inf") if excess.mean() > 0 else 0.0

        return float(excess.mean() / downside.std() * np.sqrt(self._af))

    def max_drawdown(self, returns: pd.Series) -> dict[str, Any]:
        """Maximum drawdown and its duration.

        Returns dict with max_drawdown (negative), peak_date, trough_date, duration.
        """
        if len(returns) == 0:
            return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None, "duration": 0}

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1

        max_dd = float(drawdown.min())
        trough_idx = drawdown.idxmin()

        # Peak is the last point where cumulative was at its max before trough
        peak_mask = (cumulative.index <= trough_idx) & (cumulative == running_max)
        peak_candidates = cumulative[peak_mask]
        peak_idx = peak_candidates.index[-1] if len(peak_candidates) > 0 else trough_idx

        duration = len(cumulative.loc[peak_idx:trough_idx]) - 1

        return {
            "max_drawdown": max_dd,
            "peak_date": peak_idx.date() if hasattr(peak_idx, "date") else None,
            "trough_date": trough_idx.date() if hasattr(trough_idx, "date") else None,
            "duration": duration,
        }

    def annualized_volatility(self, returns: pd.Series) -> float:
        """Annualized volatility of the return series."""
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(self._af))

    def win_rate(self, returns: pd.Series) -> float:
        """Fraction of periods with positive returns."""
        if len(returns) == 0:
            return 0.0
        return float((returns > 0).sum() / len(returns))

    def information_coefficient(
        self,
        signals: list,
        universe_returns: pd.DataFrame,
    ) -> dict[str, float | None]:
        """Information coefficient: rank correlation between signal and forward returns.

        At each signal date:
        1. Take signal scores for all tickers.
        2. Take forward returns (signal_date to next signal_date).
        3. Compute Spearman rank correlation.
        4. Average across all dates.
        """
        from scipy import stats as scipy_stats

        # Group signals by date
        signals_by_date: dict[date, list] = {}
        for sig in signals:
            signals_by_date.setdefault(sig.signal_date, []).append(sig)

        ic_values: list[float] = []

        sorted_dates = sorted(signals_by_date.keys())
        for i, sig_date in enumerate(sorted_dates):
            # Forward return period: sig_date to next sig_date (or end)
            if i + 1 < len(sorted_dates):
                fwd_end = sorted_dates[i + 1]
            else:
                continue  # No forward return for last date

            date_signals = signals_by_date[sig_date]

            # Build signal vector and forward return vector
            sig_scores: list[float] = []
            fwd_returns: list[float] = []

            for sig in date_signals:
                if sig.ticker not in universe_returns.columns:
                    continue

                # Forward return for this ticker
                mask = (universe_returns.index.date > sig_date) & (
                    universe_returns.index.date <= fwd_end
                )
                rets = universe_returns.loc[mask, sig.ticker].dropna()
                if rets.empty:
                    continue

                fwd_ret = float(np.prod(1 + rets) - 1)
                sig_scores.append(sig.composite_score)
                fwd_returns.append(fwd_ret)

            if len(sig_scores) >= 5:  # Minimum for meaningful rank correlation
                rho, _ = scipy_stats.spearmanr(sig_scores, fwd_returns)
                if not np.isnan(rho):
                    ic_values.append(rho)

        if not ic_values:
            return {"mean_ic": None, "ic_std": None, "t_statistic": None, "icir": None}

        mean_ic = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values, ddof=1))
        n = len(ic_values)

        t_stat = mean_ic / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
        icir = mean_ic / ic_std if ic_std > 0 else 0.0

        return {
            "mean_ic": mean_ic,
            "ic_std": ic_std,
            "t_statistic": float(t_stat),
            "icir": float(icir),
        }

    def factor_regression(
        self,
        returns: pd.Series,
        factor_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Fama-French 5-factor + momentum regression.

        Runs OLS: r_portfolio - r_f = alpha + b1*MktRF + b2*SMB + b3*HML
                                       + b4*RMW + b5*CMA + b6*UMD + epsilon
        """
        import statsmodels.api as sm

        if factor_data is None:
            factor_data = self._fetch_french_factors(returns.index)

        if factor_data.empty:
            raise ValueError("No factor data available for regression")

        # Align dates
        aligned = pd.concat([returns.rename("portfolio"), factor_data], axis=1).dropna()

        if len(aligned) < 12:
            raise ValueError(
                f"Only {len(aligned)} overlapping months — need at least 12"
            )

        y = aligned["portfolio"] - aligned.get("RF", 0)

        factor_cols = [
            c
            for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
            if c in aligned.columns
        ]
        X = sm.add_constant(aligned[factor_cols])

        model = sm.OLS(y, X).fit()

        loadings = {col: float(model.params[col]) for col in factor_cols}

        return {
            "alpha": float(model.params["const"]),
            "alpha_t_stat": float(model.tvalues["const"]),
            "loadings": loadings,
            "r_squared": float(model.rsquared),
            "residual_std": float(model.resid.std()),
        }

    def _fetch_french_factors(self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Download Fama-French 5-factor + momentum data.

        Data cached locally after first download.
        Returns DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF, UMD
        """
        try:
            import pandas_datareader.data as web

            ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench")[0]
            mom = web.DataReader("F-F_Momentum_Factor", "famafrench")[0]

            # Convert from percentage to decimal
            ff5 = ff5 / 100.0
            mom = mom / 100.0

            factors = pd.concat([ff5, mom], axis=1)
            factors.index = pd.to_datetime(factors.index.to_timestamp())
            return factors

        except Exception as e:
            logger.warning(f"Failed to fetch French factors: {e}")
            return pd.DataFrame()
