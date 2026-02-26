"""Signal builder: converts analyzer outputs to normalized trading signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from scipy import stats

from edgar_sentinel.analyzers.base import AnalysisResults
from edgar_sentinel.core import Signal, SignalName, Ticker
from edgar_sentinel.core.config import SignalsConfig
from edgar_sentinel.core.models import SentimentResult, SimilarityResult
from edgar_sentinel.signals.decay import SignalDecay

logger = logging.getLogger("edgar_sentinel.signals")


@dataclass(frozen=True)
class FilingDateMapping:
    """Maps a filing to its signal date after buffer adjustment.

    Carries metadata needed to:
    1. Compute signal_date from filed_date + buffer_days
    2. Group signals by ticker for cross-sectional normalization
    """

    ticker: Ticker
    filing_id: str
    filed_date: date
    signal_date: date


class SignalBuilder:
    """Converts analyzer results into normalized trading signals.

    The builder operates in three phases:
    1. Extract: Pull raw values from SentimentResult/SimilarityResult
    2. Normalize: Cross-sectional z-score and percentile within cohort
    3. Decay: Apply time-based weight decay

    Usage:
        builder = SignalBuilder(config.signals)
        signals = builder.build(
            results=analysis_results,
            filing_dates=filing_date_map,
            as_of_date=date(2024, 3, 15),
        )
    """

    def __init__(
        self,
        config: SignalsConfig,
        decay: SignalDecay | None = None,
    ):
        self._buffer_days = config.buffer_days
        self._decay = decay or SignalDecay(config.decay_half_life)

    def build(
        self,
        results: AnalysisResults,
        filing_dates: dict[str, FilingDateMapping],
        as_of_date: date,
    ) -> list[Signal]:
        """Build normalized signals from analyzer results.

        Args:
            results: AnalysisResults from run_analyzers().
            filing_dates: Map of filing_id -> FilingDateMapping.
            as_of_date: The date for which signals are being computed.
                Signals from filings whose signal_date > as_of_date are excluded.

        Returns:
            List of normalized Signal objects. May be empty.
        """
        raw_signals = self._extract_raw_signals(results, filing_dates, as_of_date)

        if not raw_signals:
            logger.info("No raw signals produced for as_of_date=%s", as_of_date)
            return []

        # Group by signal_name for cross-sectional normalization
        by_name: dict[SignalName, list[Signal]] = {}
        for signal in raw_signals:
            by_name.setdefault(signal.signal_name, []).append(signal)

        normalized: list[Signal] = []
        for signal_name, group in by_name.items():
            normalized.extend(self._normalize_cross_section(group))

        # Apply decay weights
        decayed = [self._apply_decay(signal, as_of_date) for signal in normalized]

        logger.info(
            "Built %d signals for as_of_date=%s (%d signal types, %d tickers)",
            len(decayed),
            as_of_date,
            len(by_name),
            len({s.ticker for s in decayed}),
        )
        return decayed

    def _extract_raw_signals(
        self,
        results: AnalysisResults,
        filing_dates: dict[str, FilingDateMapping],
        as_of_date: date,
    ) -> list[Signal]:
        """Extract raw Signal objects from analyzer results."""
        raw: list[Signal] = []

        # Process sentiment results
        for sr in results.sentiment_results:
            mapping = filing_dates.get(sr.filing_id)
            if mapping is None:
                logger.debug("No filing date mapping for %s, skipping", sr.filing_id)
                continue

            signal_date = self._compute_signal_date(mapping.filed_date)
            if signal_date > as_of_date:
                continue

            signal_name = f"{sr.analyzer_name}_{sr.section_name}"
            raw.append(
                Signal(
                    ticker=mapping.ticker,
                    signal_date=signal_date,
                    signal_name=signal_name,
                    raw_value=sr.sentiment_score,
                )
            )

        # Process similarity results
        for sim in results.similarity_results:
            mapping = filing_dates.get(sim.filing_id)
            if mapping is None:
                logger.debug("No filing date mapping for %s, skipping", sim.filing_id)
                continue

            signal_date = self._compute_signal_date(mapping.filed_date)
            if signal_date > as_of_date:
                continue

            signal_name = f"similarity_{sim.section_name}"
            raw.append(
                Signal(
                    ticker=mapping.ticker,
                    signal_date=signal_date,
                    signal_name=signal_name,
                    raw_value=sim.change_score,
                )
            )

        return raw

    def _compute_signal_date(self, filed_date: date) -> date:
        """Compute signal date by adding buffer days to filing date."""
        return filed_date + timedelta(days=self._buffer_days)

    def _normalize_cross_section(
        self,
        signals: list[Signal],
    ) -> list[Signal]:
        """Normalize a group of signals cross-sectionally.

        All signals must have the same signal_name. Computes z_score and percentile.
        When group has < 3 signals, normalization is skipped.
        When std == 0, z_score=0.0 and percentile=50.0 for all.
        """
        if len(signals) < 3:
            logger.debug(
                "Signal '%s' has %d values — skipping normalization",
                signals[0].signal_name if signals else "unknown",
                len(signals),
            )
            return signals  # z_score/percentile remain None

        raw_values = np.array([s.raw_value for s in signals])
        mean = float(np.mean(raw_values))
        std = float(np.std(raw_values, ddof=1))  # Sample std (Bessel's correction)

        normalized: list[Signal] = []
        for signal in signals:
            if std == 0.0:
                z = 0.0
                pct = 50.0
            else:
                z = (signal.raw_value - mean) / std
                pct = float(
                    stats.percentileofscore(raw_values, signal.raw_value, kind="rank")
                )

            normalized.append(
                Signal(
                    ticker=signal.ticker,
                    signal_date=signal.signal_date,
                    signal_name=signal.signal_name,
                    raw_value=signal.raw_value,
                    z_score=round(z, 6),
                    percentile=round(pct, 2),
                    decay_weight=signal.decay_weight,
                )
            )

        return normalized

    def _apply_decay(self, signal: Signal, as_of_date: date) -> Signal:
        """Apply time decay to a signal."""
        return self._decay.apply(signal, as_of_date)
