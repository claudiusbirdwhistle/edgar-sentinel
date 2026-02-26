"""Signal composite: combines multiple signals into ensemble scores per ticker."""

from __future__ import annotations

import logging
from datetime import date

from edgar_sentinel.core import (
    CompositeMethod,
    CompositeSignal,
    Signal,
    SignalName,
    Ticker,
)

logger = logging.getLogger("edgar_sentinel.signals")


class SignalComposite:
    """Combines multiple signals into a single composite score per ticker.

    Supports three weighting methods:
    - EQUAL: Simple decay-weighted average of z-scores
    - IC_WEIGHTED: Weight by trailing information coefficient
    - Custom: User-supplied weight dictionary

    Usage:
        composite = SignalComposite(method=CompositeMethod.EQUAL)
        results = composite.combine(signals, as_of_date=date(2024, 3, 15))

    For IC-weighted:
        composite = SignalComposite(
            method=CompositeMethod.IC_WEIGHTED,
            ic_values={"dictionary_mda": 0.05, "llm_mda": 0.08},
        )
    """

    def __init__(
        self,
        method: CompositeMethod = CompositeMethod.EQUAL,
        ic_values: dict[SignalName, float] | None = None,
        custom_weights: dict[SignalName, float] | None = None,
    ) -> None:
        """Initialize composite combiner.

        Args:
            method: Weighting method to use.
            ic_values: Signal-name -> trailing IC values. Required for
                IC_WEIGHTED method (falls back to EQUAL if missing).
            custom_weights: Signal-name -> weight. Normalized internally.
                Only used when method is not EQUAL or IC_WEIGHTED.

        Raises:
            ValueError: If custom_weights are provided but empty, or if
                all weights are zero.
        """
        if custom_weights is not None and len(custom_weights) == 0:
            raise ValueError("custom_weights must not be empty")

        self._method = method
        self._ic_values = ic_values
        self._custom_weights = self._normalize_weights(custom_weights)

    def combine(
        self,
        signals: list[Signal],
        as_of_date: date,
    ) -> list[CompositeSignal]:
        """Combine signals into composite scores.

        Groups signals by ticker, computes a composite score for each,
        and ranks tickers by composite score (descending: rank 1 = highest).

        Signals with decay_weight == 0.0 (expired) are excluded.

        Args:
            signals: All signals for the given as_of_date.
            as_of_date: The date of composite computation.

        Returns:
            List of CompositeSignal objects, one per ticker, sorted by
            composite_score descending. Tickers with no valid signals
            are excluded.
        """
        # Filter expired signals
        active = [s for s in signals if s.decay_weight > 0.0]

        if not active:
            logger.info("No active signals for as_of_date=%s", as_of_date)
            return []

        # Group by ticker
        by_ticker: dict[Ticker, list[Signal]] = {}
        for signal in active:
            by_ticker.setdefault(signal.ticker, []).append(signal)

        # Compute composite for each ticker
        composites: list[CompositeSignal] = []
        for ticker, ticker_signals in by_ticker.items():
            result = self._compute_composite(ticker, ticker_signals, as_of_date)
            if result is not None:
                composites.append(result)

        # Rank by composite_score descending, ties broken by ticker asc
        composites.sort(key=lambda c: (-c.composite_score, c.ticker))
        ranked = [
            CompositeSignal(
                ticker=c.ticker,
                signal_date=c.signal_date,
                composite_score=c.composite_score,
                components=c.components,
                rank=rank,
            )
            for rank, c in enumerate(composites, start=1)
        ]

        logger.info(
            "Combined %d tickers into composite signals for %s (method=%s)",
            len(ranked),
            as_of_date,
            self._method.value,
        )
        return ranked

    def _compute_composite(
        self,
        ticker: Ticker,
        signals: list[Signal],
        as_of_date: date,
    ) -> CompositeSignal | None:
        """Compute composite score for a single ticker.

        Returns None if no valid signals remain after weighting.
        """
        weights = self._get_weights(signals)

        if not weights:
            return None

        # Compute weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        components: dict[SignalName, float] = {}

        for signal in signals:
            signal_weight = weights.get(signal.signal_name, 0.0)
            if signal_weight == 0.0:
                continue

            # Use z_score if available, else raw_value
            value = signal.z_score if signal.z_score is not None else signal.raw_value

            effective_weight = signal_weight * signal.decay_weight
            weighted_sum += value * effective_weight
            total_weight += abs(effective_weight)
            components[signal.signal_name] = round(effective_weight, 6)

        if total_weight == 0.0:
            return None

        composite_score = round(weighted_sum / total_weight, 6)

        return CompositeSignal(
            ticker=ticker,
            signal_date=as_of_date,
            composite_score=composite_score,
            components=components,
        )

    def _get_weights(
        self,
        signals: list[Signal],
    ) -> dict[SignalName, float]:
        """Get the weight for each signal name based on the configured method.

        Returns:
            Dict mapping signal_name -> weight. Weights are normalized
            to sum to 1.0.
        """
        signal_names = sorted({s.signal_name for s in signals})

        if self._method == CompositeMethod.EQUAL:
            n = len(signal_names)
            return {name: 1.0 / n for name in signal_names}

        elif self._method == CompositeMethod.IC_WEIGHTED:
            if not self._ic_values:
                logger.warning(
                    "IC_WEIGHTED method requested but no IC values provided. "
                    "Falling back to EQUAL weighting."
                )
                n = len(signal_names)
                return {name: 1.0 / n for name in signal_names}

            # Weight by absolute IC value
            raw: dict[str, float] = {}
            for name in signal_names:
                ic = self._ic_values.get(name, 0.0)
                raw[name] = abs(ic)

            return self._normalize_weights(raw) or {
                name: 1.0 / len(signal_names) for name in signal_names
            }

        else:
            # Custom weights
            if self._custom_weights:
                return {
                    name: self._custom_weights.get(name, 0.0)
                    for name in signal_names
                }
            n = len(signal_names)
            return {name: 1.0 / n for name in signal_names}

    @staticmethod
    def _normalize_weights(
        weights: dict[SignalName, float] | None,
    ) -> dict[SignalName, float] | None:
        """Normalize weights to sum to 1.0.

        Returns None if weights are None or all zero.
        """
        if weights is None:
            return None

        total = sum(abs(v) for v in weights.values())
        if total == 0.0:
            return None

        return {k: round(v / total, 6) for k, v in weights.items()}
