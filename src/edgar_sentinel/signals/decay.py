"""Exponential time decay for filing-based signals."""

from __future__ import annotations

import math
import logging
from datetime import date

from edgar_sentinel.core import Signal

logger = logging.getLogger("edgar_sentinel.signals")


class SignalDecay:
    """Applies exponential time decay to filing-based signals.

    The decay weight follows: weight = 2^(-age_days / half_life_days)

    At age 0: weight = 1.0 (full strength)
    At age = half_life: weight = 0.5
    At age = 2 * half_life: weight = 0.25
    At age = 3 * half_life: weight = 0.125

    Usage:
        decay = SignalDecay(half_life_days=90)
        decayed_signal = decay.apply(signal, as_of_date=date.today())
    """

    def __init__(self, half_life_days: int = 90):
        if half_life_days < 1:
            raise ValueError(f"half_life_days must be >= 1, got {half_life_days}")
        self._half_life = half_life_days

    @property
    def half_life_days(self) -> int:
        return self._half_life

    def compute_weight(self, signal_date: date, as_of_date: date) -> float:
        """Compute the decay weight for a signal.

        Returns decay weight in [0.0, 1.0].
        Returns 1.0 if as_of_date <= signal_date (no decay).
        Returns 0.0 if age exceeds 10 * half_life (effectively expired).
        """
        age_days = (as_of_date - signal_date).days

        if age_days <= 0:
            return 1.0

        # Cap at 10 half-lives to avoid floating point issues
        if age_days > 10 * self._half_life:
            return 0.0

        weight = math.pow(2, -age_days / self._half_life)
        return round(weight, 6)

    def apply(self, signal: Signal, as_of_date: date) -> Signal:
        """Apply decay to a signal, returning a new Signal with updated weight.

        If the signal already has a decay_weight < 1.0 (pre-decayed), the
        new weight is multiplied with the existing weight.
        """
        new_weight = self.compute_weight(signal.signal_date, as_of_date)
        final_weight = round(signal.decay_weight * new_weight, 6)

        return Signal(
            ticker=signal.ticker,
            signal_date=signal.signal_date,
            signal_name=signal.signal_name,
            raw_value=signal.raw_value,
            z_score=signal.z_score,
            percentile=signal.percentile,
            decay_weight=final_weight,
        )

    def is_expired(self, signal_date: date, as_of_date: date) -> bool:
        """Check if a signal has decayed below the usable threshold.

        A signal is considered expired when its weight falls to 0.0
        (beyond 10 half-lives).
        """
        return self.compute_weight(signal_date, as_of_date) == 0.0
