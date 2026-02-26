"""Tests for edgar_sentinel.signals.decay — SignalDecay."""

import math
from datetime import date

import pytest

from edgar_sentinel.core import Signal
from edgar_sentinel.signals.decay import SignalDecay


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_half_life(self):
        decay = SignalDecay()
        assert decay.half_life_days == 90

    def test_custom_half_life(self):
        decay = SignalDecay(half_life_days=30)
        assert decay.half_life_days == 30

    def test_minimum_half_life(self):
        decay = SignalDecay(half_life_days=1)
        assert decay.half_life_days == 1

    def test_zero_half_life_rejected(self):
        with pytest.raises(ValueError, match="half_life_days must be >= 1"):
            SignalDecay(half_life_days=0)

    def test_negative_half_life_rejected(self):
        with pytest.raises(ValueError, match="half_life_days must be >= 1"):
            SignalDecay(half_life_days=-10)


# ---------------------------------------------------------------------------
# compute_weight
# ---------------------------------------------------------------------------


class TestComputeWeight:
    def test_same_day_returns_one(self):
        decay = SignalDecay(half_life_days=90)
        w = decay.compute_weight(date(2024, 1, 1), date(2024, 1, 1))
        assert w == 1.0

    def test_future_signal_returns_one(self):
        decay = SignalDecay(half_life_days=90)
        w = decay.compute_weight(date(2024, 3, 1), date(2024, 1, 1))
        assert w == 1.0

    def test_half_life_returns_half(self):
        decay = SignalDecay(half_life_days=90)
        w = decay.compute_weight(date(2024, 1, 1), date(2024, 3, 31))
        assert w == 0.5

    def test_double_half_life_returns_quarter(self):
        decay = SignalDecay(half_life_days=90)
        w = decay.compute_weight(date(2024, 1, 1), date(2024, 6, 29))
        assert w == 0.25

    def test_one_day_with_half_life_one(self):
        decay = SignalDecay(half_life_days=1)
        w = decay.compute_weight(date(2024, 1, 1), date(2024, 1, 2))
        assert w == 0.5

    def test_expired_at_ten_half_lives(self):
        decay = SignalDecay(half_life_days=90)
        # 901 days > 10 * 90 = 900
        w = decay.compute_weight(date(2024, 1, 1), date(2026, 6, 20))
        assert w == 0.0

    def test_just_at_ten_half_lives_not_expired(self):
        decay = SignalDecay(half_life_days=90)
        # Exactly 900 days = 10 * 90 (not > 900, so still computes)
        w = decay.compute_weight(date(2024, 1, 1), date(2026, 6, 19))
        assert w > 0.0

    def test_30_day_weight(self):
        decay = SignalDecay(half_life_days=90)
        w = decay.compute_weight(date(2024, 1, 1), date(2024, 1, 31))
        expected = round(math.pow(2, -30 / 90), 6)
        assert w == expected

    def test_weight_rounded_to_six_decimals(self):
        decay = SignalDecay(half_life_days=90)
        w = decay.compute_weight(date(2024, 1, 1), date(2024, 2, 15))
        # 45 days
        expected = round(math.pow(2, -45 / 90), 6)
        assert w == expected

    def test_large_half_life(self):
        decay = SignalDecay(half_life_days=365)
        w = decay.compute_weight(date(2024, 1, 1), date(2025, 1, 1))
        # 366 days (2024 is leap year)
        expected = round(math.pow(2, -366 / 365), 6)
        assert w == expected


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def _make_signal(
    signal_date: date = date(2024, 1, 3),
    decay_weight: float = 1.0,
    z_score: float | None = 0.5,
    percentile: float | None = 75.0,
) -> Signal:
    return Signal(
        ticker="AAPL",
        signal_date=signal_date,
        signal_name="dictionary_mda",
        raw_value=0.15,
        z_score=z_score,
        percentile=percentile,
        decay_weight=decay_weight,
    )


class TestApply:
    def test_apply_preserves_fields(self):
        decay = SignalDecay(half_life_days=90)
        signal = _make_signal()
        result = decay.apply(signal, as_of_date=date(2024, 4, 2))
        assert result.ticker == "AAPL"
        assert result.signal_name == "dictionary_mda"
        assert result.raw_value == 0.15
        assert result.z_score == 0.5
        assert result.percentile == 75.0
        assert result.signal_date == date(2024, 1, 3)

    def test_apply_same_day_keeps_weight(self):
        decay = SignalDecay(half_life_days=90)
        signal = _make_signal()
        result = decay.apply(signal, as_of_date=date(2024, 1, 3))
        assert result.decay_weight == 1.0

    def test_apply_at_half_life(self):
        decay = SignalDecay(half_life_days=90)
        signal = _make_signal(signal_date=date(2024, 1, 1))
        result = decay.apply(signal, as_of_date=date(2024, 3, 31))
        assert result.decay_weight == 0.5

    def test_apply_composes_with_existing_weight(self):
        decay = SignalDecay(half_life_days=90)
        signal = _make_signal(signal_date=date(2024, 1, 1), decay_weight=0.8)
        result = decay.apply(signal, as_of_date=date(2024, 3, 31))
        # 0.8 * 0.5 = 0.4
        assert result.decay_weight == 0.4

    def test_apply_returns_frozen_signal(self):
        decay = SignalDecay(half_life_days=90)
        signal = _make_signal()
        result = decay.apply(signal, as_of_date=date(2024, 4, 2))
        with pytest.raises(Exception):
            result.decay_weight = 0.0  # type: ignore[misc]

    def test_apply_with_none_z_score(self):
        decay = SignalDecay(half_life_days=90)
        signal = _make_signal(z_score=None, percentile=None)
        result = decay.apply(signal, as_of_date=date(2024, 4, 2))
        assert result.z_score is None
        assert result.percentile is None


# ---------------------------------------------------------------------------
# is_expired
# ---------------------------------------------------------------------------


class TestIsExpired:
    def test_fresh_signal_not_expired(self):
        decay = SignalDecay(half_life_days=90)
        assert not decay.is_expired(date(2024, 1, 1), date(2024, 1, 1))

    def test_old_signal_expired(self):
        decay = SignalDecay(half_life_days=90)
        assert decay.is_expired(date(2024, 1, 1), date(2026, 7, 1))

    def test_at_boundary_not_expired(self):
        decay = SignalDecay(half_life_days=90)
        # Exactly 900 days (10 * 90) — not > 900
        assert not decay.is_expired(date(2024, 1, 1), date(2026, 6, 19))

    def test_just_beyond_boundary_expired(self):
        decay = SignalDecay(half_life_days=90)
        # 901 days > 900
        assert decay.is_expired(date(2024, 1, 1), date(2026, 6, 20))

    def test_future_signal_not_expired(self):
        decay = SignalDecay(half_life_days=90)
        assert not decay.is_expired(date(2025, 1, 1), date(2024, 1, 1))
