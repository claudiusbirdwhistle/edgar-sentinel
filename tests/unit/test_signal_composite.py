"""Tests for SignalComposite — signal ensemble combiner."""

from __future__ import annotations

import logging
from datetime import date

import pytest

from edgar_sentinel.core import CompositeMethod, CompositeSignal, Signal, SignalName
from edgar_sentinel.signals.composite import SignalComposite


# --- Helpers ---


def _signal(
    ticker: str = "AAPL",
    signal_name: str = "dictionary_mda",
    raw_value: float = 0.5,
    z_score: float | None = 1.0,
    percentile: float | None = 75.0,
    decay_weight: float = 1.0,
    signal_date: date | None = None,
) -> Signal:
    return Signal(
        ticker=ticker,
        signal_date=signal_date or date(2024, 3, 15),
        signal_name=signal_name,
        raw_value=raw_value,
        z_score=z_score,
        percentile=percentile,
        decay_weight=decay_weight,
    )


AS_OF = date(2024, 3, 15)


# === Construction Tests ===


class TestConstruction:
    def test_default_equal(self) -> None:
        c = SignalComposite()
        assert c._method == CompositeMethod.EQUAL

    def test_ic_weighted(self) -> None:
        c = SignalComposite(
            method=CompositeMethod.IC_WEIGHTED,
            ic_values={"dictionary_mda": 0.05},
        )
        assert c._method == CompositeMethod.IC_WEIGHTED
        assert c._ic_values == {"dictionary_mda": 0.05}

    def test_custom_weights(self) -> None:
        c = SignalComposite(
            custom_weights={"dictionary_mda": 0.6, "llm_mda": 0.4},
        )
        assert c._custom_weights is not None
        assert abs(sum(c._custom_weights.values()) - 1.0) < 1e-6

    def test_empty_custom_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            SignalComposite(custom_weights={})

    def test_all_zero_custom_weights(self) -> None:
        c = SignalComposite(custom_weights={"a": 0.0, "b": 0.0})
        # Normalized to None (all-zero)
        assert c._custom_weights is None


# === Equal Weight Tests ===


class TestEqualWeight:
    def test_single_signal_single_ticker(self) -> None:
        signals = [_signal(ticker="AAPL", z_score=1.5)]
        c = SignalComposite(method=CompositeMethod.EQUAL)
        result = c.combine(signals, AS_OF)

        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        assert result[0].composite_score == 1.5
        assert result[0].rank == 1

    def test_multiple_signals_same_ticker(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0),
        ]
        c = SignalComposite(method=CompositeMethod.EQUAL)
        result = c.combine(signals, AS_OF)

        assert len(result) == 1
        # Equal weight: (1.0 * 0.5 * 1.0 + 2.0 * 0.5 * 1.0) / (0.5 + 0.5) = 1.5
        assert result[0].composite_score == 1.5

    def test_multiple_tickers_ranking(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=2.0),
            _signal(ticker="MSFT", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="GOOG", signal_name="dictionary_mda", z_score=3.0),
        ]
        c = SignalComposite(method=CompositeMethod.EQUAL)
        result = c.combine(signals, AS_OF)

        assert len(result) == 3
        assert result[0].ticker == "GOOG"
        assert result[0].rank == 1
        assert result[1].ticker == "AAPL"
        assert result[1].rank == 2
        assert result[2].ticker == "MSFT"
        assert result[2].rank == 3

    def test_decay_weighted_mean(self) -> None:
        signals = [
            _signal(
                ticker="AAPL",
                signal_name="dictionary_mda",
                z_score=2.0,
                decay_weight=1.0,
            ),
            _signal(
                ticker="AAPL",
                signal_name="llm_mda",
                z_score=0.0,
                decay_weight=0.5,
            ),
        ]
        c = SignalComposite(method=CompositeMethod.EQUAL)
        result = c.combine(signals, AS_OF)

        # dictionary_mda: weight=0.5 * decay=1.0 = 0.5, value=2.0
        # llm_mda: weight=0.5 * decay=0.5 = 0.25, value=0.0
        # weighted_sum = 2.0*0.5 + 0.0*0.25 = 1.0
        # total_weight = 0.5 + 0.25 = 0.75
        # score = 1.0 / 0.75 = 1.333...
        assert abs(result[0].composite_score - 1.333333) < 1e-5

    def test_z_score_none_falls_back_to_raw(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=None, raw_value=0.3),
        ]
        c = SignalComposite(method=CompositeMethod.EQUAL)
        result = c.combine(signals, AS_OF)

        assert result[0].composite_score == 0.3

    def test_mixed_z_score_and_none(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=None, raw_value=0.5),
        ]
        c = SignalComposite(method=CompositeMethod.EQUAL)
        result = c.combine(signals, AS_OF)

        # dictionary_mda: ew=0.5*1.0=0.5, value=1.0
        # llm_mda: ew=0.5*1.0=0.5, value=0.5 (raw)
        # score = (1.0*0.5 + 0.5*0.5) / (0.5+0.5) = 0.75
        assert result[0].composite_score == 0.75


# === IC-Weighted Tests ===


class TestICWeighted:
    def test_basic_ic_weighting(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0),
        ]
        ic_values = {"dictionary_mda": 0.04, "llm_mda": 0.08}
        c = SignalComposite(method=CompositeMethod.IC_WEIGHTED, ic_values=ic_values)
        result = c.combine(signals, AS_OF)

        # IC weights normalized: dict=0.04/0.12=0.333, llm=0.08/0.12=0.667
        # ew dict: 0.333333 * 1.0 = 0.333333, ew llm: 0.666667 * 1.0 = 0.666667
        # score = (1.0*0.333333 + 2.0*0.666667) / (0.333333 + 0.666667)
        # = (0.333333 + 1.333334) / 1.0 = 1.666667
        assert abs(result[0].composite_score - 1.666667) < 1e-4

    def test_ic_fallback_to_equal_when_no_ics(self, caplog: pytest.LogCaptureFixture) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0),
        ]
        c = SignalComposite(method=CompositeMethod.IC_WEIGHTED)

        with caplog.at_level(logging.WARNING, logger="edgar_sentinel.signals"):
            result = c.combine(signals, AS_OF)

        assert "Falling back to EQUAL" in caplog.text
        assert result[0].composite_score == 1.5  # Equal-weight average

    def test_ic_unknown_signal_gets_zero(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0),
        ]
        # Only provide IC for one signal — the other gets 0
        ic_values = {"dictionary_mda": 0.05}
        c = SignalComposite(method=CompositeMethod.IC_WEIGHTED, ic_values=ic_values)
        result = c.combine(signals, AS_OF)

        # dictionary_mda: ic=0.05/0.05=1.0, llm_mda: ic=0.0/0.05=0.0
        # Only dictionary_mda contributes
        # score = 1.0 * 1.0 * 1.0 / 1.0 = 1.0
        assert result[0].composite_score == 1.0

    def test_ic_all_zero_falls_back(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
        ]
        ic_values = {"dictionary_mda": 0.0}
        c = SignalComposite(method=CompositeMethod.IC_WEIGHTED, ic_values=ic_values)
        result = c.combine(signals, AS_OF)

        # All ICs zero → fallback to equal weight
        assert result[0].composite_score == 1.0

    def test_negative_ic_uses_absolute_value(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0),
        ]
        # Negative IC means signal is contrarian — abs value used for weight
        ic_values = {"dictionary_mda": -0.04, "llm_mda": 0.08}
        c = SignalComposite(method=CompositeMethod.IC_WEIGHTED, ic_values=ic_values)
        result = c.combine(signals, AS_OF)

        # Same abs weights as basic test: 0.333, 0.667
        assert abs(result[0].composite_score - 1.666667) < 1e-4


# === Custom Weight Tests ===


class TestCustomWeights:
    def test_basic_custom_weights(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0),
        ]
        c = SignalComposite(
            method=CompositeMethod.CUSTOM,
            custom_weights={"dictionary_mda": 0.7, "llm_mda": 0.3},
        )
        result = c.combine(signals, AS_OF)

        # Normalized: dict=0.7, llm=0.3
        # score = (1.0*0.7 + 2.0*0.3) / (0.7+0.3) = 1.3
        assert abs(result[0].composite_score - 1.3) < 1e-5

    def test_missing_signal_in_weights_excluded(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0),
        ]
        # Only weight for dictionary, llm gets 0
        c = SignalComposite(
            method=CompositeMethod.CUSTOM,
            custom_weights={"dictionary_mda": 1.0},
        )
        result = c.combine(signals, AS_OF)

        # Only dictionary contributes: score=1.0
        assert result[0].composite_score == 1.0

    def test_all_zero_custom_falls_back_to_equal(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0),
        ]
        # All-zero custom weights → _normalize_weights returns None → falls back to equal
        c = SignalComposite(
            method=CompositeMethod.CUSTOM,
            custom_weights={"dictionary_mda": 0.0, "llm_mda": 0.0},
        )
        result = c.combine(signals, AS_OF)

        # Fallback to equal: (1.0 + 2.0) / 2 = 1.5
        assert result[0].composite_score == 1.5

    def test_custom_without_method_uses_equal(self) -> None:
        """Custom weights ignored if method is EQUAL (default)."""
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0),
        ]
        # Method defaults to EQUAL — custom_weights are stored but not used
        c = SignalComposite(custom_weights={"dictionary_mda": 0.9, "llm_mda": 0.1})
        result = c.combine(signals, AS_OF)

        # Equal weight: (1.0 + 2.0) / 2 = 1.5
        assert result[0].composite_score == 1.5


# === Edge Cases ===


class TestEdgeCases:
    def test_empty_signals(self) -> None:
        c = SignalComposite()
        result = c.combine([], AS_OF)
        assert result == []

    def test_all_expired(self) -> None:
        signals = [
            _signal(ticker="AAPL", decay_weight=0.0),
            _signal(ticker="MSFT", decay_weight=0.0),
        ]
        c = SignalComposite()
        result = c.combine(signals, AS_OF)
        assert result == []

    def test_mixed_expired_and_active(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0, decay_weight=0.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0, decay_weight=1.0),
        ]
        c = SignalComposite()
        result = c.combine(signals, AS_OF)

        assert len(result) == 1
        # Only llm_mda contributes (dictionary_mda expired)
        assert result[0].composite_score == 2.0

    def test_single_ticker_gets_rank_one(self) -> None:
        signals = [_signal(ticker="AAPL", z_score=0.5)]
        c = SignalComposite()
        result = c.combine(signals, AS_OF)
        assert result[0].rank == 1

    def test_tie_breaking_by_ticker(self) -> None:
        signals = [
            _signal(ticker="MSFT", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="GOOG", signal_name="dictionary_mda", z_score=1.0),
        ]
        c = SignalComposite()
        result = c.combine(signals, AS_OF)

        # All same score, break by ticker asc
        assert [r.ticker for r in result] == ["AAPL", "GOOG", "MSFT"]
        assert [r.rank for r in result] == [1, 2, 3]

    def test_composite_signal_is_frozen(self) -> None:
        signals = [_signal(ticker="AAPL", z_score=1.0)]
        c = SignalComposite()
        result = c.combine(signals, AS_OF)

        with pytest.raises(Exception):
            result[0].composite_score = 99.0  # type: ignore[misc]

    def test_components_dict_populated(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0),
        ]
        c = SignalComposite()
        result = c.combine(signals, AS_OF)

        assert "dictionary_mda" in result[0].components
        assert "llm_mda" in result[0].components

    def test_signal_date_matches_as_of(self) -> None:
        as_of = date(2024, 6, 1)
        signals = [_signal(ticker="AAPL", z_score=1.0)]
        c = SignalComposite()
        result = c.combine(signals, as_of)
        assert result[0].signal_date == as_of

    def test_negative_z_scores(self) -> None:
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=-2.0),
            _signal(ticker="MSFT", signal_name="dictionary_mda", z_score=1.0),
        ]
        c = SignalComposite()
        result = c.combine(signals, AS_OF)

        assert result[0].ticker == "MSFT"  # Higher score
        assert result[1].ticker == "AAPL"  # Negative
        assert result[1].composite_score == -2.0


# === Logging Tests ===


class TestLogging:
    def test_combine_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        signals = [_signal(ticker="AAPL", z_score=1.0)]
        c = SignalComposite()

        with caplog.at_level(logging.INFO, logger="edgar_sentinel.signals"):
            c.combine(signals, AS_OF)

        assert "Combined 1 tickers" in caplog.text

    def test_empty_signals_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        c = SignalComposite()

        with caplog.at_level(logging.INFO, logger="edgar_sentinel.signals"):
            c.combine([], AS_OF)

        assert "No active signals" in caplog.text


# === Normalize Weights Tests ===


class TestNormalizeWeights:
    def test_none_returns_none(self) -> None:
        assert SignalComposite._normalize_weights(None) is None

    def test_all_zero_returns_none(self) -> None:
        assert SignalComposite._normalize_weights({"a": 0.0, "b": 0.0}) is None

    def test_normalizes_to_sum_one(self) -> None:
        result = SignalComposite._normalize_weights({"a": 3.0, "b": 7.0})
        assert result is not None
        assert abs(result["a"] - 0.3) < 1e-6
        assert abs(result["b"] - 0.7) < 1e-6

    def test_handles_negative_values(self) -> None:
        result = SignalComposite._normalize_weights({"a": -2.0, "b": 8.0})
        assert result is not None
        total = sum(abs(v) for v in result.values())
        assert abs(total - 1.0) < 1e-6


# === Module Import Tests ===


class TestModuleImports:
    def test_import_from_package(self) -> None:
        from edgar_sentinel.signals import SignalComposite as SC
        assert SC is SignalComposite

    def test_in_all(self) -> None:
        from edgar_sentinel import signals
        assert "SignalComposite" in signals.__all__


# === Multi-signal Integration Tests ===


class TestIntegration:
    def test_realistic_cross_section(self) -> None:
        """Simulate a realistic cross-section with multiple tickers and signals."""
        signals = [
            # AAPL: strong positive
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.5, decay_weight=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=2.0, decay_weight=0.8),
            _signal(ticker="AAPL", signal_name="similarity_mda", z_score=0.5, decay_weight=0.6),
            # MSFT: moderate
            _signal(ticker="MSFT", signal_name="dictionary_mda", z_score=0.3, decay_weight=1.0),
            _signal(ticker="MSFT", signal_name="llm_mda", z_score=0.5, decay_weight=0.9),
            _signal(ticker="MSFT", signal_name="similarity_mda", z_score=-0.1, decay_weight=0.7),
            # GOOG: negative
            _signal(ticker="GOOG", signal_name="dictionary_mda", z_score=-1.0, decay_weight=1.0),
            _signal(ticker="GOOG", signal_name="llm_mda", z_score=-0.5, decay_weight=0.5),
        ]
        c = SignalComposite(method=CompositeMethod.EQUAL)
        result = c.combine(signals, AS_OF)

        assert len(result) == 3
        assert result[0].ticker == "AAPL"
        assert result[0].rank == 1
        assert result[2].ticker == "GOOG"
        assert result[2].rank == 3

        # All composite scores should be finite
        for r in result:
            assert r.composite_score == r.composite_score  # not NaN

    def test_ic_weighted_cross_section(self) -> None:
        """IC-weighted with multiple tickers."""
        signals = [
            _signal(ticker="AAPL", signal_name="dictionary_mda", z_score=1.0),
            _signal(ticker="AAPL", signal_name="llm_mda", z_score=-1.0),
            _signal(ticker="MSFT", signal_name="dictionary_mda", z_score=-0.5),
            _signal(ticker="MSFT", signal_name="llm_mda", z_score=0.5),
        ]
        # llm has higher IC — tilt toward llm scores
        ic_values = {"dictionary_mda": 0.02, "llm_mda": 0.06}
        c = SignalComposite(method=CompositeMethod.IC_WEIGHTED, ic_values=ic_values)
        result = c.combine(signals, AS_OF)

        # AAPL: dict z=1.0 weight=0.25, llm z=-1.0 weight=0.75 → tilted negative
        # MSFT: dict z=-0.5 weight=0.25, llm z=0.5 weight=0.75 → tilted positive
        assert result[0].ticker == "MSFT"  # Higher due to llm weight
        assert result[1].ticker == "AAPL"
