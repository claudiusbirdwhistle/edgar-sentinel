"""Signals module: transforms analyzer outputs into normalized trading signals."""

from edgar_sentinel.signals.builder import FilingDateMapping, SignalBuilder
from edgar_sentinel.signals.composite import SignalComposite
from edgar_sentinel.signals.decay import SignalDecay

__all__ = [
    "FilingDateMapping",
    "SignalBuilder",
    "SignalComposite",
    "SignalDecay",
]
