"""End-to-end test fixtures — requires network access to EDGAR."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgar_sentinel.core.config import (
    AnalyzersConfig,
    DictionaryAnalyzerConfig,
    EdgarConfig,
    SentinelConfig,
    SimilarityAnalyzerConfig,
    StorageConfig,
)


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def pytest_configure(config):
    """Register e2e and slow markers."""
    config.addinivalue_line("markers", "e2e: end-to-end test requiring network")
    config.addinivalue_line("markers", "slow: tests that take > 10 seconds")


@pytest.fixture
def e2e_config(tmp_path: Path) -> SentinelConfig:
    """Config for e2e tests with real EDGAR access."""
    return SentinelConfig(
        edgar=EdgarConfig(
            user_agent="EdgarSentinel-E2E-Test e2e@example.com",
            rate_limit=8,
            cache_dir=str(tmp_path / "cache"),
            request_timeout=60,
        ),
        storage=StorageConfig(
            sqlite_path=str(tmp_path / "e2e.db"),
        ),
        analyzers=AnalyzersConfig(
            dictionary=DictionaryAnalyzerConfig(
                enabled=True,
                dictionary_path=str(FIXTURES_DIR / "sample_lm_dictionary.csv"),
            ),
            similarity=SimilarityAnalyzerConfig(enabled=True),
        ),
    )


@pytest.fixture
def e2e_workspace(tmp_path: Path) -> Path:
    """Temporary workspace for e2e test artifacts."""
    workspace = tmp_path / "e2e_workspace"
    workspace.mkdir()
    return workspace
