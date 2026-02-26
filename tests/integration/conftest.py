"""Integration test fixtures — real I/O but no network."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from edgar_sentinel.core.config import (
    APIConfig,
    EdgarConfig,
    SentinelConfig,
    StorageConfig,
)
from edgar_sentinel.core.models import (
    Filing,
    FilingMetadata,
    FilingSection,
    FormType,
    StorageBackend,
)
from edgar_sentinel.ingestion.store import SqliteStore


@pytest.fixture
async def integration_store(tmp_path: Path) -> SqliteStore:
    """An initialized SqliteStore for integration tests."""
    config = StorageConfig(
        backend=StorageBackend.SQLITE,
        sqlite_path=str(tmp_path / "integration.db"),
    )
    store = SqliteStore(config)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def populated_store(integration_store: SqliteStore) -> SqliteStore:
    """A SqliteStore with 5 years of AAPL 10-K filings pre-loaded."""
    mda_text = (
        "The Company's business environment is highly competitive. "
        "Revenue increased driven by strong customer demand for our products. "
        "We continue to invest in research and development to sustain growth. "
        "Operating expenses decreased due to cost optimization initiatives. "
        "Management believes the Company is well positioned for future success."
    )

    for year in range(2019, 2024):
        accession = f"0000320193-{year % 100:02d}-000106"
        meta = FilingMetadata(
            cik="320193",
            ticker="AAPL",
            company_name="Apple Inc.",
            form_type=FormType.FORM_10K,
            filed_date=date(year, 11, 3),
            accession_number=accession,
            url=f"https://www.sec.gov/test/{accession}",
        )
        section = FilingSection(
            filing_id=accession,
            section_name="mda",
            raw_text=mda_text,
            word_count=len(mda_text.split()),
            extracted_at=datetime(year, 11, 4, 10, 0, 0),
        )
        filing = Filing(metadata=meta, sections={"mda": section})
        await integration_store.save_filing(filing)

    return integration_store


@pytest.fixture
def integration_config(tmp_path: Path) -> SentinelConfig:
    """Config for integration tests with all features enabled."""
    return SentinelConfig(
        edgar=EdgarConfig(user_agent="TestAgent test@example.com"),
        storage=StorageConfig(
            backend=StorageBackend.SQLITE,
            sqlite_path=str(tmp_path / "integration.db"),
        ),
        api=APIConfig(),
    )


@pytest.fixture
def api_config(tmp_path: Path) -> SentinelConfig:
    """Config for API integration tests."""
    return SentinelConfig(
        edgar=EdgarConfig(user_agent="TestAgent test@example.com"),
        storage=StorageConfig(
            backend=StorageBackend.SQLITE,
            sqlite_path=str(tmp_path / "api-test.db"),
        ),
        api=APIConfig(),
    )
