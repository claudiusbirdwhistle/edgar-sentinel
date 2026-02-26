"""Configuration loading, validation, and access."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from edgar_sentinel.core.exceptions import ConfigError
from edgar_sentinel.core.models import (
    CompositeMethod,
    LLMProvider,
    RebalanceFrequency,
    StorageBackend,
)


class EdgarConfig(BaseModel):
    """EDGAR API access configuration."""

    model_config = ConfigDict(frozen=True)

    user_agent: str
    rate_limit: int = 10
    cache_dir: str = "./data/filings"
    request_timeout: int = 30

    @field_validator("user_agent")
    @classmethod
    def user_agent_has_contact(cls, v: str) -> str:
        """SEC requires user-agent with name and email."""
        if "@" not in v:
            raise ValueError(
                "user_agent must contain an email address per SEC policy. "
                "Example: 'YourName your@email.com'"
            )
        return v

    @field_validator("rate_limit")
    @classmethod
    def rate_limit_within_sec_policy(cls, v: int) -> int:
        if v < 1 or v > 10:
            raise ValueError("rate_limit must be between 1 and 10 (SEC policy)")
        return v


class StorageConfig(BaseModel):
    """Storage backend configuration."""

    model_config = ConfigDict(frozen=True)

    backend: StorageBackend = StorageBackend.SQLITE
    sqlite_path: str = "./data/edgar_sentinel.db"
    postgresql_url: str | None = None

    @model_validator(mode="after")
    def pg_url_required_for_pg(self) -> StorageConfig:
        if self.backend == StorageBackend.POSTGRESQL and not self.postgresql_url:
            raise ValueError("postgresql_url is required when backend is 'postgresql'")
        return self


class DictionaryAnalyzerConfig(BaseModel):
    """Configuration for the Loughran-McDonald dictionary analyzer."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    dictionary_path: str = "./data/lm_dictionary.csv"


class SimilarityAnalyzerConfig(BaseModel):
    """Configuration for the cosine similarity analyzer."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True


class LLMAnalyzerConfig(BaseModel):
    """Configuration for the LLM-based analyzer."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    provider: LLMProvider = LLMProvider.CLAUDE_CLI
    model: str = "claude-sonnet-4-6"
    max_concurrent: int = 2
    timeout_seconds: int = 60

    @field_validator("max_concurrent")
    @classmethod
    def max_concurrent_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_concurrent must be >= 1")
        return v


class AnalyzersConfig(BaseModel):
    """Aggregated analyzer configuration."""

    model_config = ConfigDict(frozen=True)

    dictionary: DictionaryAnalyzerConfig = DictionaryAnalyzerConfig()
    similarity: SimilarityAnalyzerConfig = SimilarityAnalyzerConfig()
    llm: LLMAnalyzerConfig = LLMAnalyzerConfig()


class SignalsConfig(BaseModel):
    """Signal generation configuration."""

    model_config = ConfigDict(frozen=True)

    buffer_days: int = 2
    decay_half_life: int = 90
    composite_method: CompositeMethod = CompositeMethod.EQUAL

    @field_validator("buffer_days")
    @classmethod
    def buffer_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("buffer_days must be >= 0")
        return v

    @field_validator("decay_half_life")
    @classmethod
    def half_life_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("decay_half_life must be >= 1")
        return v


class BacktestSettingsConfig(BaseModel):
    """Default backtest settings (overridable per-run via BacktestConfig)."""

    model_config = ConfigDict(frozen=True)

    start_date: date = date(2020, 1, 1)
    end_date: date = date(2025, 12, 31)
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.QUARTERLY
    num_quantiles: int = 5
    transaction_cost_bps: int = 10


class APIConfig(BaseModel):
    """FastAPI server configuration."""

    model_config = ConfigDict(frozen=True)

    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None


class SentinelConfig(BaseModel):
    """Root configuration for the entire edgar-sentinel system."""

    model_config = ConfigDict(frozen=True)

    edgar: EdgarConfig
    storage: StorageConfig = StorageConfig()
    analyzers: AnalyzersConfig = AnalyzersConfig()
    signals: SignalsConfig = SignalsConfig()
    backtest: BacktestSettingsConfig = BacktestSettingsConfig()
    api: APIConfig = APIConfig()


def load_config(
    config_path: str | None = None,
    env_prefix: str = "EDGAR_SENTINEL_",
) -> SentinelConfig:
    """Load configuration from environment + YAML file + defaults.

    Resolution order (highest priority first):
    1. Environment variables (EDGAR_SENTINEL_EDGAR__USER_AGENT, etc.)
    2. YAML file at config_path
    3. Built-in defaults

    Nested keys use double-underscore in env vars:
        EDGAR_SENTINEL_EDGAR__RATE_LIMIT=5  ->  edgar.rate_limit = 5
    """
    try:
        yaml_path = _resolve_config_path(config_path)
        base: dict = {}
        if yaml_path is not None:
            base = _load_yaml(yaml_path)

        merged = _merge_env_vars(base, env_prefix)
        return SentinelConfig.model_validate(merged)
    except Exception as e:
        if isinstance(e, ConfigError):
            raise
        raise ConfigError(str(e), context={"source": "load_config"}) from e


def _resolve_config_path(explicit: str | None) -> Path | None:
    """Determine config file path."""
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise ConfigError(
                f"Config file not found: {explicit}",
                context={"field": "config_path", "value": explicit},
            )
        return p

    env_path = os.environ.get("EDGAR_SENTINEL_CONFIG")
    if env_path:
        p = Path(env_path)
        if not p.exists():
            raise ConfigError(
                f"Config file from EDGAR_SENTINEL_CONFIG not found: {env_path}",
                context={"field": "EDGAR_SENTINEL_CONFIG", "value": env_path},
            )
        return p

    default = Path("edgar-sentinel.yml")
    if default.exists():
        return default

    return None


def _load_yaml(path: Path) -> dict:
    """Load and parse YAML file."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(
                f"YAML config must be a mapping, got {type(data).__name__}",
                context={"field": "config_file", "value": str(path)},
            )
        return data
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Failed to parse YAML config: {e}",
            context={"field": "config_file", "value": str(path)},
        ) from e


def _merge_env_vars(base: dict, prefix: str) -> dict:
    """Overlay environment variables onto base config dict.

    Double-underscore separates nesting levels.
    Values are auto-cast: "true"/"false" -> bool, numeric strings -> int.
    """
    result = dict(base)

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Strip prefix, split by double-underscore for nesting
        remainder = key[len(prefix) :]
        parts = [p.lower() for p in remainder.split("__")]

        # Skip the CONFIG env var itself
        if parts == ["config"]:
            continue

        # Auto-cast values
        cast_value = _auto_cast(value)

        # Nest into result dict
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = cast_value

    return result


def _auto_cast(value: str) -> str | int | float | bool:
    """Auto-cast string values from environment variables."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
