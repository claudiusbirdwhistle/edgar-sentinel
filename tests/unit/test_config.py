"""Tests for edgar_sentinel.core.config."""

import os
import pytest
from pathlib import Path

from pydantic import ValidationError

from edgar_sentinel.core.config import (
    EdgarConfig,
    SentinelConfig,
    StorageConfig,
    load_config,
    _auto_cast,
    _merge_env_vars,
)
from edgar_sentinel.core.exceptions import ConfigError
from edgar_sentinel.core.models import StorageBackend


class TestEdgarConfig:
    def test_valid_construction(self):
        c = EdgarConfig(user_agent="TestBot test@example.com")
        assert c.rate_limit == 10
        assert c.user_agent == "TestBot test@example.com"

    def test_user_agent_requires_email(self):
        with pytest.raises(ValidationError, match="email address"):
            EdgarConfig(user_agent="NoEmailHere")

    def test_rate_limit_max(self):
        with pytest.raises(ValidationError, match="between 1 and 10"):
            EdgarConfig(user_agent="Bot test@test.com", rate_limit=15)

    def test_rate_limit_min(self):
        with pytest.raises(ValidationError, match="between 1 and 10"):
            EdgarConfig(user_agent="Bot test@test.com", rate_limit=0)


class TestStorageConfig:
    def test_defaults_to_sqlite(self):
        c = StorageConfig()
        assert c.backend == StorageBackend.SQLITE

    def test_pg_requires_url(self):
        with pytest.raises(ValidationError, match="postgresql_url is required"):
            StorageConfig(backend=StorageBackend.POSTGRESQL)

    def test_pg_with_url_valid(self):
        c = StorageConfig(
            backend=StorageBackend.POSTGRESQL,
            postgresql_url="postgresql://localhost/edgar",
        )
        assert c.postgresql_url is not None


class TestLoadConfig:
    def test_defaults_with_env_user_agent(self, monkeypatch):
        monkeypatch.setenv("EDGAR_SENTINEL_EDGAR__USER_AGENT", "Bot test@test.com")
        config = load_config()
        assert config.edgar.user_agent == "Bot test@test.com"
        assert config.storage.backend == StorageBackend.SQLITE

    def test_yaml_loading(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text(
            "edgar:\n  user_agent: 'YAMLBot yaml@test.com'\n  rate_limit: 5\n"
        )
        # Clear any env vars that might conflict
        for key in list(os.environ):
            if key.startswith("EDGAR_SENTINEL_"):
                monkeypatch.delenv(key, raising=False)
        config = load_config(config_path=str(yaml_file))
        assert config.edgar.user_agent == "YAMLBot yaml@test.com"
        assert config.edgar.rate_limit == 5

    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text(
            "edgar:\n  user_agent: 'YAMLBot yaml@test.com'\n  rate_limit: 5\n"
        )
        monkeypatch.setenv("EDGAR_SENTINEL_EDGAR__RATE_LIMIT", "8")
        config = load_config(config_path=str(yaml_file))
        assert config.edgar.rate_limit == 8

    def test_nested_env_vars(self, monkeypatch):
        monkeypatch.setenv("EDGAR_SENTINEL_EDGAR__USER_AGENT", "Bot test@test.com")
        monkeypatch.setenv("EDGAR_SENTINEL_ANALYZERS__LLM__ENABLED", "true")
        config = load_config()
        assert config.analyzers.llm.enabled is True

    def test_missing_user_agent_raises(self, monkeypatch):
        for key in list(os.environ):
            if key.startswith("EDGAR_SENTINEL_"):
                monkeypatch.delenv(key, raising=False)
        with pytest.raises(ConfigError):
            load_config()

    def test_missing_config_file_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config(config_path="/nonexistent/file.yml")

    def test_config_is_frozen(self, monkeypatch):
        monkeypatch.setenv("EDGAR_SENTINEL_EDGAR__USER_AGENT", "Bot test@test.com")
        config = load_config()
        with pytest.raises(ValidationError):
            config.edgar = None


class TestAutoCast:
    def test_true(self):
        assert _auto_cast("true") is True
        assert _auto_cast("True") is True
        assert _auto_cast("TRUE") is True

    def test_false(self):
        assert _auto_cast("false") is False

    def test_int(self):
        assert _auto_cast("42") == 42

    def test_float(self):
        assert _auto_cast("3.14") == 3.14

    def test_string(self):
        assert _auto_cast("hello") == "hello"


class TestMergeEnvVars:
    def test_simple_override(self, monkeypatch):
        monkeypatch.setenv("TEST_EDGAR__RATE_LIMIT", "5")
        result = _merge_env_vars({"edgar": {"rate_limit": 10}}, "TEST_")
        assert result["edgar"]["rate_limit"] == 5

    def test_creates_nested_structure(self, monkeypatch):
        monkeypatch.setenv("TEST_ANALYZERS__LLM__ENABLED", "true")
        result = _merge_env_vars({}, "TEST_")
        assert result["analyzers"]["llm"]["enabled"] is True

    def test_skips_config_key(self, monkeypatch):
        monkeypatch.setenv("TEST_CONFIG", "/some/path")
        result = _merge_env_vars({}, "TEST_")
        assert "config" not in result
