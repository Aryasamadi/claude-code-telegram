from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from src.providers.service import ProviderService


def _service(tmp_path) -> ProviderService:
    return ProviderService(SimpleNamespace(database_path=tmp_path / "bot.db"))


def test_catalog_error_is_logged_and_supplied_model_is_kept(
    tmp_path, monkeypatch, caplog
):
    service = _service(tmp_path)

    def fail_catalog(_base_url: str, _api_key: str) -> list[str]:
        raise OSError("catalog request failed for secret-api-key")

    monkeypatch.setattr(service, "_fetch_model_ids", fail_catalog)
    monkeypatch.setattr(service, "_write_config", lambda: None)
    monkeypatch.setattr(service, "_restart_proxy", lambda: None)

    with caplog.at_level(logging.WARNING, logger="src.providers.service"):
        count, catalog_available, selected_alias = asyncio.run(
            service.add_provider(
                "Example Provider",
                "https://api.example.com/v1",
                "example-model",
                "secret-api-key",
            )
        )

    assert count == 1
    assert catalog_available is False
    assert selected_alias.startswith("tg-example-provider-")
    assert "error_type=OSError" in caplog.text
    assert "[REDACTED]" in caplog.text
    assert "secret-api-key" not in caplog.text


@pytest.mark.parametrize("failure_stage", ["write_config", "proxy_startup"])
def test_registration_logging_does_not_mask_original_error(
    tmp_path, monkeypatch, caplog, failure_stage
):
    service = _service(tmp_path)
    monkeypatch.setattr(
        service, "_fetch_model_ids", lambda _base_url, _api_key: ["example-model"]
    )

    original_error = RuntimeError("controlled failure: secret-api-key")
    if failure_stage == "write_config":

        def fail_write_config() -> None:
            raise original_error

        monkeypatch.setattr(service, "_write_config", fail_write_config)
        monkeypatch.setattr(service, "_restart_proxy", lambda: None)
    else:
        monkeypatch.setattr(service, "_write_config", lambda: None)

        def fail_proxy_startup() -> None:
            raise original_error

        monkeypatch.setattr(service, "_restart_proxy", fail_proxy_startup)

    with caplog.at_level(logging.ERROR, logger="src.providers.service"):
        with pytest.raises(RuntimeError, match="controlled failure"):
            asyncio.run(
                service.add_provider(
                    "Example Provider",
                    "https://api.example.com/v1",
                    "example-model",
                    "secret-api-key",
                )
            )

    assert f"stage={failure_stage}" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert "[REDACTED]" in caplog.text
    assert "secret-api-key" not in caplog.text
