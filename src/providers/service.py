"""Runtime custom-provider registry and local LiteLLM bridge.

Provider credentials are held in process memory/environment only. The generated
LiteLLM config contains environment-variable references, never raw API keys.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import Settings

LITELLM_HOST = "127.0.0.1"
LITELLM_PORT = 4000
LITELLM_BASE_URL = f"http://{LITELLM_HOST}:{LITELLM_PORT}"
logger = logging.getLogger(__name__)


def _safe_error_message(error: BaseException, secret: str = "") -> str:
    """Return a bounded exception summary with credentials removed."""
    message = str(error)
    if secret:
        message = message.replace(secret, "[REDACTED]")
    message = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)((?:api[_-]?key|token|secret)\s*[:=]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    return message[:1000]


@dataclass(frozen=True)
class ModelRoute:
    alias: str
    provider_name: str
    model_id: str
    base_url: str
    key_env: str


class ProviderService:
    """Manage arbitrary OpenAI-compatible providers for the Claude Agent SDK."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._models: dict[str, ModelRoute] = {}
        self._provider_keys: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._process: subprocess.Popen[Any] | None = None
        self._log_handle: Any = None
        data_dir = (
            settings.database_path.parent
            if settings.database_path
            else Path.cwd() / "data"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        self._config_path = data_dir / "litellm-providers.yaml"
        self._log_path = data_dir / "litellm-proxy.log"
        self._master_key = "sk-" + secrets.token_urlsafe(36)
        os.environ["LITELLM_MASTER_KEY"] = self._master_key
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["ANTHROPIC_AUTH_TOKEN"] = self._master_key
        os.environ["ANTHROPIC_BASE_URL"] = LITELLM_BASE_URL

    @staticmethod
    def _validate_base_url(base_url: str) -> str:
        parsed = urllib.parse.urlsplit(base_url.strip())
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Base URL must be a public HTTPS OpenAI-compatible API URL."
            )
        host = parsed.hostname.lower()
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            raise ValueError(
                "Local/private provider URLs cannot be reached safely from this runner."
            )
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("Private or non-public IP addresses are not supported.")
        normalized = base_url.strip().rstrip("/")
        return normalized

    @staticmethod
    def _fetch_model_ids(base_url: str, api_key: str) -> list[str]:
        request = urllib.request.Request(
            f"{base_url}/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "claude-code-telegram-provider-manager/1.0",
            },
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        model_ids: list[str] = []
        if isinstance(rows, list):
            for row in rows:
                model_id = row.get("id") if isinstance(row, dict) else None
                if isinstance(model_id, str) and model_id.strip():
                    model_ids.append(model_id.strip())
        return list(dict.fromkeys(model_ids))[:500]

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return (slug or "provider")[:28]

    async def add_provider(
        self, provider_name: str, base_url: str, model_id: str, api_key: str
    ) -> tuple[int, bool, str]:
        provider_name = provider_name.strip()
        model_id = model_id.strip()
        api_key = api_key.strip()
        if not provider_name or not model_id or not api_key:
            raise ValueError("All four values are required.")
        normalized_url = self._validate_base_url(base_url)
        if len(provider_name) > 80 or len(model_id) > 180:
            raise ValueError("Provider name or model ID is too long.")

        catalog_available = True
        try:
            model_ids = await asyncio.to_thread(
                self._fetch_model_ids, normalized_url, api_key
            )
        except Exception as exc:
            model_ids = []
            catalog_available = False
            logger.warning(
                "Provider model catalog request failed; using the supplied model ID "
                "(error_type=%s, error=%s)",
                type(exc).__name__,
                _safe_error_message(exc, api_key),
            )
        if model_id not in model_ids:
            model_ids.insert(0, model_id)
        if not model_ids:
            raise ValueError("The provider returned no model IDs.")

        provider_identity = (
            hashlib.sha256(f"{provider_name}|{normalized_url}".encode("utf-8"))
            .hexdigest()[:12]
            .upper()
        )
        key_env = f"TG_PROVIDER_{provider_identity}_API_KEY"
        os.environ[key_env] = api_key

        slug = self._slug(provider_name)
        selected_alias = ""
        async with self._lock:
            for item_id in model_ids:
                digest = hashlib.sha256(
                    f"{provider_name}|{normalized_url}|{item_id}".encode("utf-8")
                ).hexdigest()[:8]
                alias = f"tg-{slug}-{digest}"
                if alias not in self._models:
                    self._models[alias] = ModelRoute(
                        alias=alias,
                        provider_name=provider_name,
                        model_id=item_id,
                        base_url=normalized_url,
                        key_env=key_env,
                    )
                if item_id == model_id:
                    selected_alias = alias
            self._provider_keys[key_env] = api_key
            try:
                await asyncio.to_thread(self._write_config)
            except Exception as exc:
                logger.error(
                    "Provider registration failed "
                    "(stage=%s provider=%s model=%s error_type=%s error=%s)",
                    "write_config",
                    provider_name,
                    model_id,
                    type(exc).__name__,
                    _safe_error_message(exc, api_key),
                )
                raise
            try:
                await asyncio.to_thread(self._restart_proxy)
            except Exception as exc:
                logger.error(
                    "Provider registration failed "
                    "(stage=%s provider=%s model=%s error_type=%s error=%s)",
                    "proxy_startup",
                    provider_name,
                    model_id,
                    type(exc).__name__,
                    _safe_error_message(exc, api_key),
                )
                raise
        return len(model_ids), catalog_available, selected_alias

    def _write_config(self) -> None:
        model_list = []
        for route in self._models.values():
            model_list.append(
                {
                    "model_name": route.alias,
                    "litellm_params": {
                        "model": f"openai/{route.model_id}",
                        "api_base": route.base_url,
                        "api_key": f"os.environ/{route.key_env}",
                        "drop_params": True,
                    },
                }
            )
        payload = {
            "model_list": model_list,
            "litellm_settings": {"drop_params": True},
            "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
        }
        temporary_path = self._config_path.with_suffix(".yaml.tmp")
        # JSON is a valid YAML subset, so LiteLLM can load this without adding
        # another direct dependency to the bot project.
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary_path, 0o600)
        temporary_path.replace(self._config_path)
        os.chmod(self._config_path, 0o600)

    def _restart_proxy(self) -> None:
        configured_executable = os.environ.get("LITELLM_EXECUTABLE")
        executable = configured_executable or shutil.which("litellm")
        if not executable or not Path(executable).is_file():
            logger.error(
                "LiteLLM executable is unavailable (configured_executable=%s)",
                bool(configured_executable),
            )
            raise RuntimeError(
                "LiteLLM is missing. Install the LiteLLM proxy dependency."
            )
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._log_handle:
            self._log_handle.close()
        self._log_handle = self._log_path.open("a", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                [
                    executable,
                    "--config",
                    str(self._config_path),
                    "--host",
                    LITELLM_HOST,
                    "--port",
                    str(LITELLM_PORT),
                ],
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        except Exception as exc:
            logger.error(
                "Could not launch LiteLLM proxy (error_type=%s, error=%s)",
                type(exc).__name__,
                _safe_error_message(exc),
            )
            raise
        for _ in range(45):
            if self._process.poll() is not None:
                self._log_startup_failure()
                raise RuntimeError(
                    f"LiteLLM exited during startup (code {self._process.returncode})."
                )
            try:
                request = urllib.request.Request(
                    f"{LITELLM_BASE_URL}/health/liveliness", method="GET"
                )
                with urllib.request.urlopen(request, timeout=2):
                    return
            except Exception:
                time.sleep(1)
        self._log_startup_failure()
        raise RuntimeError("LiteLLM did not become healthy within 45 seconds.")

    def _log_startup_failure(self) -> None:
        """Copy a redacted proxy startup tail into the GitHub Actions log."""
        try:
            lines = self._log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            logger.exception("Could not read LiteLLM startup log")
            return

        secret_values = [self._master_key, *self._provider_keys.values()]
        for line in lines[-80:]:
            safe_line = line
            for secret in secret_values:
                if secret:
                    safe_line = safe_line.replace(secret, "[REDACTED]")
            logger.error("LiteLLM startup: %s", safe_line[:2000])

    async def list_models(self) -> list[ModelRoute]:
        return sorted(
            self._models.values(),
            key=lambda route: (
                route.provider_name.casefold(),
                route.model_id.casefold(),
            ),
        )

    async def select_model(self, alias: str) -> ModelRoute | None:
        route = self._models.get(alias)
        if route is not None:
            self.settings.claude_model = route.alias
        return route

    async def stop(self) -> None:
        async with self._lock:
            if self._process and self._process.poll() is None:
                self._process.terminate()
                try:
                    await asyncio.to_thread(self._process.wait, 10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    await asyncio.to_thread(self._process.wait, 5)
            self._process = None
            if self._log_handle:
                self._log_handle.close()
                self._log_handle = None
            for key_env in self._provider_keys:
                os.environ.pop(key_env, None)
            self._provider_keys.clear()
            os.environ.pop("LITELLM_MASTER_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
            os.environ.pop("ANTHROPIC_BASE_URL", None)
