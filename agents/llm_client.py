"""Network LLM Client module for Sverk PikoClaw Swarm platform.

Enforces strict Zero Offline Mode. Interacts exclusively with Sverk LLM network REST API.
Offline rule-based fallbacks or dummy client generators are strictly prohibited.
"""
from __future__ import annotations

import abc
import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("sverk.llm_client")


class LLMConnectionError(RuntimeError):
    """Exception raised when network connection to Sverk AI fails.
    
    Indicates inability to reach LLM endpoint after max retry attempts.
    Zero Offline Mode is strictly enforced; this exception triggers safe abort
    and exit code 1 in mission runner.
    """
    pass


# Alias for compatibility with safety handlers
LLMUnavailableError = LLMConnectionError


def load_env_file(filepath: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Lightweight .env file parser with zero external dependencies.
    
    Searches for .env file at specified path or upwards from current working directory / module file path.
    Loads parsed key-value pairs into os.environ (if not already set).
    """
    loaded: Dict[str, str] = {}
    env_path: Optional[Path] = None

    if filepath is not None:
        p = Path(filepath)
        if p.exists() and p.is_file():
            env_path = p
    else:
        curr = Path(os.getcwd()).resolve()
        for parent in [curr] + list(curr.parents):
            candidate = parent / ".env"
            if candidate.exists() and candidate.is_file():
                env_path = candidate
                break

        if env_path is None:
            this_dir = Path(__file__).resolve().parent
            for parent in [this_dir] + list(this_dir.parents):
                candidate = parent / ".env"
                if candidate.exists() and candidate.is_file():
                    env_path = candidate
                    break

    if env_path is not None and env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip()
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            val = val[1:-1]
                        loaded[key] = val
                        if key not in os.environ:
                            os.environ[key] = val
        except OSError as exc:
            logger.warning("Failed to read .env file at %s: %s", env_path, exc)

    return loaded


class LLMClient(abc.ABC):
    """Abstract base class for LLM clients.
    
    All implementations MUST operate strictly online via network API.
    Offline fallback generators (OfflineRuleBasedClient) are forbidden.
    """

    @abc.abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_s: Optional[float] = None,
    ) -> str:
        """Sends chat completion request to network LLM endpoint."""
        pass


class SverkLLMClient(LLMClient):
    """Network HTTP REST client for Sverk AI (Gemma4-VLM / OpenAI compatible API).
    
    Zero Offline Mode: Performs up to max_retries attempts with exponential backoff on network errors.
    Raises LLMConnectionError on final failure.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://ai.sverk.io/v1",
        model: str = "gemma4-vlm",
        max_retries: int = 2,
        backoff_factor: float = 0.5,
        env_file: Optional[Union[str, Path]] = None,
    ) -> None:
        load_env_file(env_file)
        self.api_key = api_key or os.environ.get("SVERK_LLM_API_KEY")
        self.base_url = os.environ.get("SVERK_LLM_BASE_URL", base_url).rstrip("/")
        self.model = os.environ.get("SVERK_LLM_MODEL", model)
        self.max_retries = max(0, max_retries)
        self.backoff_factor = backoff_factor

        if not self.api_key:
            logger.warning("API key SVERK_LLM_API_KEY is not set in arguments or .env.")

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_s: Optional[float] = None,
    ) -> str:
        if not self.api_key:
            raise LLMConnectionError(
                "Missing API key SVERK_LLM_API_KEY. Zero Offline Mode is strictly enforced. Mission aborted."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
        }
        data = json.dumps(payload).encode("utf-8")
        url_timeout = float(timeout_s) if timeout_s is not None and float(timeout_s) > 0 else None

        last_exception: Optional[Exception] = None
        total_attempts = self.max_retries + 1

        for attempt in range(total_attempts):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=url_timeout) as response:
                    status_code = response.getcode()
                    if status_code == 200:
                        body = response.read().decode("utf-8")
                        res_json = json.loads(body)
                        if "choices" in res_json and isinstance(res_json["choices"], list) and len(res_json["choices"]) > 0:
                            choice = res_json["choices"][0]
                            if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict):
                                return str(choice["message"].get("content", "")).strip()
                            elif isinstance(choice, dict) and "text" in choice:
                                return str(choice.get("text", "")).strip()
                        elif "content" in res_json:
                            return str(res_json.get("content", "")).strip()
                        elif "response" in res_json:
                            return str(res_json.get("response", "")).strip()
                        return body.strip()
                    else:
                        raise urllib.error.HTTPError(url, status_code, f"Unexpected HTTP status: {status_code}", response.headers, None)
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                last_exception = exc
                logger.warning(
                    "Network error connecting to Sverk API (attempt %d/%d): %s",
                    attempt + 1,
                    total_attempts,
                    exc,
                )
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    time.sleep(sleep_time)
            except Exception as exc:
                last_exception = exc
                logger.error("Unexpected error during LLM request: %s", exc)
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** attempt)
                    time.sleep(sleep_time)

        logger.error(
            "Exhausted all %d retry attempts to Sverk API. Zero Offline Mode prevents offline fallback.",
            total_attempts,
        )
        raise LLMConnectionError(
            f"Failed to receive response from Sverk API after {total_attempts} attempts. "
            f"Offline fallback is disabled. Last error: {last_exception}"
        ) from last_exception
