from __future__ import annotations

import os
import time
from typing import Any

class ClientConfigurationError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, model_config: dict[str, Any], request_config: dict[str, Any]):
        api_key_env = model_config.get("api_key_env")
        model_env = model_config.get("model_env")
        base_url = model_config.get("base_url")
        base_url_env = model_config.get("base_url_env")

        api_key = os.getenv(api_key_env, "") if api_key_env else ""
        model = os.getenv(model_env, "") if model_env else ""
        if base_url_env:
            base_url = os.getenv(base_url_env, base_url or "")

        if not model:
            raise ClientConfigurationError(
                f"Missing model name. Set environment variable {model_env}."
            )
        if not api_key and not model_config.get("allow_missing_key", False):
            raise ClientConfigurationError(
                f"Missing API key. Set environment variable {api_key_env}."
            )

        self.model = model
        self.temperature = request_config.get("temperature")
        self.max_output_tokens = int(request_config.get("max_output_tokens", 8))
        self.max_tokens_field = request_config.get("max_tokens_field", "max_tokens")
        self.retries = int(request_config.get("retries", 3))
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ClientConfigurationError(
                "The openai package is not installed. Run: pip install -e ."
            ) from exc

        self.client = OpenAI(
            api_key=api_key or "local",
            base_url=base_url or None,
            timeout=float(model_config.get("timeout_seconds", 120)),
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        request[self.max_tokens_field] = self.max_output_tokens

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(**request)
                content = response.choices[0].message.content
                return "" if content is None else str(content)
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= self.retries:
                    break
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(
            f"Model request failed after {self.retries} attempts: {last_error}"
        )
