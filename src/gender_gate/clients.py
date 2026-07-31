from __future__ import annotations

import os
import time
from typing import Any


class ClientConfigurationError(RuntimeError):
    pass


def resolve_model_name(
    model_config: dict[str, Any], request_config: dict[str, Any]
) -> str:
    """Resolve the model without coupling the fixed Judge to target-model env vars.

    Precedence:
      1. per-run ``model`` override;
      2. provider-specific model environment variable;
      3. fixed/default model declared in ``models.yaml``.
    """

    explicit = str(request_config.get("model") or "").strip()
    if explicit:
        return explicit

    model_env = model_config.get("model_env")
    if model_env:
        env_value = os.getenv(str(model_env), "").strip()
        if env_value:
            return env_value

    return str(
        model_config.get("model") or model_config.get("default_model") or ""
    ).strip()


class OpenAICompatibleClient:
    def __init__(self, model_config: dict[str, Any], request_config: dict[str, Any]):
        api_key_env = model_config.get("api_key_env")
        base_url = request_config.get("base_url") or model_config.get("base_url")
        base_url_env = model_config.get("base_url_env")

        api_key = os.getenv(str(api_key_env), "") if api_key_env else ""
        model = resolve_model_name(model_config, request_config)
        if base_url_env:
            base_url = os.getenv(str(base_url_env), str(base_url or ""))

        if not model:
            model_hint = model_config.get("model_env") or "the config field 'model'"
            raise ClientConfigurationError(
                f"Missing model name. Set {model_hint} or pass --model."
            )
        if not api_key and not model_config.get("allow_missing_key", False):
            raise ClientConfigurationError(
                f"Missing API key. Set environment variable {api_key_env}."
            )

        self.model = model
        self.provider = str(model_config.get("provider") or "openai_compatible")
        self.temperature = request_config.get("temperature")
        self.max_output_tokens = int(request_config.get("max_output_tokens", 8))
        # Provider configuration may require a different OpenAI-compatible token field.
        # For example, Qwen thinking models should use max_completion_tokens so the
        # budget explicitly covers reasoning_content + final answer.
        self.max_tokens_field = str(
            model_config.get("max_tokens_field")
            or request_config.get("max_tokens_field")
            or "max_tokens"
        )
        self.retries = int(request_config.get("retries", 3))

        # Merge provider defaults with per-run overrides instead of replacing the
        # provider block wholesale. This lets a run override only thinking_budget
        # while retaining enable_thinking=True from models.yaml.
        model_extra = dict(model_config.get("extra_body") or {})
        request_extra = dict(request_config.get("extra_body") or {})
        model_extra.update(request_extra)
        self.extra_body = model_extra or None
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ClientConfigurationError(
                "The openai package is not installed. Run: pip install -e ."
            ) from exc

        self.client = OpenAI(
            api_key=api_key or "local",
            base_url=str(base_url) if base_url else None,
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
        if self.extra_body:
            request["extra_body"] = self.extra_body

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(**request)
                choice = response.choices[0]
                message = choice.message
                content = message.content
                text = "" if content is None else str(content).strip()
                if not text:
                    reasoning = getattr(message, "reasoning_content", None)
                    reasoning_chars = len(str(reasoning)) if reasoning else 0
                    finish_reason = getattr(choice, "finish_reason", None)
                    raise RuntimeError(
                        "Model returned empty final content "
                        f"(finish_reason={finish_reason!r}, "
                        f"reasoning_chars={reasoning_chars}, "
                        f"max_output_tokens={self.max_output_tokens}). "
                        "For thinking models, increase the output-token budget."
                    )
                return text
            except Exception as exc:  # pragma: no cover - provider/network dependent
                last_error = exc
                if attempt + 1 >= self.retries:
                    break
                time.sleep(min(2**attempt, 8))

        raise RuntimeError(
            f"Model request failed after {self.retries} attempts: {last_error}"
        )
