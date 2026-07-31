from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class ClientConfigurationError(RuntimeError):
    pass


_LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["POSITIVE", "NEGATIVE"],
        }
    },
    "required": ["label"],
    "additionalProperties": False,
}


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
    """Unified model client.

    Most providers use the OpenAI-compatible API. Ollama profiles may opt into
    its native ``/api/chat`` endpoint when exact runtime controls such as
    ``think: false`` or JSON-schema constrained output are required.
    """

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
        self.temperature = request_config.get(
            "temperature", model_config.get("temperature")
        )
        self.max_output_tokens = int(request_config.get("max_output_tokens", 8))
        self.max_tokens_field = str(
            model_config.get("max_tokens_field")
            or request_config.get("max_tokens_field")
            or "max_tokens"
        )
        self.retries = int(request_config.get("retries", model_config.get("retries", 3)))
        self.timeout_seconds = float(model_config.get("timeout_seconds", 120))

        model_extra = dict(model_config.get("extra_body") or {})
        request_extra = dict(request_config.get("extra_body") or {})
        model_extra.update(request_extra)
        self.extra_body = model_extra or None

        self._native_ollama = self.provider == "ollama_native"
        self._ollama_think = model_config.get("think")
        self._ollama_structured_output = bool(
            model_config.get("structured_output", False)
        )
        self._ollama_keep_alive = model_config.get("keep_alive")
        self._ollama_url = ""
        self.client: Any = None

        if self._native_ollama:
            if not base_url:
                raise ClientConfigurationError(
                    "Ollama native profiles require a base_url such as "
                    "http://127.0.0.1:11434."
                )
            root = str(base_url).rstrip("/")
            if root.endswith("/v1"):
                root = root[:-3]
            self._ollama_url = f"{root}/api/chat"
            return

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ClientConfigurationError(
                "The openai package is not installed. Run: pip install -e ."
            ) from exc

        self.client = OpenAI(
            api_key=api_key or "local",
            base_url=str(base_url) if base_url else None,
            timeout=self.timeout_seconds,
        )

    def _complete_openai(self, messages: list[dict[str, str]]) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        request[self.max_tokens_field] = self.max_output_tokens
        if self.extra_body:
            request["extra_body"] = self.extra_body

        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message
        content = message.content
        text = "" if content is None else str(content).strip()
        if text:
            return text

        model_extra = getattr(message, "model_extra", None) or {}
        reasoning = (
            getattr(message, "reasoning_content", None)
            or model_extra.get("reasoning_content")
        )
        thinking = getattr(message, "thinking", None) or model_extra.get("thinking")
        reasoning_chars = len(str(reasoning)) if reasoning else 0
        thinking_chars = len(str(thinking)) if thinking else 0
        finish_reason = getattr(choice, "finish_reason", None)
        raise RuntimeError(
            "Model returned empty final content "
            f"(finish_reason={finish_reason!r}, "
            f"reasoning_chars={reasoning_chars}, "
            f"thinking_chars={thinking_chars}, "
            f"max_output_tokens={self.max_output_tokens}). "
            "The final answer was not available for label parsing."
        )

    def _complete_ollama_native(self, messages: list[dict[str, str]]) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self.max_output_tokens,
            },
        }
        if self.temperature is not None:
            body["options"]["temperature"] = self.temperature
        if self._ollama_think is not None:
            body["think"] = self._ollama_think
        if self._ollama_structured_output:
            body["format"] = _LABEL_SCHEMA
        if self._ollama_keep_alive is not None:
            body["keep_alive"] = self._ollama_keep_alive

        request = urllib.request.Request(
            self._ollama_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama native HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama native request failed: {exc.reason}") from exc

        message = payload.get("message") or {}
        content = message.get("content")
        text = "" if content is None else str(content).strip()
        if text:
            return text

        thinking = message.get("thinking")
        done_reason = payload.get("done_reason")
        eval_count = payload.get("eval_count")
        raise RuntimeError(
            "Ollama returned empty final content "
            f"(done_reason={done_reason!r}, "
            f"thinking_chars={len(str(thinking)) if thinking else 0}, "
            f"eval_count={eval_count!r}, "
            f"think={self._ollama_think!r}, "
            f"max_output_tokens={self.max_output_tokens})."
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                if self._native_ollama:
                    return self._complete_ollama_native(messages)
                return self._complete_openai(messages)
            except Exception as exc:  # pragma: no cover - provider/network dependent
                last_error = exc
                if attempt + 1 >= self.retries:
                    break
                time.sleep(min(2**attempt, 8))

        raise RuntimeError(
            f"Model request failed after {self.retries} attempts: {last_error}"
        )
