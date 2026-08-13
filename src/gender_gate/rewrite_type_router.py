from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cache import SQLiteCache, request_key
from .clients import OpenAICompatibleClient
from .prompts import load_text, render_prompt
from .rewrite_judge import RewriteType, normalize_rewrite_type
from .schema import DatasetItem


@dataclass
class RewriteTypePrediction:
    id: str
    text: str
    predicted_type: RewriteType | None
    raw_output: str
    model: str
    prompt_version: str
    latency_ms: int
    cache_hit: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_rewrite_type_output(raw_output: str) -> RewriteType:
    """Parse a compact rewrite-type label with light provider robustness."""
    text = raw_output.strip()
    if not text:
        raise ValueError("Empty rewrite-type output")

    # Accept JSON as a harmless fallback, even though the prompt asks for a bare label.
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("rewrite_type", "label", "type"):
            value = payload.get(key)
            if value is not None:
                return normalize_rewrite_type(str(value))

    # Prefer exact labels if the model adds a tiny amount of extra text.
    upper = cleaned.upper().replace("-", "_")
    labels = re.findall(
        r"(?<![A-Z_])(LOCAL_REPAIR|PROPOSITION_RECONSTRUCTION)(?![A-Z_])",
        upper,
    )
    unique = list(dict.fromkeys(labels))
    if len(unique) == 1:
        return normalize_rewrite_type(unique[0])
    if len(unique) > 1:
        raise ValueError(f"Ambiguous rewrite-type output: {raw_output!r}")

    # Final compatibility fallback for short aliases.
    if len(cleaned.split()) <= 4:
        return normalize_rewrite_type(cleaned)
    raise ValueError(f"Could not parse rewrite type from: {raw_output!r}")


class RewriteTypeRouter:
    """Simple LLM binary router for LOCAL_REPAIR vs PROPOSITION_RECONSTRUCTION."""

    def __init__(
        self,
        model_config: dict[str, Any],
        router_config: dict[str, Any],
        project_root: Path,
        *,
        client: Any | None = None,
    ):
        self.project_root = project_root
        prompt_value = str(router_config["prompt"])
        prompt_path = Path(prompt_value).expanduser()
        self.prompt_path = (
            prompt_path if prompt_path.is_absolute() else project_root / prompt_path
        )
        self.template = load_text(self.prompt_path)
        self.client = client or OpenAICompatibleClient(model_config, router_config)

        cache_value = Path(
            str(router_config.get("cache_db", ".cache/llm_cache.sqlite"))
        ).expanduser()
        cache_path = (
            cache_value if cache_value.is_absolute() else project_root / cache_value
        )
        self.cache = SQLiteCache(cache_path)
        self.prompt_version = self.prompt_path.stem

    @property
    def model(self) -> str:
        return str(self.client.model)

    def predict(self, item: DatasetItem) -> RewriteTypePrediction:
        if item.label != "POSITIVE":
            raise ValueError(
                f"Rewrite-type router accepts POSITIVE items only: "
                f"{item.id}={item.label}"
            )

        prompt = render_prompt(self.template, item.text)
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "task": "rewrite_type_router_v01",
            "provider": getattr(self.client, "provider", "unknown"),
            "model": self.client.model,
            "temperature": getattr(self.client, "temperature", None),
            "max_output_tokens": getattr(self.client, "max_output_tokens", None),
            "messages": messages,
        }
        key = request_key(payload)

        started = time.perf_counter()
        cached = self.cache.get(key)
        cache_hit = cached is not None
        raw_output = cached or ""
        predicted_type: RewriteType | None = None
        error: str | None = None

        if cached is None:
            try:
                raw_output = self.client.complete(messages)
                self.cache.put(key, raw_output)
            except Exception as exc:  # pragma: no cover - provider/network dependent
                error = f"{type(exc).__name__}: {exc}"

        if error is None:
            try:
                predicted_type = parse_rewrite_type_output(raw_output)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        latency_ms = int((time.perf_counter() - started) * 1000)
        return RewriteTypePrediction(
            id=item.id,
            text=item.text,
            predicted_type=predicted_type,
            raw_output=raw_output,
            model=self.model,
            prompt_version=self.prompt_version,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            error=error,
        )
