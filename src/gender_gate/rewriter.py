from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cache import SQLiteCache, request_key
from .clients import OpenAICompatibleClient
from .prompts import load_text, render_prompt
from .schema import DatasetItem


def normalize_rewrite_text(value: str) -> str:
    """Normalize only surrounding whitespace and line endings."""
    return value.strip().replace("\r\n", "\n").replace("\r", "\n")


@dataclass
class RewriterPrediction:
    id: str
    text: str
    gold: str
    raw_output: str
    final_output: str
    changed: bool
    model: str
    prompt_version: str
    latency_ms: int
    cache_hit: bool
    error: str | None
    reference_output: str | None
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PositiveTextRewriter:
    """Independent rewriter for sentences whose gold label is POSITIVE."""

    def __init__(
        self,
        model_config: dict[str, Any],
        experiment_config: dict[str, Any],
        project_root: Path,
    ):
        self.project_root = project_root
        prompt_value = str(experiment_config["prompt"])
        prompt_path = Path(prompt_value).expanduser()
        self.prompt_path = (
            prompt_path if prompt_path.is_absolute() else project_root / prompt_path
        )
        self.template = load_text(self.prompt_path)
        self.client = OpenAICompatibleClient(model_config, experiment_config)
        cache_value = Path(str(experiment_config["cache_db"])).expanduser()
        cache_path = cache_value if cache_value.is_absolute() else project_root / cache_value
        self.cache = SQLiteCache(cache_path)
        self.prompt_version = self.prompt_path.stem

    @property
    def model(self) -> str:
        return self.client.model

    def rewrite(self, item: DatasetItem) -> tuple[str, int, bool, str | None]:
        if item.label != "POSITIVE":
            raise ValueError(
                f"Independent rewriter accepts POSITIVE items only: {item.id}={item.label}"
            )

        prompt = render_prompt(self.template, item.text)
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "task": "positive_gender_inclusive_rewriter",
            "model": self.client.model,
            "temperature": self.client.temperature,
            "max_output_tokens": self.client.max_output_tokens,
            "max_tokens_field": self.client.max_tokens_field,
            "messages": messages,
        }
        key = request_key(payload)

        started = time.perf_counter()
        cached = self.cache.get(key)
        cache_hit = cached is not None
        error = None
        raw_output = cached or ""

        if cached is None:
            try:
                raw_output = self.client.complete(messages)
                self.cache.put(key, raw_output)
            except Exception as exc:  # pragma: no cover - provider dependent
                error = f"{type(exc).__name__}: {exc}"

        latency_ms = int((time.perf_counter() - started) * 1000)
        return raw_output, latency_ms, cache_hit, error


class MockReferenceRewriter:
    """Offline plumbing smoke test; never use its scores as experiment results."""

    model = "mock-reference"
    prompt_version = "mock_reference"

    def rewrite(self, item: DatasetItem) -> tuple[str, int, bool, str | None]:
        if item.label != "POSITIVE":
            raise ValueError(
                f"Independent rewriter accepts POSITIVE items only: {item.id}={item.label}"
            )
        reference = item.meta.get("reference_output")
        output = str(reference) if reference else item.text
        return output, 0, False, None


def build_rewriter_prediction(
    item: DatasetItem,
    rewriter: PositiveTextRewriter | MockReferenceRewriter,
) -> RewriterPrediction:
    if item.label != "POSITIVE":
        raise ValueError(
            f"Independent rewriter accepts POSITIVE items only: {item.id}={item.label}"
        )

    raw_output, latency_ms, cache_hit, error = rewriter.rewrite(item)
    final_output = item.text if error else normalize_rewrite_text(raw_output)
    if not final_output:
        error = error or "EMPTY_OUTPUT"
        final_output = item.text

    reference = item.meta.get("reference_output")
    return RewriterPrediction(
        id=item.id,
        text=item.text,
        gold=item.label,
        raw_output=raw_output,
        final_output=final_output,
        changed=normalize_rewrite_text(final_output)
        != normalize_rewrite_text(item.text),
        model=rewriter.model,
        prompt_version=rewriter.prompt_version,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        error=error,
        reference_output=str(reference) if reference is not None else None,
        meta=item.meta,
    )
