from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .cache import SQLiteCache, request_key
from .clients import OpenAICompatibleClient
from .prompts import load_text, render_prompt
from .schema import DatasetItem, Label

RewriteMode = Literal["direct", "gated"]


def normalize_text(value: str) -> str:
    """Normalize only surrounding whitespace for exact preservation checks."""
    return value.strip().replace("\r\n", "\n").replace("\r", "\n")


@dataclass
class RewritePrediction:
    id: str
    text: str
    gold: Label
    mode: RewriteMode
    gate_predicted: Label | None
    gate_raw_output: str | None
    gate_error: str | None
    rewrite_called: bool
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


class TextRewriter:
    def __init__(
        self,
        model_config: dict[str, Any],
        experiment_config: dict[str, Any],
        project_root: Path,
    ):
        self.project_root = project_root
        self.prompt_path = project_root / experiment_config["prompt"]
        self.template = load_text(self.prompt_path)
        self.client = OpenAICompatibleClient(model_config, experiment_config)
        self.cache = SQLiteCache(project_root / experiment_config["cache_db"])
        self.prompt_version = self.prompt_path.stem

    @property
    def model(self) -> str:
        return self.client.model

    def rewrite(self, item: DatasetItem) -> tuple[str, int, bool, str | None]:
        prompt = render_prompt(self.template, item.text)
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "task": "gender_inclusive_rewrite",
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
            except Exception as exc:  # pragma: no cover - network/provider dependent
                error = f"{type(exc).__name__}: {exc}"

        latency_ms = int((time.perf_counter() - started) * 1000)
        return raw_output, latency_ms, cache_hit, error


class MockOracleRewriter:
    """Offline smoke-test rewriter. Never use its scores as experiment results."""

    model = "mock-oracle"
    prompt_version = "mock_oracle"

    def rewrite(self, item: DatasetItem) -> tuple[str, int, bool, str | None]:
        reference = item.meta.get("reference_output")
        output = str(reference) if reference else item.text
        return output, 0, False, None


def build_skipped_prediction(
    item: DatasetItem,
    mode: RewriteMode,
    model: str,
    prompt_version: str,
    gate_prediction: dict[str, Any] | None,
    error: str | None = None,
) -> RewritePrediction:
    gate_prediction = gate_prediction or {}
    return RewritePrediction(
        id=item.id,
        text=item.text,
        gold=item.label,
        mode=mode,
        gate_predicted=gate_prediction.get("predicted"),
        gate_raw_output=gate_prediction.get("raw_output"),
        gate_error=gate_prediction.get("error"),
        rewrite_called=False,
        raw_output=item.text,
        final_output=item.text,
        changed=False,
        model=model,
        prompt_version=prompt_version,
        latency_ms=0,
        cache_hit=False,
        error=error,
        reference_output=(
            str(item.meta["reference_output"])
            if item.meta.get("reference_output") is not None
            else None
        ),
        meta=item.meta,
    )


def build_rewrite_prediction(
    item: DatasetItem,
    mode: RewriteMode,
    rewriter: TextRewriter | MockOracleRewriter,
    gate_prediction: dict[str, Any] | None = None,
) -> RewritePrediction:
    raw_output, latency_ms, cache_hit, error = rewriter.rewrite(item)
    final_output = item.text if error else normalize_text(raw_output)
    if not final_output:
        error = error or "EMPTY_OUTPUT"
        final_output = item.text

    gate_prediction = gate_prediction or {}
    return RewritePrediction(
        id=item.id,
        text=item.text,
        gold=item.label,
        mode=mode,
        gate_predicted=gate_prediction.get("predicted"),
        gate_raw_output=gate_prediction.get("raw_output"),
        gate_error=gate_prediction.get("error"),
        rewrite_called=True,
        raw_output=raw_output,
        final_output=final_output,
        changed=normalize_text(final_output) != normalize_text(item.text),
        model=rewriter.model,
        prompt_version=rewriter.prompt_version,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        error=error,
        reference_output=(
            str(item.meta["reference_output"])
            if item.meta.get("reference_output") is not None
            else None
        ),
        meta=item.meta,
    )
