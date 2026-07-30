from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .cache import SQLiteCache, request_key
from .clients import OpenAICompatibleClient
from .parser import parse_label
from .prompts import load_text, render_examples, render_prompt
from .schema import DatasetItem, Prediction


class BinaryClassifier:
    def __init__(
        self,
        model_config: dict[str, Any],
        experiment_config: dict[str, Any],
        project_root: Path,
    ):
        self.project_root = project_root
        self.prompt_path = project_root / experiment_config["prompt"]
        examples_value = experiment_config.get("examples")
        self.examples_path = project_root / examples_value if examples_value else None

        self.template = load_text(self.prompt_path)
        self.examples = render_examples(self.examples_path)
        self.client = OpenAICompatibleClient(model_config, experiment_config)
        self.cache = SQLiteCache(project_root / experiment_config["cache_db"])
        self.prompt_version = self.prompt_path.stem

    def predict(self, item: DatasetItem) -> Prediction:
        prompt = render_prompt(self.template, item.text, self.examples)
        messages = [{"role": "user", "content": prompt}]
        payload = {
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
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        latency_ms = int((time.perf_counter() - started) * 1000)
        predicted = None if error else parse_label(raw_output)
        if error is None and predicted is None:
            error = "FORMAT_ERROR: output did not contain exactly one valid label"

        return Prediction(
            id=item.id,
            text=item.text,
            gold=item.label,
            predicted=predicted,
            raw_output=raw_output,
            model=self.client.model,
            prompt_version=self.prompt_version,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            error=error,
            meta=item.meta,
        )
