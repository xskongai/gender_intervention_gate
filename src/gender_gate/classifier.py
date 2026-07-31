from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .cache import SQLiteCache, request_key
from .clients import OpenAICompatibleClient
from .deterministic_rules import deterministic_label
from .parser import parse_label
from .prompts import load_text, render_examples, render_prompt
from .schema import DatasetItem, Prediction


class BinaryClassifier:
    """Frozen LLM Gate with an optional deterministic front route."""

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

        rule_first = experiment_config.get("rule_first") or {}
        self.rule_first_enabled = bool(rule_first.get("enabled", False))
        self.ruleset = str(rule_first.get("ruleset", "deterministic_v01"))
        if self.rule_first_enabled and self.ruleset != "deterministic_v01":
            raise ValueError(
                f"Unsupported rule_first.ruleset: {self.ruleset}. "
                "Expected 'deterministic_v01'."
            )

    def predict(self, item: DatasetItem) -> Prediction:
        started = time.perf_counter()

        if self.rule_first_enabled:
            rule_result = deterministic_label(item.text)
            if rule_result is not None:
                latency_ms = int((time.perf_counter() - started) * 1000)
                predicted = rule_result["label"]
                return Prediction(
                    id=item.id,
                    text=item.text,
                    gold=item.label,
                    predicted=predicted,  # type: ignore[arg-type]
                    raw_output=predicted,
                    model=self.client.model,
                    prompt_version=self.prompt_version,
                    latency_ms=latency_ms,
                    cache_hit=False,
                    error=None,
                    meta=item.meta,
                    route="RULE",
                    rule=rule_result["rule"],
                    ruleset=self.ruleset,
                )

        prompt = render_prompt(self.template, item.text, self.examples)
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "provider": self.client.provider,
            "model": self.client.model,
            "temperature": self.client.temperature,
            "extra_body": self.client.extra_body,
            "max_output_tokens": self.client.max_output_tokens,
            "max_tokens_field": self.client.max_tokens_field,
            "messages": messages,
        }
        key = request_key(payload)

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
            route="LLM",
            rule=None,
            ruleset=self.ruleset if self.rule_first_enabled else None,
        )
