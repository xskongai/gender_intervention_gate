from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cache import SQLiteCache, request_key
from .clients import OpenAICompatibleClient
from .prompts import load_text
from .rewrite_judge import RewriteType, normalize_rewrite_type


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Rewrite-type judge output does not contain a JSON object")
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Rewrite-type judge output JSON must be an object")
    return payload


def parse_rewrite_type_output(raw_output: str) -> tuple[RewriteType, str]:
    """Parse the independent rewrite-type judge output."""
    payload = _extract_json_object(raw_output)
    raw_type = payload.get("rewrite_type")
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise ValueError("Missing rewrite_type")
    rewrite_type = normalize_rewrite_type(raw_type)
    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    return rewrite_type, reason.strip()


def render_rewrite_type_prompt(
    template: str,
    *,
    item_id: str,
    text: str,
) -> str:
    """Render from ORIGINAL text only; candidate rewrites are intentionally absent."""
    return template.replace("{{ID}}", item_id).replace("{{TEXT}}", text)


@dataclass
class RewriteTypeJudgment:
    id: str
    text: str
    rewrite_type: RewriteType | None
    reason: str | None
    raw_output: str
    model: str
    prompt_version: str
    latency_ms: int
    cache_hit: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RewriteTypeJudge:
    """Independent strategy judgment using only the original sentence."""

    def __init__(
        self,
        model_config: dict[str, Any],
        experiment_config: dict[str, Any],
        project_root: Path,
    ):
        prompt_value = str(experiment_config["prompt"])
        prompt_path = Path(prompt_value).expanduser()
        self.prompt_path = (
            prompt_path if prompt_path.is_absolute() else project_root / prompt_path
        )
        self.template = load_text(self.prompt_path)
        self.client = OpenAICompatibleClient(model_config, experiment_config)
        cache_value = Path(str(experiment_config["cache_db"])).expanduser()
        cache_path = (
            cache_value if cache_value.is_absolute() else project_root / cache_value
        )
        self.cache = SQLiteCache(cache_path)
        self.prompt_version = self.prompt_path.stem

    @property
    def model(self) -> str:
        return self.client.model

    def judge(
        self,
        *,
        item_id: str,
        text: str,
    ) -> RewriteTypeJudgment:
        prompt = render_rewrite_type_prompt(
            self.template,
            item_id=item_id,
            text=text,
        )
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "task": f"rewrite_type_judge:{self.prompt_version}",
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
        raw_output = cached or ""
        error: str | None = None

        if cached is None:
            try:
                raw_output = self.client.complete(messages)
                self.cache.put(key, raw_output)
            except Exception as exc:  # pragma: no cover - provider dependent
                error = f"{type(exc).__name__}: {exc}"

        rewrite_type: RewriteType | None = None
        reason: str | None = None
        if error is None:
            try:
                rewrite_type, reason = parse_rewrite_type_output(raw_output)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        latency_ms = int((time.perf_counter() - started) * 1000)
        return RewriteTypeJudgment(
            id=item_id,
            text=text,
            rewrite_type=rewrite_type,
            reason=reason,
            raw_output=raw_output,
            model=self.client.model,
            prompt_version=self.prompt_version,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            error=error,
        )
