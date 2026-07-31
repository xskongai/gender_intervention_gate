from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .cache import SQLiteCache, request_key
from .clients import OpenAICompatibleClient
from .prompts import load_text

RewriteType = Literal["LOCAL_REPAIR", "PROPOSITION_RECONSTRUCTION"]
VALID_REWRITE_TYPES = {"LOCAL_REPAIR", "PROPOSITION_RECONSTRUCTION"}


def normalize_rewrite_type(value: str) -> RewriteType:
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "LOCAL": "LOCAL_REPAIR",
        "LOCAL_REWRITE": "LOCAL_REPAIR",
        "LOCAL_REPAIR": "LOCAL_REPAIR",
        "RECONSTRUCTION": "PROPOSITION_RECONSTRUCTION",
        "PROPOSITION": "PROPOSITION_RECONSTRUCTION",
        "PROPOSITION_REFRAME": "PROPOSITION_RECONSTRUCTION",
        "PROPOSITION_REFRAMING": "PROPOSITION_RECONSTRUCTION",
        "PROPOSITION_RECONSTRUCTION": "PROPOSITION_RECONSTRUCTION",
    }
    result = aliases.get(normalized)
    if result is None:
        raise ValueError(
            f"Unknown rewrite_type {value!r}; expected LOCAL_REPAIR or "
            "PROPOSITION_RECONSTRUCTION"
        )
    return result  # type: ignore[return-value]


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge output does not contain a JSON object")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Judge output JSON must be an object")
    return parsed


def _parse_dimension(
    payload: dict[str, Any], name: str, *, required: bool
) -> tuple[int | None, str | None]:
    value = payload.get(name)
    if value is None:
        if required:
            raise ValueError(f"Missing required judge dimension: {name}")
        return None, None
    if not isinstance(value, dict):
        raise ValueError(f"Judge dimension {name} must be an object or null")
    score = value.get("score")
    if isinstance(score, str) and score.strip().isdigit():
        score = int(score.strip())
    if not isinstance(score, int) or score not in {1, 2, 3}:
        raise ValueError(f"Judge dimension {name}.score must be 1, 2, or 3")
    reason = value.get("reason")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        reason = str(reason)
    return score, reason.strip()


@dataclass(frozen=True)
class ParsedJudgeScores:
    debiasing_score: int
    debiasing_reason: str
    naturalness_score: int
    naturalness_reason: str
    fidelity_score: int | None
    fidelity_reason: str | None
    no_added_facts_score: int | None
    no_added_facts_reason: str | None
    relevance_score: int | None
    relevance_reason: str | None


def parse_judge_output(
    raw_output: str, rewrite_type: RewriteType
) -> ParsedJudgeScores:
    """Parse v02 Judge output while retaining v01 local-output compatibility."""
    payload = _extract_json_object(raw_output)
    debiasing_score, debiasing_reason = _parse_dimension(
        payload, "debiasing", required=True
    )
    naturalness_score, naturalness_reason = _parse_dimension(
        payload, "naturalness", required=True
    )

    fidelity_score: int | None = None
    fidelity_reason: str | None = None
    facts_score: int | None = None
    facts_reason: str | None = None

    if rewrite_type == "LOCAL_REPAIR":
        # v02 uses fidelity. v01 used no_added_facts; accept the legacy field so
        # older configs and cached outputs remain readable.
        has_fidelity = payload.get("fidelity") is not None
        has_legacy_facts = payload.get("no_added_facts") is not None
        if has_fidelity and has_legacy_facts:
            raise ValueError(
                "LOCAL_REPAIR judge output must provide fidelity or "
                "no_added_facts, not both"
            )
        if has_fidelity:
            fidelity_score, fidelity_reason = _parse_dimension(
                payload, "fidelity", required=True
            )
        else:
            facts_score, facts_reason = _parse_dimension(
                payload, "no_added_facts", required=True
            )

        relevance_score, relevance_reason = _parse_dimension(
            payload, "relevance", required=False
        )
        if relevance_score is not None:
            raise ValueError("LOCAL_REPAIR judge output must set relevance to null")
    else:
        relevance_score, relevance_reason = _parse_dimension(
            payload, "relevance", required=True
        )
        fidelity_score, fidelity_reason = _parse_dimension(
            payload, "fidelity", required=False
        )
        facts_score, facts_reason = _parse_dimension(
            payload, "no_added_facts", required=False
        )
        if fidelity_score is not None or facts_score is not None:
            raise ValueError(
                "PROPOSITION_RECONSTRUCTION judge output must set fidelity and "
                "no_added_facts to null"
            )

    assert debiasing_score is not None
    assert naturalness_score is not None
    return ParsedJudgeScores(
        debiasing_score=debiasing_score,
        debiasing_reason=debiasing_reason or "",
        naturalness_score=naturalness_score,
        naturalness_reason=naturalness_reason or "",
        fidelity_score=fidelity_score,
        fidelity_reason=fidelity_reason,
        no_added_facts_score=facts_score,
        no_added_facts_reason=facts_reason,
        relevance_score=relevance_score,
        relevance_reason=relevance_reason,
    )


def render_judge_prompt(
    template: str,
    *,
    item_id: str,
    rewrite_type: RewriteType,
    text: str,
    output: str,
) -> str:
    return (
        template.replace("{{ID}}", item_id)
        .replace("{{REWRITE_TYPE}}", rewrite_type)
        .replace("{{TEXT}}", text)
        .replace("{{OUTPUT}}", output)
    )


@dataclass
class RewriteJudgePrediction:
    id: str
    text: str
    output: str
    rewrite_type: RewriteType
    raw_output: str
    debiasing_score: int | None
    debiasing_reason: str | None
    naturalness_score: int | None
    naturalness_reason: str | None
    fidelity_score: int | None
    fidelity_reason: str | None
    no_added_facts_score: int | None
    no_added_facts_reason: str | None
    relevance_score: int | None
    relevance_reason: str | None
    model: str
    prompt_version: str
    latency_ms: int
    cache_hit: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RewriteQualityJudge:
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
        rewrite_type: RewriteType,
        text: str,
        output: str,
    ) -> tuple[str, int, bool, str | None]:
        prompt = render_judge_prompt(
            self.template,
            item_id=item_id,
            rewrite_type=rewrite_type,
            text=text,
            output=output,
        )
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "task": f"rewrite_quality_judge:{self.prompt_version}",
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


class MockPerfectRewriteJudge:
    model = "mock-perfect-judge"
    prompt_version = "mock_perfect_judge_v02"

    def judge(
        self,
        *,
        item_id: str,
        rewrite_type: RewriteType,
        text: str,
        output: str,
    ) -> tuple[str, int, bool, str | None]:
        if rewrite_type == "LOCAL_REPAIR":
            payload = {
                "debiasing": {"score": 3, "reason": "mock"},
                "naturalness": {"score": 3, "reason": "mock"},
                "fidelity": {"score": 3, "reason": "mock"},
                "relevance": None,
            }
        else:
            payload = {
                "debiasing": {"score": 3, "reason": "mock"},
                "naturalness": {"score": 3, "reason": "mock"},
                "fidelity": None,
                "relevance": {"score": 3, "reason": "mock"},
            }
        return json.dumps(payload, ensure_ascii=False), 0, False, None


def build_judge_prediction(
    row: dict[str, Any],
    judge: RewriteQualityJudge | MockPerfectRewriteJudge,
) -> RewriteJudgePrediction:
    rewrite_type = normalize_rewrite_type(str(row["rewrite_type"]))
    item_id = str(row["id"])
    text = str(row["text"])
    output = str(row["output"])
    raw_output, latency_ms, cache_hit, error = judge.judge(
        item_id=item_id,
        rewrite_type=rewrite_type,
        text=text,
        output=output,
    )

    parsed: ParsedJudgeScores | None = None
    if error is None:
        try:
            parsed = parse_judge_output(raw_output, rewrite_type)
        except Exception as exc:
            error = f"JUDGE_PARSE_ERROR: {exc}"

    return RewriteJudgePrediction(
        id=item_id,
        text=text,
        output=output,
        rewrite_type=rewrite_type,
        raw_output=raw_output,
        debiasing_score=None if parsed is None else parsed.debiasing_score,
        debiasing_reason=None if parsed is None else parsed.debiasing_reason,
        naturalness_score=None if parsed is None else parsed.naturalness_score,
        naturalness_reason=None if parsed is None else parsed.naturalness_reason,
        fidelity_score=None if parsed is None else parsed.fidelity_score,
        fidelity_reason=None if parsed is None else parsed.fidelity_reason,
        no_added_facts_score=None
        if parsed is None
        else parsed.no_added_facts_score,
        no_added_facts_reason=None
        if parsed is None
        else parsed.no_added_facts_reason,
        relevance_score=None if parsed is None else parsed.relevance_score,
        relevance_reason=None if parsed is None else parsed.relevance_reason,
        model=judge.model,
        prompt_version=judge.prompt_version,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        error=error,
    )
