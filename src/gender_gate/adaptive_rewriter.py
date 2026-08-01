from __future__ import annotations

import csv
import difflib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from .cache import SQLiteCache, request_key
from .clients import OpenAICompatibleClient
from .prompts import load_text
from .rewrite_judge import (
    RewriteQualityJudge,
    RewriteType,
    build_judge_prediction,
    normalize_rewrite_type,
)
from .rewrite_judge_metrics import score_judgment
from .rewriter import PositiveTextRewriter, normalize_rewrite_text
from .schema import DatasetItem

RepairRoute = Literal[
    "debiasing",
    "fidelity",
    "relevance",
    "naturalness",
    "generic",
]


@dataclass(frozen=True)
class AdaptivePolicy:
    threshold: float = 80.0
    max_rounds: int = 3
    debiasing_min_score: int = 1
    naturalness_min_score: int = 1
    type_specific_min_score: int = 1

    def validate(self) -> None:
        if not 0 <= self.threshold <= 100:
            raise ValueError("threshold must be between 0 and 100")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        for name, value in {
            "debiasing_min_score": self.debiasing_min_score,
            "naturalness_min_score": self.naturalness_min_score,
            "type_specific_min_score": self.type_specific_min_score,
        }.items():
            if value not in {1, 2, 3}:
                raise ValueError(f"{name} must be 1, 2, or 3")


@dataclass
class AdaptiveCandidate:
    round_id: int
    route: str
    raw_output: str
    output: str
    generator_prompt_version: str
    generator_latency_ms: int
    generator_cache_hit: bool
    generator_error: str | None
    verifier_raw_output: str
    verifier_prompt_version: str
    verifier_latency_ms: int
    verifier_cache_hit: bool
    verifier_error: str | None
    debiasing_score: int | None
    debiasing_reason: str | None
    naturalness_score: int | None
    naturalness_reason: str | None
    type_specific_name: str | None
    type_specific_score: int | None
    type_specific_reason: str | None
    quality_score: float | None
    verdict: str
    accepted: bool
    edit_distance: int
    is_best_so_far: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveTrajectory:
    id: str
    text: str
    gold: str
    rewrite_type: RewriteType
    candidates: list[AdaptiveCandidate] = field(default_factory=list)
    selected_round: int = 1
    final_output: str = ""
    final_quality_score: float | None = None
    initial_quality_score: float | None = None
    first_pass_round: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "gold": self.gold,
            "rewrite_type": self.rewrite_type,
            "selected_round": self.selected_round,
            "final_output": self.final_output,
            "final_quality_score": self.final_quality_score,
            "initial_quality_score": self.initial_quality_score,
            "first_pass_round": self.first_pass_round,
            "rounds_used": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "meta": self.meta,
        }


def load_rewrite_type_map(path: Path) -> dict[str, RewriteType]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, RewriteType] = {}
    for row in rows:
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            continue
        result[item_id] = normalize_rewrite_type(str(row["rewrite_type"]))
    return result


def char_edit_distance(left: str, right: str) -> int:
    left = normalize_rewrite_text(left)
    right = normalize_rewrite_text(right)
    matcher = difflib.SequenceMatcher(a=left, b=right)
    unchanged = sum(block.size for block in matcher.get_matching_blocks())
    return max(len(left), len(right)) - unchanged


def _type_specific_fields(
    scored: dict[str, Any], rewrite_type: RewriteType
) -> tuple[str, int | None, str | None]:
    if rewrite_type == "LOCAL_REPAIR":
        if scored.get("fidelity_score") is not None:
            return (
                "fidelity",
                int(scored["fidelity_score"]),
                str(scored.get("fidelity_reason") or ""),
            )
        return (
            "no_added_facts",
            None if scored.get("no_added_facts_score") is None else int(scored["no_added_facts_score"]),
            str(scored.get("no_added_facts_reason") or ""),
        )
    return (
        "relevance",
        None if scored.get("relevance_score") is None else int(scored["relevance_score"]),
        str(scored.get("relevance_reason") or ""),
    )


def candidate_passes(candidate: AdaptiveCandidate, policy: AdaptivePolicy) -> bool:
    if candidate.verifier_error is not None or candidate.quality_score is None:
        return False
    if candidate.quality_score < policy.threshold:
        return False
    if (candidate.debiasing_score or 0) < policy.debiasing_min_score:
        return False
    if (candidate.naturalness_score or 0) < policy.naturalness_min_score:
        return False
    if (candidate.type_specific_score or 0) < policy.type_specific_min_score:
        return False
    return True


def choose_repair_route(candidate: AdaptiveCandidate) -> RepairRoute:
    if candidate.generator_error or candidate.verifier_error:
        return "generic"

    dimensions: list[tuple[float, int, RepairRoute]] = []
    if candidate.debiasing_score is not None:
        dimensions.append(((3 - candidate.debiasing_score) * 0.50, 0, "debiasing"))
    if candidate.type_specific_score is not None:
        route: RepairRoute = (
            "fidelity"
            if candidate.type_specific_name in {"fidelity", "no_added_facts"}
            else "relevance"
        )
        dimensions.append(((3 - candidate.type_specific_score) * 0.25, 1, route))
    if candidate.naturalness_score is not None:
        dimensions.append(((3 - candidate.naturalness_score) * 0.25, 2, "naturalness"))

    if not dimensions:
        return "generic"
    # Larger weighted deficit wins. Earlier priority wins ties:
    # debiasing -> type-specific -> naturalness.
    dimensions.sort(key=lambda row: (-row[0], row[1]))
    return dimensions[0][2]


def render_repair_prompt(
    template: str,
    *,
    item: DatasetItem,
    rewrite_type: RewriteType,
    previous: AdaptiveCandidate,
    round_id: int,
    history: list[AdaptiveCandidate],
) -> str:
    history_lines = []
    for candidate in history:
        score = "N/A" if candidate.quality_score is None else f"{candidate.quality_score:.2f}"
        history_lines.append(
            f"Round {candidate.round_id}: score={score}; output={candidate.output}"
        )
    history_text = "\n".join(history_lines) or "无"
    replacements = {
        "{{ID}}": item.id,
        "{{TEXT}}": item.text,
        "{{REWRITE_TYPE}}": rewrite_type,
        "{{ROUND}}": str(round_id),
        "{{PREVIOUS_OUTPUT}}": previous.output,
        "{{QUALITY_SCORE}}": "N/A"
        if previous.quality_score is None
        else f"{previous.quality_score:.2f}",
        "{{DEBIASING_SCORE}}": "N/A"
        if previous.debiasing_score is None
        else str(previous.debiasing_score),
        "{{DEBIASING_REASON}}": previous.debiasing_reason or "无",
        "{{NATURALNESS_SCORE}}": "N/A"
        if previous.naturalness_score is None
        else str(previous.naturalness_score),
        "{{NATURALNESS_REASON}}": previous.naturalness_reason or "无",
        "{{TYPE_SPECIFIC_NAME}}": previous.type_specific_name or "type_specific",
        "{{TYPE_SPECIFIC_SCORE}}": "N/A"
        if previous.type_specific_score is None
        else str(previous.type_specific_score),
        "{{TYPE_SPECIFIC_REASON}}": previous.type_specific_reason or "无",
        "{{HISTORY}}": history_text,
    }
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


class FeedbackRepairRewriter:
    def __init__(
        self,
        model_config: dict[str, Any],
        request_config: dict[str, Any],
        project_root: Path,
        prompt_paths: dict[str, str],
    ):
        self.client = OpenAICompatibleClient(model_config, request_config)
        cache_value = Path(str(request_config["cache_db"])).expanduser()
        cache_path = cache_value if cache_value.is_absolute() else project_root / cache_value
        self.cache = SQLiteCache(cache_path)
        self.templates: dict[str, str] = {}
        self.prompt_versions: dict[str, str] = {}
        for route, value in prompt_paths.items():
            candidate = Path(value).expanduser()
            path = candidate if candidate.is_absolute() else project_root / candidate
            self.templates[route] = load_text(path)
            self.prompt_versions[route] = path.stem

    @property
    def model(self) -> str:
        return self.client.model

    def generate(
        self,
        *,
        item: DatasetItem,
        rewrite_type: RewriteType,
        previous: AdaptiveCandidate,
        round_id: int,
        history: list[AdaptiveCandidate],
        route: RepairRoute,
    ) -> tuple[str, str, int, bool, str | None]:
        template = self.templates.get(route) or self.templates["generic"]
        prompt_version = self.prompt_versions.get(route) or self.prompt_versions["generic"]
        prompt = render_repair_prompt(
            template,
            item=item,
            rewrite_type=rewrite_type,
            previous=previous,
            round_id=round_id,
            history=history,
        )
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "task": f"adaptive_rewriter_repair:{route}:round{round_id}",
            "provider": self.client.provider,
            "model": self.client.model,
            "temperature": self.client.temperature,
            "extra_body": self.client.extra_body,
            "max_output_tokens": self.client.max_output_tokens,
            "max_tokens_field": self.client.max_tokens_field,
            "messages": messages,
        }
        key = request_key(payload)
        started = time.perf_counter()
        cached = self.cache.get(key)
        cache_hit = cached is not None
        raw_output = cached or ""
        error = None
        if cached is None:
            try:
                raw_output = self.client.complete(messages)
                self.cache.put(key, raw_output)
            except Exception as exc:  # pragma: no cover - provider dependent
                error = f"{type(exc).__name__}: {exc}"
        latency_ms = int((time.perf_counter() - started) * 1000)
        return raw_output, prompt_version, latency_ms, cache_hit, error


def evaluate_candidate(
    *,
    item: DatasetItem,
    rewrite_type: RewriteType,
    round_id: int,
    route: str,
    raw_output: str,
    output: str,
    generator_prompt_version: str,
    generator_latency_ms: int,
    generator_cache_hit: bool,
    generator_error: str | None,
    verifier: RewriteQualityJudge,
    policy: AdaptivePolicy,
) -> AdaptiveCandidate:
    row = {
        "id": item.id,
        "text": item.text,
        "output": output,
        "rewrite_type": rewrite_type,
    }
    judgment = build_judge_prediction(row, verifier).to_dict()
    scored = score_judgment(judgment)
    type_name, type_score, type_reason = _type_specific_fields(scored, rewrite_type)
    candidate = AdaptiveCandidate(
        round_id=round_id,
        route=route,
        raw_output=raw_output,
        output=output,
        generator_prompt_version=generator_prompt_version,
        generator_latency_ms=generator_latency_ms,
        generator_cache_hit=generator_cache_hit,
        generator_error=generator_error,
        verifier_raw_output=str(scored.get("raw_output") or ""),
        verifier_prompt_version=str(scored.get("prompt_version") or ""),
        verifier_latency_ms=int(scored.get("latency_ms") or 0),
        verifier_cache_hit=bool(scored.get("cache_hit")),
        verifier_error=scored.get("error"),
        debiasing_score=None
        if scored.get("debiasing_score") is None
        else int(scored["debiasing_score"]),
        debiasing_reason=scored.get("debiasing_reason"),
        naturalness_score=None
        if scored.get("naturalness_score") is None
        else int(scored["naturalness_score"]),
        naturalness_reason=scored.get("naturalness_reason"),
        type_specific_name=type_name,
        type_specific_score=type_score,
        type_specific_reason=type_reason,
        quality_score=None
        if scored.get("quality_score") is None
        else float(scored["quality_score"]),
        verdict=str(scored.get("verdict") or "ERROR"),
        accepted=False,
        edit_distance=char_edit_distance(item.text, output),
    )
    candidate.accepted = candidate_passes(candidate, policy)
    return candidate


def candidate_rank(candidate: AdaptiveCandidate) -> tuple[int, float, int, int, int, int]:
    return (
        int(candidate.accepted),
        -1.0 if candidate.quality_score is None else candidate.quality_score,
        candidate.debiasing_score or 0,
        candidate.type_specific_score or 0,
        candidate.naturalness_score or 0,
        -candidate.edit_distance,
    )


def select_best_candidate(candidates: list[AdaptiveCandidate]) -> AdaptiveCandidate:
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list")
    return max(candidates, key=candidate_rank)


class AdaptiveRewriterRunner:
    def __init__(
        self,
        *,
        initial_rewriter: PositiveTextRewriter,
        repair_rewriter: FeedbackRepairRewriter,
        verifier: RewriteQualityJudge,
        rewrite_types: dict[str, RewriteType],
        policy: AdaptivePolicy,
    ):
        policy.validate()
        self.initial_rewriter = initial_rewriter
        self.repair_rewriter = repair_rewriter
        self.verifier = verifier
        self.rewrite_types = rewrite_types
        self.policy = policy

    def run_item(self, item: DatasetItem) -> AdaptiveTrajectory:
        if item.label != "POSITIVE":
            raise ValueError(f"Adaptive rewriter requires POSITIVE items: {item.id}")
        rewrite_type = self.rewrite_types.get(item.id)
        if rewrite_type is None:
            raise KeyError(f"Missing rewrite type for {item.id}")

        trajectory = AdaptiveTrajectory(
            id=item.id,
            text=item.text,
            gold=item.label,
            rewrite_type=rewrite_type,
            meta=item.meta,
        )

        raw, latency_ms, cache_hit, error = self.initial_rewriter.rewrite(item)
        output = item.text if error else normalize_rewrite_text(raw)
        if not output:
            error = error or "EMPTY_OUTPUT"
            output = item.text
        candidate = evaluate_candidate(
            item=item,
            rewrite_type=rewrite_type,
            round_id=1,
            route="initial",
            raw_output=raw,
            output=output,
            generator_prompt_version=self.initial_rewriter.prompt_version,
            generator_latency_ms=latency_ms,
            generator_cache_hit=cache_hit,
            generator_error=error,
            verifier=self.verifier,
            policy=self.policy,
        )
        candidate.is_best_so_far = True
        trajectory.candidates.append(candidate)
        if candidate.accepted:
            trajectory.first_pass_round = 1

        for round_id in range(2, self.policy.max_rounds + 1):
            previous = trajectory.candidates[-1]
            if previous.accepted:
                break
            route = choose_repair_route(previous)
            raw, prompt_version, latency_ms, cache_hit, error = self.repair_rewriter.generate(
                item=item,
                rewrite_type=rewrite_type,
                previous=previous,
                round_id=round_id,
                history=trajectory.candidates,
                route=route,
            )
            output = item.text if error else normalize_rewrite_text(raw)
            if not output:
                error = error or "EMPTY_OUTPUT"
                output = item.text
            repaired = evaluate_candidate(
                item=item,
                rewrite_type=rewrite_type,
                round_id=round_id,
                route=route,
                raw_output=raw,
                output=output,
                generator_prompt_version=prompt_version,
                generator_latency_ms=latency_ms,
                generator_cache_hit=cache_hit,
                generator_error=error,
                verifier=self.verifier,
                policy=self.policy,
            )
            previous_best = select_best_candidate(trajectory.candidates)
            repaired.is_best_so_far = candidate_rank(repaired) > candidate_rank(previous_best)
            trajectory.candidates.append(repaired)
            if repaired.accepted and trajectory.first_pass_round is None:
                trajectory.first_pass_round = round_id

        selected = select_best_candidate(trajectory.candidates)
        trajectory.selected_round = selected.round_id
        trajectory.final_output = selected.output
        trajectory.final_quality_score = selected.quality_score
        trajectory.initial_quality_score = trajectory.candidates[0].quality_score
        return trajectory


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize_trajectories(trajectories: list[AdaptiveTrajectory]) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("No trajectories to summarize")
    initial_scores = [
        float(t.initial_quality_score)
        for t in trajectories
        if t.initial_quality_score is not None
    ]
    final_scores = [
        float(t.final_quality_score)
        for t in trajectories
        if t.final_quality_score is not None
    ]
    gains = [
        float(t.final_quality_score - t.initial_quality_score)
        for t in trajectories
        if t.final_quality_score is not None and t.initial_quality_score is not None
    ]
    initial_pass = sum(bool(t.candidates and t.candidates[0].accepted) for t in trajectories)
    final_pass = sum(any(c.accepted for c in t.candidates) for t in trajectories)
    triggered = sum(len(t.candidates) > 1 for t in trajectories)
    rescue_2 = sum(t.first_pass_round == 2 for t in trajectories)
    rescue_3_plus = sum(
        t.first_pass_round is not None and t.first_pass_round >= 3 for t in trajectories
    )
    selected_not_last = sum(t.selected_round != len(t.candidates) for t in trajectories)
    regression_trajectories = 0
    for trajectory in trajectories:
        best = -1.0
        regressed = False
        for candidate in trajectory.candidates:
            score = -1.0 if candidate.quality_score is None else candidate.quality_score
            if best >= 0 and score < best:
                regressed = True
            best = max(best, score)
        regression_trajectories += regressed

    all_candidates = [c for t in trajectories for c in t.candidates]
    round_counts: dict[str, int] = {}
    for trajectory in trajectories:
        key = str(len(trajectory.candidates))
        round_counts[key] = round_counts.get(key, 0) + 1

    by_type: dict[str, dict[str, Any]] = {}
    for rewrite_type in ("LOCAL_REPAIR", "PROPOSITION_RECONSTRUCTION"):
        subset = [t for t in trajectories if t.rewrite_type == rewrite_type]
        if not subset:
            continue
        subset_initial = [
            float(t.initial_quality_score)
            for t in subset
            if t.initial_quality_score is not None
        ]
        subset_final = [
            float(t.final_quality_score)
            for t in subset
            if t.final_quality_score is not None
        ]
        by_type[rewrite_type] = {
            "count": len(subset),
            "initial_quality": _mean(subset_initial),
            "final_quality": _mean(subset_final),
            "mean_gain": _mean(
                [
                    float(t.final_quality_score - t.initial_quality_score)
                    for t in subset
                    if t.final_quality_score is not None
                    and t.initial_quality_score is not None
                ]
            ),
            "final_pass_rate": sum(any(c.accepted for c in t.candidates) for t in subset)
            / len(subset),
            "average_rounds": mean(len(t.candidates) for t in subset),
        }

    return {
        "count": len(trajectories),
        "initial_quality": _mean(initial_scores),
        "final_quality": _mean(final_scores),
        "mean_quality_gain": _mean(gains),
        "initial_pass_rate": initial_pass / len(trajectories),
        "final_pass_rate": final_pass / len(trajectories),
        "refinement_trigger_rate": triggered / len(trajectories),
        "rescue_at_round_2_rate": rescue_2 / len(trajectories),
        "rescue_at_round_3_plus_rate": rescue_3_plus / len(trajectories),
        "average_rounds": mean(len(t.candidates) for t in trajectories),
        "round_count_distribution": round_counts,
        "selected_not_last_rate": selected_not_last / len(trajectories),
        "trajectory_regression_rate": regression_trajectories / len(trajectories),
        "generation_calls": len(all_candidates),
        "verifier_calls": len(all_candidates),
        "generator_error_count": sum(c.generator_error is not None for c in all_candidates),
        "verifier_error_count": sum(c.verifier_error is not None for c in all_candidates),
        "generator_cache_hit_rate": sum(c.generator_cache_hit for c in all_candidates)
        / len(all_candidates),
        "verifier_cache_hit_rate": sum(c.verifier_cache_hit for c in all_candidates)
        / len(all_candidates),
        "total_generator_latency_seconds": sum(c.generator_latency_ms for c in all_candidates)
        / 1000.0,
        "total_verifier_latency_seconds": sum(c.verifier_latency_ms for c in all_candidates)
        / 1000.0,
        "by_rewrite_type": by_type,
    }


def final_prediction_dict(
    trajectory: AdaptiveTrajectory,
    *,
    model: str,
    prompt_version: str,
) -> dict[str, Any]:
    selected = next(
        candidate
        for candidate in trajectory.candidates
        if candidate.round_id == trajectory.selected_round
    )
    return {
        "id": trajectory.id,
        "text": trajectory.text,
        "gold": trajectory.gold,
        "raw_output": selected.raw_output,
        "final_output": trajectory.final_output,
        "changed": normalize_rewrite_text(trajectory.final_output)
        != normalize_rewrite_text(trajectory.text),
        "model": model,
        "prompt_version": prompt_version,
        "latency_ms": sum(c.generator_latency_ms for c in trajectory.candidates),
        "cache_hit": all(c.generator_cache_hit for c in trajectory.candidates),
        "error": selected.generator_error,
        "reference_output": trajectory.meta.get("reference_output"),
        "meta": {
            **trajectory.meta,
            "rewrite_type": trajectory.rewrite_type,
            "selected_round": trajectory.selected_round,
            "rounds_used": len(trajectory.candidates),
            "initial_quality_score": trajectory.initial_quality_score,
            "final_quality_score": trajectory.final_quality_score,
            "first_pass_round": trajectory.first_pass_round,
        },
    }
