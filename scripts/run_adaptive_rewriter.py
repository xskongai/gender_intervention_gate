#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from gender_gate.adaptive_rewriter import (
    AdaptivePolicy,
    AdaptiveRewriterRunner,
    FeedbackRepairRewriter,
    final_prediction_dict,
    load_rewrite_type_map,
    summarize_trajectories,
)
from gender_gate.data import load_items
from gender_gate.rewrite_judge import RewriteQualityJudge
from gender_gate.rewriter import PositiveTextRewriter


def resolve_path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_rows(trajectories: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for candidate in trajectory.candidates:
            rows.append(
                {
                    "id": trajectory.id,
                    "rewrite_type": trajectory.rewrite_type,
                    "round": candidate.round_id,
                    "route": candidate.route,
                    "output": candidate.output,
                    "quality_score": candidate.quality_score,
                    "accepted": candidate.accepted,
                    "verdict": candidate.verdict,
                    "debiasing_score": candidate.debiasing_score,
                    "debiasing_reason": candidate.debiasing_reason,
                    "naturalness_score": candidate.naturalness_score,
                    "naturalness_reason": candidate.naturalness_reason,
                    "type_specific_name": candidate.type_specific_name,
                    "type_specific_score": candidate.type_specific_score,
                    "type_specific_reason": candidate.type_specific_reason,
                    "generator_error": candidate.generator_error,
                    "verifier_error": candidate.verifier_error,
                    "generator_latency_ms": candidate.generator_latency_ms,
                    "verifier_latency_ms": candidate.verifier_latency_ms,
                    "generator_cache_hit": candidate.generator_cache_hit,
                    "verifier_cache_hit": candidate.verifier_cache_hit,
                    "edit_distance": candidate.edit_distance,
                    "is_best_so_far": candidate.is_best_so_far,
                    "selected": candidate.round_id == trajectory.selected_round,
                }
            )
    return rows


def summary_markdown(
    metrics: dict[str, Any],
    *,
    model: str,
    verifier_model: str,
    threshold: float,
    max_rounds: int,
) -> str:
    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    def number(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2f}"

    lines = [
        "# Feedback-Guided Adaptive Rewriter",
        "",
        f"- Rewriter model: `{model}`",
        f"- Online verifier: `{verifier_model}`",
        f"- Acceptance threshold: `{threshold:.2f}`",
        f"- Maximum rounds: `{max_rounds}`",
        "",
        "## Overall",
        "",
        f"- Count: {metrics['count']}",
        f"- Initial quality: {number(metrics['initial_quality'])}",
        f"- Final quality: {number(metrics['final_quality'])}",
        f"- Mean quality gain: {number(metrics['mean_quality_gain'])}",
        f"- Initial pass rate: {pct(metrics['initial_pass_rate'])}",
        f"- Final pass rate: {pct(metrics['final_pass_rate'])}",
        f"- Refinement trigger rate: {pct(metrics['refinement_trigger_rate'])}",
        f"- Rescue at round 2: {pct(metrics['rescue_at_round_2_rate'])}",
        f"- Rescue at round 3+: {pct(metrics['rescue_at_round_3_plus_rate'])}",
        f"- Average rounds: {metrics['average_rounds']:.3f}",
        f"- Selected candidate was not last: {pct(metrics['selected_not_last_rate'])}",
        f"- Trajectory regression rate: {pct(metrics['trajectory_regression_rate'])}",
        "",
        "## Cost",
        "",
        f"- Generation calls: {metrics['generation_calls']}",
        f"- Verifier calls: {metrics['verifier_calls']}",
        f"- Generator errors: {metrics['generator_error_count']}",
        f"- Verifier errors: {metrics['verifier_error_count']}",
        f"- Generator latency: {metrics['total_generator_latency_seconds']:.2f}s",
        f"- Verifier latency: {metrics['total_verifier_latency_seconds']:.2f}s",
        "",
        "## Important evaluation note",
        "",
        "The online verifier guides generation and must not be the only final evaluator. "
        "For paper results, evaluate `predictions.jsonl` with an independent judge "
        "or blinded human review.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run feedback-guided adaptive positive rewriting: generate, verify, "
            "route failure-specific feedback, repair for up to N rounds, and select "
            "the best candidate in the trajectory."
        )
    )
    parser.add_argument(
        "--config", default="configs/adaptive_rewriter/adaptive_rewriter_v01.yaml"
    )
    parser.add_argument("--split")
    parser.add_argument("--model-key", help="Rewriter model key override")
    parser.add_argument("--model", help="Exact rewriter model id override")
    parser.add_argument("--verifier-model-key")
    parser.add_argument("--verifier-model")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--name")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    config_path = resolve_path(root, args.config)
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if args.split:
        config["split"] = args.split
    if args.name:
        config["name"] = args.name
    if args.concurrency is not None:
        if args.concurrency <= 0:
            raise ValueError("--concurrency must be positive")
        config["concurrency"] = args.concurrency

    rewriter_config = dict(config["rewriter"])
    verifier_config = dict(config["verifier"])
    adaptive_config = dict(config.get("adaptive") or {})
    if args.model_key:
        rewriter_config["model_key"] = args.model_key
    if args.model:
        rewriter_config["model"] = args.model
    if args.verifier_model_key:
        verifier_config["model_key"] = args.verifier_model_key
    if args.verifier_model:
        verifier_config["model"] = args.verifier_model
    if args.threshold is not None:
        adaptive_config["threshold"] = args.threshold
    if args.max_rounds is not None:
        adaptive_config["max_rounds"] = args.max_rounds
    if args.max_output_tokens is not None:
        if args.max_output_tokens <= 0:
            raise ValueError("--max-output-tokens must be positive")
        rewriter_config["max_output_tokens"] = args.max_output_tokens

    split_path = resolve_path(root, str(config["split"]))
    items = load_items(split_path)
    non_positive = [item.id for item in items if item.label != "POSITIVE"]
    if non_positive:
        raise ValueError(
            "Adaptive rewriter requires a POSITIVE-only split; "
            f"found {len(non_positive)} non-positive items"
        )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        items = items[: args.limit]

    models_path = resolve_path(root, str(config["models_config"]))
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]
    rewriter_model_key = str(rewriter_config["model_key"])
    verifier_model_key = str(verifier_config["model_key"])
    if rewriter_model_key not in models:
        raise KeyError(f"Unknown rewriter model_key: {rewriter_model_key}")
    if verifier_model_key not in models:
        raise KeyError(f"Unknown verifier model_key: {verifier_model_key}")

    initial_request = dict(rewriter_config)
    initial_request["prompt"] = str(rewriter_config["initial_prompt"])
    initial_rewriter = PositiveTextRewriter(
        models[rewriter_model_key], initial_request, root
    )
    repair_rewriter = FeedbackRepairRewriter(
        models[rewriter_model_key],
        rewriter_config,
        root,
        dict(config["repair_prompts"]),
    )
    verifier = RewriteQualityJudge(models[verifier_model_key], verifier_config, root)

    hard = dict(adaptive_config.get("hard_min_scores") or {})
    policy = AdaptivePolicy(
        threshold=float(adaptive_config.get("threshold", 80.0)),
        max_rounds=int(adaptive_config.get("max_rounds", 3)),
        debiasing_min_score=int(hard.get("debiasing", 1)),
        naturalness_min_score=int(hard.get("naturalness", 1)),
        type_specific_min_score=int(hard.get("type_specific", 1)),
    )
    type_map_path = resolve_path(root, str(config["type_map"]))
    rewrite_types = load_rewrite_type_map(type_map_path)
    runner = AdaptiveRewriterRunner(
        initial_rewriter=initial_rewriter,
        repair_rewriter=repair_rewriter,
        verifier=verifier,
        rewrite_types=rewrite_types,
        policy=policy,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = args.name or config.get("name") or "adaptive_rewriter_v01"
    run_dir = root / "runs" / f"{timestamp}_adaptive_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    effective_config = {
        **config,
        "rewriter": rewriter_config,
        "verifier": verifier_config,
        "adaptive": {
            **adaptive_config,
            "threshold": policy.threshold,
            "max_rounds": policy.max_rounds,
            "hard_min_scores": {
                "debiasing": policy.debiasing_min_score,
                "naturalness": policy.naturalness_min_score,
                "type_specific": policy.type_specific_min_score,
            },
        },
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(effective_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    shutil.copy2(
        resolve_path(root, str(rewriter_config["initial_prompt"])),
        run_dir / "initial_prompt.txt",
    )
    repair_dir = run_dir / "repair_prompts"
    repair_dir.mkdir()
    for route, value in dict(config["repair_prompts"]).items():
        shutil.copy2(resolve_path(root, str(value)), repair_dir / f"{route}.txt")
    shutil.copy2(
        resolve_path(root, str(verifier_config["prompt"])),
        run_dir / "verifier_prompt.txt",
    )

    trajectories: list[Any] = []
    concurrency = int(config.get("concurrency", 1))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(runner.run_item, item): item for item in items}
        for completed, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                trajectory = future.result()
            except Exception as exc:
                raise RuntimeError(f"Adaptive run failed for {item.id}: {exc}") from exc
            trajectories.append(trajectory)
            selected = trajectory.selected_round
            score = (
                "N/A"
                if trajectory.final_quality_score is None
                else f"{trajectory.final_quality_score:.1f}"
            )
            route_trace = "→".join(candidate.route for candidate in trajectory.candidates)
            print(
                f"[{completed:4d}/{len(items)}] {trajectory.id:<12} "
                f"rounds={len(trajectory.candidates)} selected=r{selected} "
                f"score={score:>5} routes={route_trace:<28} "
                f"{trajectory.final_output}",
                flush=True,
            )

    order = {item.id: index for index, item in enumerate(items)}
    trajectories.sort(key=lambda trajectory: order[trajectory.id])
    trajectory_dicts = [trajectory.to_dict() for trajectory in trajectories]
    predictions = [
        final_prediction_dict(
            trajectory,
            model=initial_rewriter.model,
            prompt_version="adaptive_verify_repair_v01",
        )
        for trajectory in trajectories
    ]
    write_jsonl(run_dir / "trajectories.jsonl", trajectory_dicts)
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    write_csv(run_dir / "candidates.csv", candidate_rows(trajectories))
    write_csv(run_dir / "predictions.csv", predictions)

    metrics = summarize_trajectories(trajectories)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(
        summary_markdown(
            metrics,
            model=initial_rewriter.model,
            verifier_model=verifier.model,
            threshold=policy.threshold,
            max_rounds=policy.max_rounds,
        ),
        encoding="utf-8",
    )

    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "task": "feedback_guided_adaptive_positive_rewriter",
        "method": "generate_verify_route_repair_select",
        "model_key": rewriter_model_key,
        "model": initial_rewriter.model,
        "verifier_model_key": verifier_model_key,
        "verifier_model": verifier.model,
        "count": len(items),
        "split": str(split_path),
        "split_sha256": sha256_file(split_path),
        "type_map": str(type_map_path),
        "type_map_sha256": sha256_file(type_map_path),
        "threshold": policy.threshold,
        "max_rounds": policy.max_rounds,
        "metrics": metrics,
        "evaluation_warning": (
            "The online verifier guided candidate generation. Use an independent "
            "judge or blinded human evaluation for final reported quality."
        ),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Run directory: {run_dir}")
    print(f"Initial quality: {metrics['initial_quality']}")
    print(f"Final quality: {metrics['final_quality']}")
    print(f"Mean gain: {metrics['mean_quality_gain']}")
    print(f"Initial pass rate: {metrics['initial_pass_rate']:.4f}")
    print(f"Final pass rate: {metrics['final_pass_rate']:.4f}")
    print(f"Average rounds: {metrics['average_rounds']:.4f}")


if __name__ == "__main__":
    main()
