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

from gender_gate.rewrite_judge import (
    MockPerfectRewriteJudge,
    RewriteQualityJudge,
    SplitRewriteQualityJudge,
    build_judge_prediction,
    normalize_rewrite_type,
)
from gender_gate.rewrite_judge_metrics import generate_judge_reports


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def load_judge_input(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "text", "output", "rewrite_type"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Judge input is missing columns: {sorted(missing)}")
    seen: set[str] = set()
    for row in rows:
        item_id = str(row["id"]).strip()
        if not item_id:
            raise ValueError("Judge input contains an empty id")
        if item_id in seen:
            raise ValueError(f"Duplicate judge input id: {item_id}")
        seen.add(item_id)
        row["id"] = item_id
        row["rewrite_type"] = normalize_rewrite_type(str(row["rewrite_type"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the independent second-layer LLM rewrite quality judge."
    )
    parser.add_argument("--config", default="configs/judge/rewrite_judge_v02_gpt4o.yaml")
    parser.add_argument("--input", required=True, help="Annotated CSV from prepare script")
    parser.add_argument("--model-key")
    parser.add_argument("--name")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--mock-perfect",
        action="store_true",
        help="Offline plumbing smoke test only; do not report as an experiment.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    config_path = resolve_path(root, args.config)
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.model_key:
        config["model_key"] = args.model_key
    if args.name:
        config["name"] = args.name
    config["input"] = args.input
    config["mock_perfect"] = bool(args.mock_perfect)

    input_path = resolve_path(root, args.input)
    rows = load_judge_input(input_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]

    if args.mock_perfect:
        judge = MockPerfectRewriteJudge()
        model_key = "mock_perfect"
    else:
        models_path = resolve_path(root, str(config["models_config"]))
        models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]
        model_key = str(config["model_key"])
        if model_key not in models:
            raise KeyError(f"Unknown model_key: {model_key}")
        if config.get("judge_mode") == "split_dimensions":
            judge = SplitRewriteQualityJudge(models[model_key], config, root)
        else:
            judge = RewriteQualityJudge(models[model_key], config, root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = args.name or config.get("name") or "rewrite_judge"
    run_dir = root / "runs" / f"{timestamp}_judge_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    prompt_hashes: dict[str, str] = {}
    if isinstance(judge, SplitRewriteQualityJudge):
        prompts_dir = run_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        for dimension, prompt_path in judge.prompt_paths.items():
            destination = prompts_dir / f"{dimension}.txt"
            shutil.copy2(prompt_path, destination)
            prompt_hashes[dimension] = sha256_file(prompt_path)
    elif not args.mock_perfect:
        prompt_path = resolve_path(root, str(config["prompt"]))
        shutil.copy2(prompt_path, run_dir / "prompt.txt")
        prompt_hashes["combined"] = sha256_file(prompt_path)
    shutil.copy2(input_path, run_dir / "judge_input.csv")

    judgments: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(config.get("concurrency", 5))) as pool:
        futures = {
            pool.submit(build_judge_prediction, row, judge): row["id"] for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            judgments.append(future.result().to_dict())
            if completed % 10 == 0 or completed == len(rows):
                print(f"Completed {completed}/{len(rows)}")

    order = {row["id"]: index for index, row in enumerate(rows)}
    judgments.sort(key=lambda row: order[row["id"]])
    metrics = generate_judge_reports(run_dir, judgments)
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "task": "rewrite_quality_judge_layer2",
        "judge_assigns_raw_scores_only": True,
        "program_computes_percentages": True,
        "uses_reference_output": False,
        "model_key": model_key,
        "model": judge.model,
        "count": len(rows),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "prompt_sha256": prompt_hashes,
        "judge_mode": config.get("judge_mode", "combined"),
        "llm_calls_per_item": 3 if config.get("judge_mode") == "split_dimensions" else 1,
        "metrics": metrics,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Run directory: {run_dir}")
    print(f"Overall quality: {metrics['overall']['quality_score']}")
    print(f"Macro quality: {metrics['macro_quality_score']}")
    print(
        f"Judge errors: {metrics['overall']['error_count']}/"
        f"{metrics['overall']['count']}"
    )
    if args.mock_perfect:
        print("WARNING: mock-perfect smoke test; these are not experiment results.")


if __name__ == "__main__":
    main()
