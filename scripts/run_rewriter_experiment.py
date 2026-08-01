#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from gender_gate.data import load_items
from gender_gate.rewriter import (
    MockReferenceRewriter,
    PositiveTextRewriter,
    build_rewriter_prediction,
)
from gender_gate.rewriter_reports import generate_rewriter_reports


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an independent POSITIVE-only gender-inclusive rewriter."
    )
    parser.add_argument(
        "--config", default="configs/rewriter/rewriter_v02_gpt4o.yaml"
    )
    parser.add_argument("--split")
    parser.add_argument("--model-key")
    parser.add_argument("--model", help="Exact model id; overrides the provider model environment variable.")
    parser.add_argument("--name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Override the Rewriter output-token budget for this run.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        help="Override provider thinking_budget via extra_body (useful for Qwen).",
    )
    parser.add_argument(
        "--mock-reference",
        action="store_true",
        help="Offline plumbing smoke test only; do not report as an experiment.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    config_path = resolve_path(root, args.config)
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.split:
        config["split"] = args.split
    if args.model_key:
        config["model_key"] = args.model_key
    if args.model:
        config["model"] = args.model
    if args.name:
        config["name"] = args.name
    if args.concurrency is not None:
        if args.concurrency <= 0:
            raise ValueError("--concurrency must be positive")
        config["concurrency"] = args.concurrency
    if args.max_output_tokens is not None:
        if args.max_output_tokens <= 0:
            raise ValueError("--max-output-tokens must be positive")
        config["max_output_tokens"] = args.max_output_tokens
    if args.thinking_budget is not None:
        if args.thinking_budget <= 0:
            raise ValueError("--thinking-budget must be positive")
        extra_body = dict(config.get("extra_body") or {})
        extra_body["thinking_budget"] = args.thinking_budget
        config["extra_body"] = extra_body
    config["mock_reference"] = bool(args.mock_reference)

    split_path = resolve_path(root, str(config["split"]))
    items = load_items(split_path)
    non_positive = [item.id for item in items if item.label != "POSITIVE"]
    if non_positive:
        raise ValueError(
            "Independent rewriter requires a POSITIVE-only split; "
            f"found {len(non_positive)} non-positive items, first: {non_positive[:5]}"
        )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        items = items[: args.limit]

    if args.mock_reference:
        rewriter = MockReferenceRewriter()
        model_key = "mock_reference"
    else:
        models_path = resolve_path(root, str(config["models_config"]))
        models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]
        model_key = str(config["model_key"])
        if model_key not in models:
            raise KeyError(f"Unknown model_key: {model_key}")
        rewriter = PositiveTextRewriter(models[model_key], config, root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = args.name or config.get("name") or "rewriter"
    run_dir = root / "runs" / f"{timestamp}_rewriter_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    prompt_path = resolve_path(root, str(config["prompt"]))
    shutil.copy2(prompt_path, run_dir / "prompt.txt")

    predictions: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(config.get("concurrency", 5))) as pool:
        futures = {
            pool.submit(build_rewriter_prediction, item, rewriter): item.id
            for item in items
        }
        # for completed, future in enumerate(as_completed(futures), start=1):
        #     predictions.append(future.result().to_dict())
        #     if completed % 10 == 0 or completed == len(items):
        #         print(f"Completed {completed}/{len(items)}")
        for completed, future in enumerate(as_completed(futures), start=1):
            prediction = future.result().to_dict()
            predictions.append(prediction)

            status = "FAILED" if prediction["error"] else (
                "CHANGED" if prediction["changed"] else "UNCHANGED"
            )
            latency = prediction["latency_ms"] / 1000

            print(
                f"[{completed:4d}/{len(items)}] "
                f"{prediction['id']:<12} "
                f"{status:<9} "
                f"{latency:7.2f}s  "
                f"{prediction['final_output']}",
                flush=True,
            )

    order = {item.id: index for index, item in enumerate(items)}
    predictions.sort(key=lambda p: order[p["id"]])
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    metrics = generate_rewriter_reports(run_dir, predictions)
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "task": "independent_positive_rewriter",
        "uses_gate_predictions": False,
        "mock_reference": bool(args.mock_reference),
        "model_key": model_key,
        "model": rewriter.model,
        "count": len(items),
        "split": str(split_path),
        "split_sha256": sha256_file(split_path),
        "prompt_sha256": sha256_file(prompt_path),
        "metrics": metrics,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Run directory: {run_dir}")
    print(f"Intervention rate: {metrics['intervention_rate']:.4f}")
    print(f"Under-edit rate: {metrics['under_edit_rate']:.4f}")
    print(f"Exact reference match: {metrics['exact_reference_match_rate']:.4f}")
    print(f"Errors: {metrics['error_count']}/{metrics['count']}")
    if args.mock_reference:
        print("WARNING: mock-reference smoke test; these are not experiment results.")


if __name__ == "__main__":
    main()
