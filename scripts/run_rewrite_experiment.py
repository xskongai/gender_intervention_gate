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

from gender_gate.data import load_items, read_jsonl
from gender_gate.rewrite import (
    MockOracleRewriter,
    TextRewriter,
    build_rewrite_prediction,
    build_skipped_prediction,
)
from gender_gate.rewrite_reports import generate_rewrite_reports


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gate_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if path.is_dir():
        path = path / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Gate predictions not found: {path}")

    predictions = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        item_id = str(prediction["id"])
        if item_id in by_id:
            raise ValueError(f"Duplicate gate prediction id: {item_id}")
        by_id[item_id] = prediction
    return by_id


def resolve_path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run direct or gate-controlled gender-inclusive rewriting."
    )
    parser.add_argument("--config", default="configs/rewrite/rewrite_gpt4o.yaml")
    parser.add_argument("--mode", required=True, choices=["direct", "gated"])
    parser.add_argument("--split")
    parser.add_argument("--gate-run")
    parser.add_argument("--model-key")
    parser.add_argument("--name")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--mock-oracle",
        action="store_true",
        help="Offline plumbing smoke test only; do not report as an experiment.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    config_path = resolve_path(root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.split:
        config["split"] = args.split
    if args.model_key:
        config["model_key"] = args.model_key
    if args.name:
        config["name"] = args.name
    config["mode"] = args.mode
    config["gate_run"] = args.gate_run
    config["mock_oracle"] = bool(args.mock_oracle)

    split_path = resolve_path(root, config["split"])
    items = load_items(split_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        items = items[: args.limit]

    gate_by_id: dict[str, dict[str, Any]] = {}
    gate_path: Path | None = None
    if args.mode == "gated":
        if not args.gate_run:
            parser.error("--gate-run is required when --mode gated")
        gate_path = resolve_path(root, args.gate_run)
        gate_by_id = load_gate_predictions(gate_path)
        missing = [item.id for item in items if item.id not in gate_by_id]
        if missing:
            raise ValueError(
                f"Gate run is missing {len(missing)} split ids; first: {missing[:5]}"
            )

    if args.mock_oracle:
        rewriter = MockOracleRewriter()
        model_key = "mock_oracle"
    else:
        models_path = resolve_path(root, config["models_config"])
        models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]
        model_key = config["model_key"]
        if model_key not in models:
            raise KeyError(f"Unknown model_key: {model_key}")
        rewriter = TextRewriter(models[model_key], config, root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = args.name or config.get("name") or "rewrite"
    run_dir = root / "runs" / f"{timestamp}_{args.mode}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    prompt_path = resolve_path(root, config["prompt"])
    shutil.copy2(prompt_path, run_dir / "prompt.txt")

    predictions: list[dict[str, Any]] = []
    output_path = run_dir / "predictions.jsonl"

    # Fail-safe behavior: a missing/invalid gate decision keeps the original text.
    to_rewrite: list[tuple[Any, dict[str, Any] | None]] = []
    for item in items:
        if args.mode == "direct":
            to_rewrite.append((item, None))
            continue

        gate_prediction = gate_by_id[item.id]
        if gate_prediction.get("predicted") == "POSITIVE" and not gate_prediction.get("error"):
            to_rewrite.append((item, gate_prediction))
        else:
            skip_error = None
            if gate_prediction.get("predicted") not in {"POSITIVE", "NEGATIVE"}:
                skip_error = "GATE_INVALID_OR_MISSING_LABEL: fail-safe KEEP"
            prediction = build_skipped_prediction(
                item=item,
                mode="gated",
                model=rewriter.model,
                prompt_version=rewriter.prompt_version,
                gate_prediction=gate_prediction,
                error=skip_error,
            ).to_dict()
            predictions.append(prediction)

    with ThreadPoolExecutor(max_workers=int(config.get("concurrency", 5))) as pool:
        futures = {
            pool.submit(
                build_rewrite_prediction,
                item,
                args.mode,
                rewriter,
                gate_prediction,
            ): item.id
            for item, gate_prediction in to_rewrite
        }
        completed = len(predictions)
        for future in as_completed(futures):
            predictions.append(future.result().to_dict())
            completed += 1
            if completed % 10 == 0 or completed == len(items):
                print(f"Completed {completed}/{len(items)}")

    # Stable dataset order makes direct/gated comparison and diffs deterministic.
    order = {item.id: index for index, item in enumerate(items)}
    predictions.sort(key=lambda p: order[p["id"]])
    with output_path.open("w", encoding="utf-8") as output:
        for prediction in predictions:
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    metrics = generate_rewrite_reports(run_dir, predictions)
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "mock_oracle": bool(args.mock_oracle),
        "model_key": model_key,
        "model": rewriter.model,
        "count": len(items),
        "split": str(split_path),
        "split_sha256": sha256_file(split_path),
        "prompt_sha256": sha256_file(prompt_path),
        "gate_predictions": str(gate_path) if gate_path else None,
        "gate_predictions_sha256": (
            sha256_file(gate_path / "predictions.jsonl")
            if gate_path and gate_path.is_dir()
            else sha256_file(gate_path)
            if gate_path
            else None
        ),
        "metrics": metrics,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Run directory: {run_dir}")
    print(f"Positive intervention rate: {metrics['positive_intervention_rate']:.4f}")
    print(f"Negative preservation: {metrics['negative_preservation']:.4f}")
    print(f"Over-edit rate: {metrics['over_edit_rate']:.4f}")
    print(f"Under-edit rate: {metrics['under_edit_rate']:.4f}")
    print(f"Rewrite calls: {metrics['rewrite_calls']}/{metrics['count']}")
    if args.mock_oracle:
        print("WARNING: mock-oracle smoke test; these are not experiment results.")


if __name__ == "__main__":
    main()
