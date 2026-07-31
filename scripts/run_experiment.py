#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from gender_gate.classifier import BinaryClassifier
from gender_gate.data import load_items
from gender_gate.reports import generate_reports


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_rule_first_override(config: dict, enabled: bool | None) -> None:
    if enabled is None:
        return
    rule_first = dict(config.get("rule_first") or {})
    rule_first.setdefault("ruleset", "deterministic_v01")
    rule_first["enabled"] = enabled
    config["rule_first"] = rule_first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split")
    parser.add_argument("--model-key")
    parser.add_argument("--model", help="Exact model id; overrides the provider model environment variable.")
    parser.add_argument("--name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument(
        "--progress-every",
        type=int,
        help="Print detailed progress every N completed items (default: config value or 10).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Override the Gate output-token budget for this run.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        help="Override provider thinking_budget via extra_body (useful for Qwen).",
    )
    rule_group = parser.add_mutually_exclusive_group()
    rule_group.add_argument(
        "--rule-first",
        action="store_true",
        help="Enable deterministic rules before the frozen LLM Gate.",
    )
    rule_group.add_argument(
        "--no-rule-first",
        action="store_true",
        help="Disable deterministic rules and use the frozen LLM Gate only.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    config_path = root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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
    if args.progress_every is not None:
        if args.progress_every <= 0:
            raise ValueError("--progress-every must be positive")
        config["progress_every"] = args.progress_every
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

    override: bool | None = None
    if args.rule_first:
        override = True
    elif args.no_rule_first:
        override = False
    apply_rule_first_override(config, override)

    models_path = root / config["models_config"]
    models = yaml.safe_load(
        models_path.read_text(encoding="utf-8")
    )["models"]
    model_key = config["model_key"]
    if model_key not in models:
        raise KeyError(f"Unknown model_key: {model_key}")

    # Named model profiles may define fast local defaults. A CLI override remains
    # authoritative, while profile defaults replace the generic experiment budget.
    model_profile = models[model_key]
    if args.max_output_tokens is None and model_profile.get("max_tokens") is not None:
        config["max_output_tokens"] = int(model_profile["max_tokens"])
    if model_profile.get("temperature") is not None:
        config["temperature"] = float(model_profile["temperature"])
    if model_profile.get("retries") is not None:
        config["retries"] = int(model_profile["retries"])

    split_path = root / config["split"]
    items = load_items(split_path)
    if args.limit:
        items = items[: args.limit]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "runs" / f"{timestamp}_{config['name']}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    prompt_path = root / config["prompt"]
    shutil.copy2(prompt_path, run_dir / "prompt.txt")
    if config.get("examples"):
        shutil.copy2(
            root / config["examples"],
            run_dir / "examples.jsonl",
        )

    classifier = BinaryClassifier(models[model_key], config, root)
    predictions: list[dict] = []
    output_path = run_dir / "predictions.jsonl"

    with output_path.open("w", encoding="utf-8") as output:
        with ThreadPoolExecutor(
            max_workers=int(config.get("concurrency", 5))
        ) as pool:
            futures = {
                pool.submit(classifier.predict, item): item
                for item in items
            }
            completed = 0
            for future in as_completed(futures):
                prediction = future.result().to_dict()
                predictions.append(prediction)
                output.write(
                    json.dumps(prediction, ensure_ascii=False) + "\n"
                )
                output.flush()
                completed += 1
                progress_every = int(config.get("progress_every", 10))
                if completed % progress_every == 0 or completed == len(items):
                    pred_text = prediction.get("predicted") or "FAILED"
                    route = prediction.get("route") or "LLM"
                    latency_s = float(prediction.get("latency_ms") or 0) / 1000.0
                    print(
                        f"[{completed:>4}/{len(items)}] "
                        f"{prediction.get('id', ''):<12} "
                        f"gold={prediction.get('gold', ''):<8} "
                        f"pred={pred_text:<8} "
                        f"route={route:<4} "
                        f"{latency_s:>7.2f}s",
                        flush=True,
                    )

    metrics = generate_reports(run_dir, predictions)
    rule_config = config.get("rule_first") or {}
    rules_path = root / "src/gender_gate/deterministic_rules.py"
    manifest = {
        "started_from_config": str(config_path.relative_to(root)),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "model": classifier.client.model,
        "count": len(items),
        "dataset_sha256": sha256_file(
            root / "data/processed/main.jsonl"
        ),
        "split_sha256": sha256_file(split_path),
        "prompt_sha256": sha256_file(prompt_path),
        "examples_sha256": (
            sha256_file(root / config["examples"])
            if config.get("examples")
            else None
        ),
        "rule_first": {
            "enabled": bool(rule_config.get("enabled", False)),
            "ruleset": rule_config.get("ruleset"),
            "rules_sha256": (
                sha256_file(rules_path)
                if bool(rule_config.get("enabled", False))
                else None
            ),
        },
        "metrics": metrics,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Run directory: {run_dir}")
    print(f"Positive Recall: {metrics['positive_recall']:.4f}")
    print(f"Negative Recall: {metrics['negative_recall']:.4f}")
    print(
        f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}"
    )
    print(f"Rule coverage: {metrics['routing']['rule_coverage']:.4f}")
    print(f"LLM call rate: {metrics['routing']['llm_call_rate']:.4f}")
    print(f"Format error rate: {metrics['format_error_rate']:.4f}")
    print(f"Passes 90% target: {metrics['passes_90_target']}")


if __name__ == "__main__":
    main()
