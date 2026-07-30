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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split")
    parser.add_argument("--model-key")
    parser.add_argument("--name")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    config_path = root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.split:
        config["split"] = args.split
    if args.model_key:
        config["model_key"] = args.model_key
    if args.name:
        config["name"] = args.name

    models_path = root / config["models_config"]
    models = yaml.safe_load(
        models_path.read_text(encoding="utf-8")
    )["models"]
    model_key = config["model_key"]
    if model_key not in models:
        raise KeyError(f"Unknown model_key: {model_key}")

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
                if completed % 10 == 0 or completed == len(items):
                    print(f"Completed {completed}/{len(items)}")

    metrics = generate_reports(run_dir, predictions)
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
    print(f"Passes 90% target: {metrics['passes_90_target']}")


if __name__ == "__main__":
    main()
