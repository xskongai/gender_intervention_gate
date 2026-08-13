#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gender_gate.data import load_items
from gender_gate.rewrite_type_router import RewriteTypeRouter


LABELS = ("LOCAL_REPAIR", "PROPOSITION_RECONSTRUCTION")


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_gold(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["id"]).strip(): str(row["rewrite_type"]).strip()
            for row in csv.DictReader(handle)
            if str(row.get("id", "")).strip()
        }


def choose_items(items, gold, sample_size: int | None, balanced: bool, seed: int):
    eligible = [item for item in items if item.id in gold]
    if sample_size is None or sample_size >= len(eligible):
        return eligible

    rng = random.Random(seed)
    if not balanced:
        rng.shuffle(eligible)
        return eligible[:sample_size]

    buckets = {label: [] for label in LABELS}
    for item in eligible:
        label = gold[item.id]
        if label in buckets:
            buckets[label].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    first = sample_size // 2
    wanted = {
        LABELS[0]: first,
        LABELS[1]: sample_size - first,
    }
    selected = []
    for label in LABELS:
        selected.extend(buckets[label][: wanted[label]])

    # If one class is too small, fill from remaining items.
    if len(selected) < sample_size:
        used = {item.id for item in selected}
        remaining = [item for item in eligible if item.id not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])

    rng.shuffle(selected)
    return selected


def class_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    requested = len(rows)
    valid = [row for row in rows if row["predicted_type"] in LABELS]
    correct = sum(row["predicted_type"] == row["gold_type"] for row in valid)

    per_class = {}
    f1s = []
    for label in LABELS:
        tp = sum(r["gold_type"] == label and r["predicted_type"] == label for r in valid)
        fp = sum(r["gold_type"] != label and r["predicted_type"] == label for r in valid)
        fn = sum(r["gold_type"] == label and r["predicted_type"] != label for r in valid)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(r["gold_type"] == label for r in rows),
        }

    return {
        "requested": requested,
        "valid_predictions": len(valid),
        "errors": requested - len(valid),
        "accuracy": correct / requested if requested else 0.0,
        "accuracy_on_valid": correct / len(valid) if valid else 0.0,
        "macro_f1": sum(f1s) / len(f1s),
        "gold_counts": dict(Counter(row["gold_type"] for row in rows)),
        "predicted_counts": dict(Counter(row["predicted_type"] for row in valid)),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick automatic rewrite-type router test")
    parser.add_argument("--data", default="data/processed/positive_full_871.jsonl")
    parser.add_argument("--gold-map", default="data/review/rewrite_type_map_positive_full871.csv")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--model-key", default="openai_judge")
    parser.add_argument("--model", default=None, help="Optional explicit model override")
    parser.add_argument("--prompt", default="prompts/rewrite_type_router_v01.txt")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--output-dir", default="runs/rewrite_type_router/quick_v01")
    args = parser.parse_args()

    root = ROOT
    load_dotenv(root / ".env")

    with resolve(root, args.models_config).open(encoding="utf-8") as handle:
        models_payload = yaml.safe_load(handle)
    model_config = dict(models_payload["models"][args.model_key])

    router_config: dict[str, Any] = {
        "prompt": args.prompt,
        "temperature": 0,
        "max_output_tokens": 32,
        "retries": 2,
        "cache_db": ".cache/llm_cache.sqlite",
    }
    if args.model:
        router_config["model"] = args.model

    gold = load_gold(resolve(root, args.gold_map))
    items = load_items(resolve(root, args.data))
    selected = choose_items(items, gold, args.sample_size, args.balanced, args.seed)

    router = RewriteTypeRouter(model_config, router_config, root)
    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(router.predict, item): item for item in selected}
        for future in as_completed(futures):
            item = futures[future]
            prediction = future.result()
            rows.append(
                {
                    "id": item.id,
                    "text": item.text,
                    "gold_type": gold[item.id],
                    "predicted_type": prediction.predicted_type,
                    "correct": prediction.predicted_type == gold[item.id],
                    "raw_output": prediction.raw_output,
                    "model": prediction.model,
                    "prompt_version": prediction.prompt_version,
                    "latency_ms": prediction.latency_ms,
                    "cache_hit": prediction.cache_hit,
                    "error": prediction.error,
                }
            )
    rows.sort(key=lambda row: row["id"])

    metrics = class_metrics(rows)
    metrics.update(
        {
            "model": router.model,
            "prompt": args.prompt,
            "sample_size": len(selected),
            "balanced_sample": args.balanced,
            "seed": args.seed,
        }
    )

    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    # Same schema as the frozen type map, so this file can directly replace type_map.
    auto_map_path = output_dir / "rewrite_type_map_auto.csv"
    valid_rows = [row for row in rows if row["predicted_type"] in LABELS]
    with auto_map_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "rewrite_type", "type_note"])
        writer.writeheader()
        for row in valid_rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "rewrite_type": row["predicted_type"],
                    "type_note": f"auto:{router.prompt_version}",
                }
            )

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Model: {router.model}")
    print(f"Sample: {metrics['requested']} (valid={metrics['valid_predictions']}, errors={metrics['errors']})")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro-F1: {metrics['macro_f1'] * 100:.2f}%")
    print(f"Predictions: {predictions_path}")
    print(f"Auto type map: {auto_map_path}")


if __name__ == "__main__":
    main()
