#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from gender_gate.rewrite_judge import (
    AutoTypeRewriteQualityJudge,
    build_auto_type_judge_prediction,
    normalize_rewrite_type,
)
from gender_gate.rewrite_judge_metrics import generate_judge_reports


def resolve_path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "text", "output", "rewrite_type"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    for row in rows:
        row["id"] = str(row["id"]).strip()
        row["gold_rewrite_type"] = normalize_rewrite_type(str(row["rewrite_type"]))
    return rows


def balanced_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0:
        raise ValueError("--limit must be positive")
    rng = random.Random(seed)
    local = [r for r in rows if r["gold_rewrite_type"] == "LOCAL_REPAIR"]
    recon = [r for r in rows if r["gold_rewrite_type"] == "PROPOSITION_RECONSTRUCTION"]
    half = n // 2
    local_n = min(half, len(local))
    recon_n = min(half, len(recon))
    chosen = rng.sample(local, local_n) + rng.sample(recon, recon_n)
    remaining = n - len(chosen)
    if remaining:
        chosen_ids = {id(r) for r in chosen}
        pool = [r for r in rows if id(r) not in chosen_ids]
        chosen.extend(rng.sample(pool, min(remaining, len(pool))))
    rng.shuffle(chosen)
    return chosen


def f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def type_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if not r.get("error")]
    labels = ["LOCAL_REPAIR", "PROPOSITION_RECONSTRUCTION"]
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for row in valid:
        matrix[row["gold_rewrite_type"]][row["rewrite_type"]] += 1

    per_type: dict[str, Any] = {}
    f1s: list[float] = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in labels if g != label)
        fn = sum(matrix[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = f1(tp, fp, fn)
        f1s.append(score)
        per_type[label] = {
            "precision": precision,
            "recall": recall,
            "f1": score,
            "support": sum(matrix[label].values()),
        }

    correct = sum(matrix[label][label] for label in labels)
    accuracy = correct / len(valid) if valid else 0.0
    return {
        "count": len(rows),
        "valid_count": len(valid),
        "error_count": len(rows) - len(valid),
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s),
        "per_type": per_type,
        "confusion_matrix": matrix,
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick-test Judge V05: infer rewrite type and score in one LLM call."
    )
    parser.add_argument("--input", required=True, help="CSV with id,text,output,rewrite_type (gold type for evaluation only)")
    parser.add_argument("--config", default="configs/judge/rewrite_judge_v05_auto_type_gpt4o.yaml")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--name", default="quick20")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    config_path = resolve_path(root, args.config)
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency

    rows = load_rows(resolve_path(root, args.input))
    if args.limit:
        rows = (
            balanced_sample(rows, args.limit, args.seed)
            if args.balanced
            else rows[: args.limit]
        )

    models_path = resolve_path(root, str(config["models_config"]))
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]
    model_key = str(config["model_key"])
    judge = AutoTypeRewriteQualityJudge(models[model_key], config, root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "runs" / f"{timestamp}_judge_v05_auto_type_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    predictions: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(config.get("concurrency", 5))) as pool:
        futures = {
            pool.submit(build_auto_type_judge_prediction, row, judge): row
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            pred = future.result().to_dict()
            pred["gold_rewrite_type"] = source["gold_rewrite_type"]
            pred["type_correct"] = (
                None if pred.get("error") else pred["rewrite_type"] == source["gold_rewrite_type"]
            )
            predictions.append(pred)
            if completed % 10 == 0 or completed == len(rows):
                print(f"Completed {completed}/{len(rows)}")

    order = {row["id"]: i for i, row in enumerate(rows)}
    predictions.sort(key=lambda r: order[r["id"]])

    tm = type_metrics(predictions)
    (run_dir / "rewrite_type_metrics.json").write_text(
        json.dumps(tm, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(run_dir / "rewrite_type_predictions.csv", predictions)
    write_csv(
        run_dir / "rewrite_type_errors.csv",
        [r for r in predictions if not r.get("error") and not r.get("type_correct")],
    )
    # Same three-column shape as the existing frozen type map, so the current
    # Adaptive Rewriter can consume V05 predictions without any code changes.
    auto_map = [
        {
            "id": r["id"],
            "rewrite_type": r["rewrite_type"],
            "type_note": "auto:rewrite_judge_v05",
        }
        for r in predictions
        if not r.get("error")
    ]
    write_csv(run_dir / "rewrite_type_map_auto.csv", auto_map)

    # Existing scoring/reporting is reused unchanged; it uses the judge-predicted
    # type and the exact v04 normalization/weights/verdict logic.
    quality_metrics = generate_judge_reports(run_dir, predictions)

    print(f"Run directory: {run_dir}")
    print(f"Rewrite-type accuracy: {tm['accuracy'] * 100:.2f}%")
    print(f"Rewrite-type macro-F1: {tm['macro_f1'] * 100:.2f}%")
    print(
        "LOCAL recall: "
        f"{tm['per_type']['LOCAL_REPAIR']['recall'] * 100:.2f}%"
    )
    print(
        "RECONSTRUCTION recall: "
        f"{tm['per_type']['PROPOSITION_RECONSTRUCTION']['recall'] * 100:.2f}%"
    )
    print(f"Judge quality (predicted type): {quality_metrics['overall']['quality_score']}")


if __name__ == "__main__":
    main()
