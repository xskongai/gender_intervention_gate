#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from gender_gate.rewrite_judge import normalize_rewrite_type
from gender_gate.rewrite_type_judge import RewriteTypeJudge


def resolve_path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Input CSV is empty")
    required = {"id", "text"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    seen: set[str] = set()
    for row in rows:
        item_id = str(row.get("id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not item_id or not text:
            raise ValueError("Input contains an empty id or text")
        if item_id in seen:
            raise ValueError(f"Duplicate id: {item_id}")
        seen.add(item_id)
        row["id"] = item_id
        row["text"] = text
        if str(row.get("rewrite_type", "")).strip():
            row["gold_rewrite_type"] = normalize_rewrite_type(
                str(row["rewrite_type"])
            )
    return rows


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    sample_size: int | None,
    balanced: bool,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size is None or sample_size >= len(rows):
        return list(rows)
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    rng = random.Random(seed)
    if not balanced:
        selected = rng.sample(rows, sample_size)
        order = {row["id"]: i for i, row in enumerate(rows)}
        return sorted(selected, key=lambda row: order[row["id"]])

    if sample_size % 2 != 0:
        raise ValueError("--balanced requires an even --sample-size")
    if not all("gold_rewrite_type" in row for row in rows):
        raise ValueError("--balanced requires a rewrite_type gold column")

    local = [r for r in rows if r["gold_rewrite_type"] == "LOCAL_REPAIR"]
    recon = [
        r
        for r in rows
        if r["gold_rewrite_type"] == "PROPOSITION_RECONSTRUCTION"
    ]
    each = sample_size // 2
    if len(local) < each or len(recon) < each:
        raise ValueError(
            f"Not enough rows for balanced sample: local={len(local)}, "
            f"reconstruction={len(recon)}, requested_each={each}"
        )
    selected_ids = {
        r["id"] for r in rng.sample(local, each) + rng.sample(recon, each)
    }
    return [r for r in rows if r["id"] in selected_ids]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [
        r
        for r in rows
        if r.get("gold_rewrite_type")
        and r.get("rewrite_type")
        and not r.get("error")
    ]
    labels = ["LOCAL_REPAIR", "PROPOSITION_RECONSTRUCTION"]
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for row in evaluable:
        matrix[str(row["gold_rewrite_type"])][str(row["rewrite_type"])] += 1

    per_class: dict[str, Any] = {}
    f1s: list[float] = []
    recalls: list[float] = []
    for label in labels:
        tp = matrix[label][label]
        fn = sum(matrix[label].values()) - tp
        fp = sum(matrix[g][label] for g in labels if g != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
        f1s.append(f1)
        recalls.append(recall)

    correct = sum(matrix[label][label] for label in labels)
    total = len(evaluable)
    return {
        "count": len(rows),
        "evaluable_count": total,
        "error_count": sum(1 for r in rows if r.get("error")),
        "accuracy": correct / total if total else None,
        "macro_f1": sum(f1s) / len(f1s) if total else None,
        "balanced_accuracy": sum(recalls) / len(recalls) if total else None,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an independent rewrite-type judgment using ORIGINAL text only. "
            "Gold rewrite_type, when present, is used only for evaluation."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--config", default="configs/judge/rewrite_type_judge_v01_gpt4o.yaml"
    )
    parser.add_argument("--model-key")
    parser.add_argument("--model")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--name", default="quick")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    config_path = resolve_path(root, args.config)
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.model_key:
        config["model_key"] = args.model_key
    if args.model:
        config["model"] = args.model
    if args.concurrency is not None:
        if args.concurrency <= 0:
            raise ValueError("--concurrency must be positive")
        config["concurrency"] = args.concurrency

    input_path = resolve_path(root, args.input)
    all_rows = read_rows(input_path)
    rows = sample_rows(
        all_rows,
        sample_size=args.sample_size,
        balanced=args.balanced,
        seed=args.seed,
    )

    models_path = resolve_path(root, str(config["models_config"]))
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]
    model_key = str(config["model_key"])
    if model_key not in models:
        raise KeyError(f"Unknown model_key: {model_key}")
    judge = RewriteTypeJudge(models[model_key], config, root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "runs" / f"{timestamp}_rewrite_type_judge_v01_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(judge.prompt_path, run_dir / "prompt.txt")
    shutil.copy2(input_path, run_dir / "source_input.csv")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    predictions: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(config.get("concurrency", 5))) as pool:
        futures = {
            pool.submit(judge.judge, item_id=row["id"], text=row["text"]): row
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            pred = future.result().to_dict()
            if source.get("gold_rewrite_type"):
                pred["gold_rewrite_type"] = source["gold_rewrite_type"]
                pred["type_correct"] = (
                    not pred.get("error")
                    and pred.get("rewrite_type") == source["gold_rewrite_type"]
                )
            predictions.append(pred)
            if completed % 10 == 0 or completed == len(rows):
                print(f"Completed {completed}/{len(rows)}")

    order = {row["id"]: i for i, row in enumerate(rows)}
    predictions.sort(key=lambda row: order[row["id"]])
    metrics = compute_metrics(predictions)

    write_csv(run_dir / "rewrite_type_predictions.csv", predictions)
    write_csv(
        run_dir / "rewrite_type_errors.csv",
        [r for r in predictions if r.get("type_correct") is False or r.get("error")],
    )

    auto_map = [
        {
            "id": row["id"],
            "rewrite_type": row["rewrite_type"],
            "type_note": "auto:rewrite_type_judge_v01",
        }
        for row in predictions
        if row.get("rewrite_type") and not row.get("error")
    ]
    write_csv(run_dir / "rewrite_type_map_auto.csv", auto_map)

    pred_by_id = {row["id"]: row for row in predictions}
    v04_input: list[dict[str, Any]] = []
    for source in rows:
        pred = pred_by_id[source["id"]]
        if pred.get("error") or not pred.get("rewrite_type"):
            continue
        out = dict(source)
        out["rewrite_type"] = pred["rewrite_type"]
        out.pop("gold_rewrite_type", None)
        v04_input.append(out)
    write_csv(run_dir / "judge_input_v04_auto_type.csv", v04_input)

    (run_dir / "rewrite_type_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "task": "rewrite_type_judge_v01",
        "uses_original_only": True,
        "uses_candidate": False,
        "gold_type_used_for_inference": False,
        "model_key": model_key,
        "model": judge.model,
        "count": len(rows),
        "source_input": str(input_path),
        "source_input_sha256": sha256_file(input_path),
        "prompt_sha256": sha256_file(judge.prompt_path),
        "metrics": metrics,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Run directory: {run_dir}")
    if metrics["accuracy"] is not None:
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Macro-F1: {metrics['macro_f1'] * 100:.2f}%")
        print(f"Balanced accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
        for label, values in metrics["per_class"].items():
            print(
                f"{label}: precision={values['precision'] * 100:.2f}% "
                f"recall={values['recall'] * 100:.2f}% "
                f"f1={values['f1'] * 100:.2f}%"
            )
    else:
        print("No gold rewrite_type column: generated predictions without evaluation.")
    print(f"V04-ready input: {run_dir / 'judge_input_v04_auto_type.csv'}")


if __name__ == "__main__":
    main()
