from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metrics import calculate_metrics


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def group_report(predictions: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        meta = prediction.get("meta") or {}
        groups[str(meta.get(field) or "UNKNOWN")].append(prediction)

    rows: list[dict[str, Any]] = []
    for name, group in sorted(groups.items()):
        correct = sum(p.get("predicted") == p["gold"] for p in group)
        rows.append(
            {
                field: name,
                "gold_label": group[0]["gold"],
                "count": len(group),
                "correct": correct,
                "errors": len(group) - correct,
                "accuracy": correct / len(group),
            }
        )
    return rows


def generate_reports(run_dir: Path, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = calculate_metrics(predictions)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(run_dir / "by_l1.csv", group_report(predictions, "l1"))
    write_csv(run_dir / "by_l2.csv", group_report(predictions, "l2"))

    positive_misses = [
        p
        for p in predictions
        if p["gold"] == "POSITIVE" and p.get("predicted") != "POSITIVE"
    ]
    negative_false_alarms = [
        p
        for p in predictions
        if p["gold"] == "NEGATIVE" and p.get("predicted") != "NEGATIVE"
    ]
    format_errors = [p for p in predictions if p.get("predicted") is None]

    write_csv(run_dir / "positive_misses.csv", positive_misses)
    write_csv(run_dir / "negative_false_alarms.csv", negative_false_alarms)
    write_csv(run_dir / "format_errors.csv", format_errors)

    confusion = metrics["confusion"]
    summary = f"""# Experiment Summary

| Metric | Value |
|---|---:|
| Count | {metrics['count']} |
| Positive Recall | {_pct(metrics['positive_recall'])} |
| Negative Recall | {_pct(metrics['negative_recall'])} |
| Positive Precision | {_pct(metrics['positive_precision'])} |
| Negative Precision | {_pct(metrics['negative_precision'])} |
| Balanced Accuracy | {_pct(metrics['balanced_accuracy'])} |
| Macro-F1 | {_pct(metrics['macro_f1'])} |
| Overall Accuracy | {_pct(metrics['accuracy'])} |
| Format Error Rate | {_pct(metrics['format_error_rate'])} |
| Both classes >= 90% | {metrics['passes_90_target']} |
| Both classes >= 94% | {metrics['passes_94_dev_target']} |

## Confusion counts

| Item | Count |
|---|---:|
| True Positive | {confusion['true_positive']} |
| False Negative | {confusion['false_negative']} |
| True Negative | {confusion['true_negative']} |
| False Positive | {confusion['false_positive']} |
| Format Errors | {confusion['format_errors']} |
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return metrics
