from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
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


def route_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts = Counter(str(p.get("route") or "LLM") for p in predictions)
    rule_counts = Counter(
        str(p["rule"])
        for p in predictions
        if p.get("route") == "RULE" and p.get("rule")
    )
    total = len(predictions)
    rule_count = route_counts.get("RULE", 0)
    llm_count = route_counts.get("LLM", 0)
    rule_correct = sum(
        p.get("route") == "RULE" and p.get("predicted") == p.get("gold")
        for p in predictions
    )
    return {
        "route_counts": dict(route_counts),
        "rule_counts": dict(rule_counts),
        "rule_routed": rule_count,
        "llm_routed": llm_count,
        "rule_coverage": 0.0 if total == 0 else rule_count / total,
        "llm_call_rate": 0.0 if total == 0 else llm_count / total,
        "rule_observed_accuracy": (
            None if rule_count == 0 else rule_correct / rule_count
        ),
    }


def generate_reports(run_dir: Path, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = calculate_metrics(predictions)
    routing = route_metrics(predictions)
    metrics["routing"] = routing
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
    rule_routed = [p for p in predictions if p.get("route") == "RULE"]

    write_csv(run_dir / "positive_misses.csv", positive_misses)
    write_csv(run_dir / "negative_false_alarms.csv", negative_false_alarms)
    write_csv(run_dir / "format_errors.csv", format_errors)
    write_csv(run_dir / "rule_routed.csv", rule_routed)

    confusion = metrics["confusion"]
    observed_rule_accuracy = routing["rule_observed_accuracy"]
    observed_rule_accuracy_text = (
        "n/a" if observed_rule_accuracy is None else _pct(observed_rule_accuracy)
    )
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

## Routing

| Metric | Value |
|---|---:|
| Rule-routed | {routing['rule_routed']} |
| LLM-routed | {routing['llm_routed']} |
| Rule coverage | {_pct(routing['rule_coverage'])} |
| LLM call rate | {_pct(routing['llm_call_rate'])} |
| Observed rule accuracy | {observed_rule_accuracy_text} |

## Rule counts

| Rule | Count |
|---|---:|
"""
    if routing["rule_counts"]:
        for rule, count in sorted(routing["rule_counts"].items()):
            summary += f"| {rule} | {count} |\n"
    else:
        summary += "| — | 0 |\n"

    summary += f"""
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
