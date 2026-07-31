from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .rewrite_metrics import calculate_rewrite_metrics, semantic_review_reasons


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


def generate_rewrite_reports(
    run_dir: Path,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = calculate_rewrite_metrics(predictions)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(run_dir / "predictions.csv", predictions)

    positive_failures = [
        p for p in predictions if p["gold"] == "POSITIVE" and not p.get("changed")
    ]
    negative_over_edits = [
        p for p in predictions if p["gold"] == "NEGATIVE" and p.get("changed")
    ]
    errors = [p for p in predictions if p.get("error")]

    review_queue: list[dict[str, Any]] = []
    for prediction in predictions:
        reasons = semantic_review_reasons(prediction)
        if not reasons:
            continue
        review_queue.append(
            {
                **prediction,
                "review_reasons": ";".join(reasons),
                "review_status": "UNREVIEWED",
                "semantic_preservation": "",
                "bias_removed": "",
                "review_note": "",
            }
        )

    write_csv(run_dir / "positive_failures.csv", positive_failures)
    write_csv(run_dir / "negative_over_edits.csv", negative_over_edits)
    write_csv(run_dir / "errors.csv", errors)
    write_csv(run_dir / "semantic_review_queue.csv", review_queue)

    counts = metrics["counts"]
    summary = f"""# Rewrite Experiment Summary

| Metric | Value |
|---|---:|
| Count | {metrics['count']} |
| Positive count | {metrics['positive_count']} |
| Negative count | {metrics['negative_count']} |
| Positive intervention rate | {_pct(metrics['positive_intervention_rate'])} |
| Under-edit rate | {_pct(metrics['under_edit_rate'])} |
| Negative preservation | {_pct(metrics['negative_preservation'])} |
| Over-edit rate | {_pct(metrics['over_edit_rate'])} |
| Rewrite calls | {metrics['rewrite_calls']} |
| Rewrite calls saved | {metrics['rewrite_calls_saved']} |
| Error rate | {_pct(metrics['error_rate'])} |
| Exact reference match rate | {_pct(metrics['exact_reference_match_rate'])} |

## Counts

| Item | Count |
|---|---:|
| Positive changed | {counts['positive_changed']} |
| Positive unchanged | {counts['positive_unchanged']} |
| Negative preserved | {counts['negative_preserved']} |
| Negative changed | {counts['negative_changed']} |
| Semantic review candidates | {len(review_queue)} |

`semantic_review_queue.csv` contains conservative heuristic candidates only. It is not an automatic semantic-violation verdict.
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return metrics
