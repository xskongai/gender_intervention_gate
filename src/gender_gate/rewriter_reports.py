from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .metrics import safe_div
from .rewriter import normalize_rewrite_text


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def calculate_rewriter_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    if any(p.get("gold") != "POSITIVE" for p in predictions):
        raise ValueError("Rewriter metrics require POSITIVE-only predictions.")

    count = len(predictions)
    changed = sum(bool(p.get("changed")) for p in predictions)
    errors = sum(bool(p.get("error")) for p in predictions)
    cache_hits = sum(bool(p.get("cache_hit")) for p in predictions)
    exact_matches = 0
    reference_count = 0
    latencies: list[int] = []

    for prediction in predictions:
        if not prediction.get("cache_hit"):
            latencies.append(int(prediction.get("latency_ms") or 0))
        reference = prediction.get("reference_output")
        if reference is None:
            continue
        reference_count += 1
        exact_matches += (
            normalize_rewrite_text(str(prediction.get("final_output") or ""))
            == normalize_rewrite_text(str(reference))
        )

    return {
        "count": count,
        "changed_count": changed,
        "unchanged_count": count - changed,
        "intervention_rate": safe_div(changed, count),
        "under_edit_rate": 1.0 - safe_div(changed, count),
        "error_count": errors,
        "error_rate": safe_div(errors, count),
        "cache_hits": cache_hits,
        "reference_count": reference_count,
        "exact_reference_match_count": exact_matches,
        "exact_reference_match_rate": safe_div(exact_matches, reference_count),
        "mean_latency_ms_non_cached": mean(latencies) if latencies else 0.0,
    }


def generate_rewriter_reports(
    run_dir: Path,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = calculate_rewriter_metrics(predictions)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(run_dir / "predictions.csv", predictions)
    _write_csv(
        run_dir / "unchanged_outputs.csv",
        [p for p in predictions if not p.get("changed")],
    )
    _write_csv(run_dir / "errors.csv", [p for p in predictions if p.get("error")])

    review_rows: list[dict[str, Any]] = []
    for p in predictions:
        review_rows.append(
            {
                "id": p["id"],
                "text": p["text"],
                "final_output": p["final_output"],
                "changed": p["changed"],
                "reference_output": p.get("reference_output"),
                "bias_removed": "",
                "semantic_preserved": "",
                "unsupported_insertion": "",
                "meaning_distortion": "",
                "fluency": "",
                "verdict": "",
                "review_note": "",
            }
        )
    _write_csv(run_dir / "manual_review.csv", review_rows)

    summary = f"""# Independent Rewriter Experiment

This run contains POSITIVE instances only. It does not read or use Gate predictions.

| Automatic metric | Value |
|---|---:|
| Count | {metrics['count']} |
| Changed | {metrics['changed_count']} |
| Unchanged | {metrics['unchanged_count']} |
| Intervention rate | {metrics['intervention_rate'] * 100:.2f}% |
| Under-edit rate | {metrics['under_edit_rate'] * 100:.2f}% |
| Exact reference match | {metrics['exact_reference_match_rate'] * 100:.2f}% |
| Error rate | {metrics['error_rate'] * 100:.2f}% |
| Mean non-cached latency | {metrics['mean_latency_ms_non_cached']:.1f} ms |

`manual_review.csv` is the authoritative template for Bias Removal and Semantic Preservation. Exact reference match is diagnostic only and is not treated as rewrite quality.
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return metrics
