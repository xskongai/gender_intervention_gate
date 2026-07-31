#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gender_gate.rewriter_reports import calculate_rewriter_metrics


def load_predictions(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def align_positive_predictions(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Align candidate POSITIVE items against a baseline that may contain negatives."""
    baseline_by_id = {p["id"]: p for p in baseline}
    candidate_by_id = {p["id"]: p for p in candidate}
    if len(baseline_by_id) != len(baseline):
        raise ValueError("Duplicate IDs in baseline run.")
    if len(candidate_by_id) != len(candidate):
        raise ValueError("Duplicate IDs in candidate run.")

    missing = [item_id for item_id in candidate_by_id if item_id not in baseline_by_id]
    if missing:
        raise ValueError(
            f"Baseline is missing {len(missing)} candidate IDs; first: {missing[:5]}"
        )

    aligned_baseline: list[dict[str, Any]] = []
    aligned_candidate: list[dict[str, Any]] = []
    for candidate_row in candidate:
        baseline_row = baseline_by_id[candidate_row["id"]]
        item_id = candidate_row["id"]
        if candidate_row.get("gold") != "POSITIVE" or baseline_row.get("gold") != "POSITIVE":
            raise ValueError(f"Non-POSITIVE item in comparison: {item_id}")
        if baseline_row["text"] != candidate_row["text"]:
            raise ValueError(f"Input mismatch for {item_id}")
        aligned_baseline.append(baseline_row)
        aligned_candidate.append(candidate_row)
    return aligned_baseline, aligned_candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a baseline rewrite run with an independent rewriter run."
    )
    parser.add_argument("baseline_run")
    parser.add_argument("candidate_run")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_run).expanduser().resolve()
    candidate_dir = Path(args.candidate_run).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_all = load_predictions(baseline_dir)
    candidate_all = load_predictions(candidate_dir)
    baseline, candidate = align_positive_predictions(baseline_all, candidate_all)

    rows: list[dict[str, Any]] = []
    for b, c in zip(baseline, candidate, strict=True):
        rows.append(
            {
                "id": c["id"],
                "text": c["text"],
                "reference_output": c.get("reference_output")
                or b.get("reference_output"),
                "v01_output": b["final_output"],
                "v02_output": c["final_output"],
                "v01_changed": b["changed"],
                "v02_changed": c["changed"],
                "preferred": "",
                "v01_verdict": "",
                "v02_verdict": "",
                "v02_bias_removed": "",
                "v02_semantic_preserved": "",
                "v02_unsupported_insertion": "",
                "v02_meaning_distortion": "",
                "v02_fluency": "",
                "review_note": "",
            }
        )

    review_path = output_dir / "paired_manual_review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()) if rows else []
        )
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    bm = calculate_rewriter_metrics(baseline)
    cm = calculate_rewriter_metrics(candidate)
    summary = f"""# Rewriter v01 vs v02

The comparison uses the same {len(candidate)} POSITIVE inputs. The baseline run may contain additional Negative items, but they are excluded here. No Gate predictions are used by the v02 Rewriter.

| Automatic metric | v01 | v02 | v02 − v01 |
|---|---:|---:|---:|
| Intervention rate | {bm['intervention_rate'] * 100:.2f}% | {cm['intervention_rate'] * 100:.2f}% | {(cm['intervention_rate'] - bm['intervention_rate']) * 100:+.2f} pp |
| Under-edit rate | {bm['under_edit_rate'] * 100:.2f}% | {cm['under_edit_rate'] * 100:.2f}% | {(cm['under_edit_rate'] - bm['under_edit_rate']) * 100:+.2f} pp |
| Exact reference match | {bm['exact_reference_match_rate'] * 100:.2f}% | {cm['exact_reference_match_rate'] * 100:.2f}% | {(cm['exact_reference_match_rate'] - bm['exact_reference_match_rate']) * 100:+.2f} pp |
| Error rate | {bm['error_rate'] * 100:.2f}% | {cm['error_rate'] * 100:.2f}% | {(cm['error_rate'] - bm['error_rate']) * 100:+.2f} pp |

Automatic metrics are diagnostic only. Use `paired_manual_review.csv` for the authoritative comparison of rewrite quality.
"""
    (output_dir / "comparison.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Paired review: {review_path}")


if __name__ == "__main__":
    main()
