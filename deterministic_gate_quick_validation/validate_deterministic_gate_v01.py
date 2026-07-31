#!/usr/bin/env python3
"""Quick offline validation for deterministic_gate_v01.

Phase 1 needs no API: verify deterministic rules against frozen Gold labels.
Phase 2 is optional: reuse an existing LLM-only decision file to compare
LLM-only vs Rule-first + the same LLM fallback without rerunning the LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from deterministic_gate_v01 import deterministic_decision


ID_CANDIDATES = ("id", "ID", "编号", "item_id", "sample_id", "instance_id")
PRED_CANDIDATES = (
    "predicted_decision", "model_decision", "predicted", "gate_predicted", "prediction", "pred",
    "final_decision", "decision",
)


def normalize_decision(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    mapping = {
        "KEEP": "KEEP", "NEGATIVE": "KEEP", "0": "KEEP",
        "EDIT": "EDIT", "POSITIVE": "EDIT", "1": "EDIT",
    }
    if text in mapping:
        return mapping[text]
    if "KEEP" in text:
        return "KEEP"
    if "EDIT" in text:
        return "EDIT"
    return None


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    return load_csv(path)


def pick_column(row: dict[str, Any], candidates: Iterable[str], kind: str) -> str:
    for name in candidates:
        if name in row:
            return name
    raise KeyError(
        f"Cannot find {kind} column. Available columns: {sorted(row.keys())}"
    )


def metrics(rows: list[dict[str, str]], predictions: dict[str, str]) -> dict[str, Any]:
    evaluated = [r for r in rows if r["id"] in predictions]
    if not evaluated:
        raise ValueError("No matching IDs between Gold data and predictions.")

    correct = sum(predictions[r["id"]] == r["gold_decision"] for r in evaluated)
    pos = [r for r in evaluated if r["gold_decision"] == "EDIT"]
    neg = [r for r in evaluated if r["gold_decision"] == "KEEP"]

    pos_recall = (
        sum(predictions[r["id"]] == "EDIT" for r in pos) / len(pos)
        if pos else None
    )
    neg_preservation = (
        sum(predictions[r["id"]] == "KEEP" for r in neg) / len(neg)
        if neg else None
    )
    return {
        "evaluated": len(evaluated),
        "accuracy": correct / len(evaluated),
        "positive_recall": pos_recall,
        "negative_preservation": neg_preservation,
        "over_edit_rate": 1 - neg_preservation if neg_preservation is not None else None,
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="golden_v23_rule_validation.csv",
        help="Combined Gold CSV included in this bundle.",
    )
    parser.add_argument(
        "--llm-results",
        help="Optional existing LLM-only decisions in CSV or JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="validation_output",
    )
    parser.add_argument(
        "--no-current-dataset-assertions",
        action="store_true",
        help="Disable expected v2.3 size/count assertions.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(dataset)
    required = {"id", "text", "gold_decision"}
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"Dataset is missing columns: {sorted(missing)}")

    matched_rows: list[dict[str, str]] = []
    rule_predictions: dict[str, str] = {}
    rule_counts: Counter[str] = Counter()

    for row in rows:
        result = deterministic_decision(row["text"])
        if result is None:
            continue
        pred = result["decision"]
        rule = result["rule"]
        rule_predictions[row["id"]] = pred
        rule_counts[rule] += 1
        matched_rows.append({
            **row,
            "rule_decision": pred,
            "rule": rule,
            "correct": str(pred == row["gold_decision"]),
        })

    conflicts = [
        r for r in matched_rows
        if r["rule_decision"] != r["gold_decision"]
    ]
    disputed_matches = [
        r for r in matched_rows
        if str(r.get("disputed", "")).strip().lower()
        not in ("", "0", "false", "否", "无", "none", "nan")
    ]

    summary = {
        "dataset_rows": len(rows),
        "rule_routed": len(matched_rows),
        "coverage": len(matched_rows) / len(rows),
        "rule_keep": sum(r["rule_decision"] == "KEEP" for r in matched_rows),
        "rule_edit": sum(r["rule_decision"] == "EDIT" for r in matched_rows),
        "llm_fallback": len(rows) - len(matched_rows),
        "llm_call_rate": (len(rows) - len(matched_rows)) / len(rows),
        "gold_conflicts": len(conflicts),
        "matched_disputed_rows": len(disputed_matches),
        "rule_counts": dict(rule_counts),
    }

    with (output_dir / "rule_validation_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_dir / "rule_matched_rows.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(matched_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matched_rows)

    print("=" * 72)
    print("PHASE 1 — OFFLINE RULE SAFETY")
    print("=" * 72)
    print(f"Dataset rows:          {len(rows)}")
    print(f"Rule-routed:           {len(matched_rows)} ({summary['coverage']:.2%})")
    print(f"Direct KEEP / EDIT:    {summary['rule_keep']} / {summary['rule_edit']}")
    print(f"LLM fallback:          {summary['llm_fallback']} ({summary['llm_call_rate']:.2%})")
    print(f"Gold conflicts:        {len(conflicts)}")
    print(f"Disputed matches:      {len(disputed_matches)}")
    for name, count in sorted(rule_counts.items()):
        print(f"  {name:<30} {count}")

    if not args.no_current_dataset_assertions:
        assert len(rows) == 1588, f"Expected 1,588 rows, got {len(rows)}"
        assert len(matched_rows) == 202, (
            f"Expected 202 routed rows, got {len(matched_rows)}"
        )
        assert len(conflicts) == 0, f"Found {len(conflicts)} rule conflicts"
        assert len(disputed_matches) == 0, (
            f"Found {len(disputed_matches)} disputed rule matches"
        )
        print("\nPASS: current v2.3 deterministic-rule audit matches expected results.")

    if not args.llm_results:
        print("\nPhase 2 not run. To compare hybrid vs LLM-only, add:")
        print("  --llm-results /path/to/existing_gate_predictions.csv")
        return

    llm_rows = load_rows(Path(args.llm_results))
    if not llm_rows:
        raise ValueError("LLM result file is empty.")
    id_col = pick_column(llm_rows[0], ID_CANDIDATES, "ID")
    pred_col = pick_column(llm_rows[0], PRED_CANDIDATES, "prediction")

    llm_predictions: dict[str, str] = {}
    for row in llm_rows:
        pred = normalize_decision(row.get(pred_col))
        if pred is not None:
            llm_predictions[str(row[id_col])] = pred

    gold_by_id = {r["id"]: r for r in rows}
    common_ids = set(gold_by_id) & set(llm_predictions)
    if not common_ids:
        raise ValueError("No overlapping IDs between dataset and LLM results.")

    subset_rule_ids = common_ids & set(rule_predictions)

    # Same frozen LLM decisions are used for fallback; deterministic rules only
    # override rows for which they provide a sufficient-condition decision.
    hybrid_predictions = dict(llm_predictions)
    overridden = 0
    for item_id, rule_pred in rule_predictions.items():
        if item_id in hybrid_predictions:
            if hybrid_predictions[item_id] != rule_pred:
                overridden += 1
            hybrid_predictions[item_id] = rule_pred

    baseline = metrics(rows, llm_predictions)
    hybrid = metrics(rows, hybrid_predictions)

    comparison = {
        "llm_only": baseline,
        "rule_first_hybrid": hybrid,
        "delta": {
            key: (
                None if baseline[key] is None or hybrid[key] is None
                else hybrid[key] - baseline[key]
            )
            for key in (
                "accuracy", "positive_recall",
                "negative_preservation", "over_edit_rate"
            )
        },
        "rule_overrides_of_llm": overridden,
        "rule_routed_in_evaluated_subset": len(subset_rule_ids),
        "hybrid_llm_calls_in_evaluated_subset": len(common_ids) - len(subset_rule_ids),
        "hybrid_llm_call_rate_in_evaluated_subset": (len(common_ids) - len(subset_rule_ids)) / len(common_ids),
        "expected_llm_call_rate_on_full_dataset": summary["llm_call_rate"],
    }

    with (output_dir / "hybrid_comparison.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    disagreement_rows = []
    for item_id in sorted(common_ids):
        llm_pred = llm_predictions[item_id]
        hybrid_pred = hybrid_predictions[item_id]
        if llm_pred != hybrid_pred:
            gold_row = gold_by_id[item_id]
            disagreement_rows.append({
                **gold_row,
                "llm_decision": llm_pred,
                "hybrid_decision": hybrid_pred,
                "hybrid_correct": str(hybrid_pred == gold_row["gold_decision"]),
            })
    if disagreement_rows:
        with (output_dir / "hybrid_overrides.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(disagreement_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(disagreement_rows)

    print("\n" + "=" * 72)
    print("PHASE 2 — REUSE EXISTING LLM RESULTS")
    print("=" * 72)
    print(f"Matched LLM results:   {baseline['evaluated']}")
    print(f"Rule-routed subset:    {len(subset_rule_ids)}")
    print(f"Hybrid LLM calls:      {len(common_ids) - len(subset_rule_ids)} ({(len(common_ids) - len(subset_rule_ids)) / len(common_ids):.2%})")
    print(f"Rule disagreements:    {overridden}")
    print("")
    print(f"{'Metric':<26}{'LLM-only':>14}{'Hybrid':>14}{'Delta':>14}")
    for key, label in (
        ("accuracy", "Decision accuracy"),
        ("positive_recall", "Positive recall"),
        ("negative_preservation", "Negative preservation"),
        ("over_edit_rate", "Over-edit rate"),
    ):
        delta = comparison["delta"][key]
        print(
            f"{label:<26}{pct(baseline[key]):>14}"
            f"{pct(hybrid[key]):>14}{pct(delta):>14}"
        )

    if hybrid["accuracy"] + 1e-12 < baseline["accuracy"]:
        raise AssertionError("Hybrid accuracy decreased.")
    if (
        baseline["positive_recall"] is not None
        and hybrid["positive_recall"] + 1e-12 < baseline["positive_recall"]
    ):
        raise AssertionError("Hybrid Positive Recall decreased.")
    if (
        baseline["negative_preservation"] is not None
        and hybrid["negative_preservation"] + 1e-12
        < baseline["negative_preservation"]
    ):
        raise AssertionError("Hybrid Negative Preservation decreased.")

    print("\nPASS: hybrid did not reduce the three frozen decision metrics.")


if __name__ == "__main__":
    main()
