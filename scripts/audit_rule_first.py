#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from gender_gate.data import load_items
from gender_gate.deterministic_rules import deterministic_label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline audit of deterministic Rule-first routing."
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    split_path = root / args.split
    items = load_items(split_path)

    rows: list[dict[str, str]] = []
    rule_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for item in items:
        result = deterministic_label(item.text)
        if result is None:
            continue
        correct = result["label"] == item.label
        rows.append(
            {
                "id": item.id,
                "text": item.text,
                "gold": item.label,
                "rule_label": result["label"],
                "decision": result["decision"],
                "rule": result["rule"],
                "correct": str(correct),
            }
        )
        rule_counts[result["rule"]] += 1
        label_counts[result["label"]] += 1

    errors = [row for row in rows if row["correct"] != "True"]
    total = len(items)
    routed = len(rows)
    summary = {
        "split": str(args.split),
        "count": total,
        "rule_routed": routed,
        "rule_coverage": 0.0 if total == 0 else routed / total,
        "llm_fallback": total - routed,
        "llm_call_rate": 0.0 if total == 0 else (total - routed) / total,
        "rule_errors": len(errors),
        "observed_rule_accuracy": None if routed == 0 else (routed - len(errors)) / routed,
        "rule_counts": dict(rule_counts),
        "rule_label_counts": dict(label_counts),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        output = root / args.output
        output.mkdir(parents=True, exist_ok=True)
        (output / "rule_audit_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if rows:
            with (output / "rule_audit_rows.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if errors:
            with (output / "rule_audit_errors.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(errors[0]))
                writer.writeheader()
                writer.writerows(errors)

    if errors:
        raise SystemExit(f"Rule audit failed: {len(errors)} Gold conflicts.")


if __name__ == "__main__":
    main()
