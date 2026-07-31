#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from gender_gate.cleaning import AuditRecord, audit_negative, audit_positive
from gender_gate.data import write_jsonl


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_queue_csv(path: Path, records: list[AuditRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "priority",
        "score",
        "id",
        "current_label",
        "suggested_action",
        "text",
        "l1",
        "l2",
        "current_reason",
        "flags",
        "controversial",
        "difficulty",
        "register_group",
        "register",
        "original_split",
        "reviewer_decision",
        "reviewer_reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "priority": record.priority,
                    "score": record.score,
                    "id": record.id,
                    "current_label": record.current_label,
                    "suggested_action": record.suggested_action,
                    "text": record.text,
                    "l1": record.l1,
                    "l2": record.l2,
                    "current_reason": record.current_reason,
                    "flags": " | ".join(record.flags),
                    "controversial": record.controversial,
                    "difficulty": record.difficulty,
                    "register_group": record.register_group,
                    "register": record.register,
                    "original_split": record.original_split,
                    "reviewer_decision": "",
                    "reviewer_reason": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive-csv", default="data/raw/positive_main_v2.1.csv"
    )
    parser.add_argument(
        "--negative-csv", default="data/raw/negative_main_v2.1.csv"
    )
    parser.add_argument("--version", default="v2.1")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    positives = read_csv(root / args.positive_csv)
    negatives = read_csv(root / args.negative_csv)

    audits = [audit_positive(row) for row in positives]
    audits.extend(audit_negative(row) for row in negatives)
    queue = [record for record in audits if record.priority != "PASS"]
    queue.sort(
        key=lambda record: (
            0 if record.priority == "HIGH" else 1,
            -record.score,
            record.current_label,
            record.id,
        )
    )

    review_dir = root / "data" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        review_dir / f"audit_{args.version}.jsonl",
        (record.to_dict() for record in audits),
    )
    write_jsonl(
        review_dir / f"queue_{args.version}.jsonl",
        (record.to_dict() for record in queue),
    )
    write_queue_csv(review_dir / f"queue_{args.version}.csv", queue)

    priority_counts = Counter(record.priority for record in audits)
    label_priority: dict[str, Counter] = defaultdict(Counter)
    suggestion_counts = Counter(record.suggested_action for record in queue)
    l2_counts = Counter((record.current_label, record.l2) for record in queue)
    for record in audits:
        label_priority[record.current_label][record.priority] += 1

    summary = {
        "version": args.version,
        "total": len(audits),
        "positive": len(positives),
        "negative": len(negatives),
        "review_queue": len(queue),
        "auto_pass": priority_counts["PASS"],
        "priority_counts": dict(priority_counts),
        "label_priority": {
            label: dict(counts) for label, counts in label_priority.items()
        },
        "suggestion_counts": dict(suggestion_counts),
        "top_review_l2": [
            {"label": label, "l2": l2, "count": count}
            for (label, l2), count in l2_counts.most_common(30)
        ],
    }
    (review_dir / f"summary_{args.version}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Review queue: {review_dir / f'queue_{args.version}.csv'}")


if __name__ == "__main__":
    main()
