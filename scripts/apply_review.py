#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from gender_gate.data import read_jsonl, write_jsonl


def read_csv(path: Path, label: str, source: str) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "id": raw["编号"].strip(),
                    "text": raw["输入句子"].strip(),
                    "label": label,
                    "meta": {
                        "l1": (raw.get("L1类别") or "").strip(),
                        "l2": (raw.get("L2子类") or "").strip(),
                        "register_group": (raw.get("语体大类") or "").strip(),
                        "register": (raw.get("语体") or "").strip(),
                        "noise": (raw.get("噪声类型") or "").strip(),
                        "difficulty": (raw.get("难度") or "").strip(),
                        "controversial": (raw.get("是否争议") or "").strip(),
                        "source": source,
                        "original_split": (raw.get("切分") or "").strip(),
                    },
                }
            )
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-csv", default="data/raw/positive_main_v2.1.csv")
    parser.add_argument("--negative-csv", default="data/raw/negative_main_v2.1.csv")
    parser.add_argument("--decisions", default="data/review/decisions_v2.1.jsonl")
    parser.add_argument("--output", default="data/processed/main_v2.2.jsonl")
    parser.add_argument("--change-log", default="data/review/change_log_v2.2.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    items = read_csv(root / args.positive_csv, "POSITIVE", "positive_v2.1.xlsx")
    items.extend(read_csv(root / args.negative_csv, "NEGATIVE", "negative_v2.1.xlsx"))

    decisions_path = root / args.decisions
    decisions = (
        {row["id"]: row for row in read_jsonl(decisions_path)}
        if decisions_path.exists()
        else {}
    )

    output: list[dict] = []
    changes: list[dict] = []

    for item in items:
        decision = decisions.get(item["id"])
        if not decision or decision["decision"] in {"KEEP_CURRENT", "DEFER"}:
            if decision:
                item["meta"]["review_status"] = decision["decision"]
                item["meta"]["review_reason"] = decision.get("reviewer_reason", "")
            output.append(item)
            continue

        old_label = item["label"]
        action = decision["decision"]
        if action == "DELETE":
            new_label = "DELETED"
        elif action == "MOVE_TO_POSITIVE":
            item["label"] = "POSITIVE"
            new_label = "POSITIVE"
            output.append(item)
        elif action == "MOVE_TO_NEGATIVE":
            item["label"] = "NEGATIVE"
            new_label = "NEGATIVE"
            output.append(item)
        else:
            raise ValueError(f"Unsupported decision: {action}")

        changes.append(
            {
                "id": item["id"],
                "text": item["text"],
                "old_label": old_label,
                "new_label": new_label,
                "decision": action,
                "reviewer_reason": decision.get("reviewer_reason", ""),
                "reviewed_at": decision.get("reviewed_at", ""),
            }
        )
        if new_label != "DELETED":
            item["meta"]["review_status"] = action
            item["meta"]["review_reason"] = decision.get("reviewer_reason", "")

    ids = [row["id"] for row in output]
    texts = [row["text"] for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs after applying review.")
    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate texts after applying review.")

    write_jsonl(root / args.output, output)

    log_path = root / args.change_log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "id",
            "text",
            "old_label",
            "new_label",
            "decision",
            "reviewer_reason",
            "reviewed_at",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changes)

    counts = Counter(row["label"] for row in output)
    print(
        json.dumps(
            {
                "output": str(root / args.output),
                "count": len(output),
                "labels": dict(counts),
                "changes": len(changes),
                "change_log": str(log_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
