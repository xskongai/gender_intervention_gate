#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from gender_gate.data import read_jsonl, write_jsonl

VALID = {
    "k": "KEEP_CURRENT",
    "p": "MOVE_TO_POSITIVE",
    "n": "MOVE_TO_NEGATIVE",
    "d": "DELETE",
    "s": "DEFER",
}


def load_decisions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def save_decisions(path: Path, decisions: dict[str, dict]) -> None:
    rows = sorted(decisions.values(), key=lambda row: row["id"])
    write_jsonl(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="data/review/queue_v2.1.jsonl")
    parser.add_argument("--decisions", default="data/review/decisions_v2.1.jsonl")
    parser.add_argument("--priority", choices=["HIGH", "MEDIUM", "ALL"], default="HIGH")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    queue = read_jsonl(root / args.queue)
    decisions_path = root / args.decisions
    decisions = load_decisions(decisions_path)

    candidates = [
        row
        for row in queue
        if args.priority == "ALL" or row["priority"] == args.priority
    ]
    candidates = [row for row in candidates if row["id"] not in decisions]
    if args.limit is not None:
        candidates = candidates[: args.limit]

    print("Commands: k=keep current, p=positive, n=negative, d=delete, s=defer, q=quit")
    print(f"Remaining candidates: {len(candidates)}")

    for index, row in enumerate(candidates, start=1):
        print("\n" + "=" * 88)
        print(f"[{index}/{len(candidates)}] {row['priority']} score={row['score']} {row['id']}")
        print(f"Current: {row['current_label']} | Suggested: {row['suggested_action']}")
        print(f"Category: {row['l1']} / {row['l2']}")
        print(f"Text: {row['text']}")
        print(f"Current reason: {row['current_reason']}")
        print("Flags: " + " | ".join(row.get("flags") or []))

        while True:
            command = input("Decision [k/p/n/d/s/q]: ").strip().lower()
            if command == "q":
                save_decisions(decisions_path, decisions)
                print(f"Saved: {decisions_path}")
                return
            if command in VALID:
                break
            print("Invalid command.")

        reason = input("Reason (optional): ").strip()
        decisions[row["id"]] = {
            "id": row["id"],
            "original_label": row["current_label"],
            "decision": VALID[command],
            "reviewer_reason": reason,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        save_decisions(decisions_path, decisions)

    print(f"Completed. Decisions saved to {decisions_path}")


if __name__ == "__main__":
    main()
