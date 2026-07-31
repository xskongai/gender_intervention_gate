#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_predictions(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a rewrite-type annotation template for LLM Judge."
    )
    parser.add_argument("run_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    predictions = [p for p in load_predictions(run_dir) if p.get("gold") == "POSITIVE"]
    rows = [
        {
            "id": p["id"],
            "text": p["text"],
            "output": p["final_output"],
            "rewrite_type": "",
            "type_note": "",
        }
        for p in predictions
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output}")
    print("Fill rewrite_type with LOCAL_REPAIR or PROPOSITION_RECONSTRUCTION.")


if __name__ == "__main__":
    main()
