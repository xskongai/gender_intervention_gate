#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    errors = [
        p for p in predictions
        if p.get("predicted") != p["gold"]
    ]

    by_l1 = Counter(
        (p["gold"], (p.get("meta") or {}).get("l1"))
        for p in errors
    )
    by_l2 = Counter(
        (p["gold"], (p.get("meta") or {}).get("l2"))
        for p in errors
    )

    rows = [
        {"gold": gold, "l1": l1, "errors": count}
        for (gold, l1), count in by_l1.most_common()
    ]
    with (run_dir / "error_matrix_l1.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["gold", "l1", "errors"],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Error Analysis",
        "",
        f"Total errors: {len(errors)}",
        "",
        "## Top L1 error groups",
        "",
        "| Gold | L1 | Errors |",
        "|---|---|---:|",
    ]
    for (gold, l1), count in by_l1.most_common(15):
        lines.append(f"| {gold} | {l1} | {count} |")

    lines += [
        "",
        "## Top L2 error groups",
        "",
        "| Gold | L2 | Errors |",
        "|---|---|---:|",
    ]
    for (gold, l2), count in by_l2.most_common(25):
        lines.append(f"| {gold} | {l2} | {count} |")

    (run_dir / "error_analysis.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"Wrote error analysis to {run_dir}")


if __name__ == "__main__":
    main()
