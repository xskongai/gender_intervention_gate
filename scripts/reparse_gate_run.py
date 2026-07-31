#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gender_gate.metrics import calculate_metrics
from gender_gate.parser import parse_label
from gender_gate.reports import generate_reports


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reparse an existing Gate run with the current common parser. "
            "The original run is never modified."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-name", default="reparsed_common_parser_v10")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    input_path = run_dir / "predictions.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    rows = load_jsonl(input_path)
    before = calculate_metrics(rows)
    recovered = 0
    unresolved_empty = 0
    unresolved_runtime = 0
    unresolved_format = 0

    updated: list[dict] = []
    for original in rows:
        row = dict(original)
        if row.get("predicted") is None:
            parsed = parse_label(str(row.get("raw_output") or ""))
            if parsed is not None:
                row["predicted"] = parsed
                row["error"] = None
                row["reparsed"] = True
                recovered += 1
            else:
                raw = str(row.get("raw_output") or "")
                error = str(row.get("error") or "")
                if not raw.strip():
                    unresolved_empty += 1
                elif "RuntimeError" in error:
                    unresolved_runtime += 1
                else:
                    unresolved_format += 1
        updated.append(row)

    output_dir = run_dir / args.output_name
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in updated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    after = generate_reports(output_dir, updated)
    diagnostic = {
        "source_run": str(run_dir),
        "recovered": recovered,
        "unresolved_empty": unresolved_empty,
        "unresolved_runtime": unresolved_runtime,
        "unresolved_format": unresolved_format,
        "before": before,
        "after": after,
    }
    (output_dir / "reparse_summary.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Source: {run_dir}")
    print(f"Recovered format outputs: {recovered}")
    print(f"Unresolved empty outputs: {unresolved_empty}")
    print(f"Unresolved runtime errors: {unresolved_runtime}")
    print(f"Unresolved format outputs: {unresolved_format}")
    print(
        "Before: "
        f"Pos={before['positive_recall']:.4f} "
        f"Neg={before['negative_recall']:.4f} "
        f"BA={before['balanced_accuracy']:.4f} "
        f"Format={before['format_error_rate']:.4f}"
    )
    print(
        "After:  "
        f"Pos={after['positive_recall']:.4f} "
        f"Neg={after['negative_recall']:.4f} "
        f"BA={after['balanced_accuracy']:.4f} "
        f"Format={after['format_error_rate']:.4f}"
    )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
