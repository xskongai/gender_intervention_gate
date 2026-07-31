#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from gender_gate.data import load_items, validate_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/main.jsonl")
    parser.add_argument("--positive", type=int)
    parser.add_argument("--negative", type=int)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    main_items = load_items(root / args.input)
    report = validate_items(main_items)

    expected = {}
    if args.positive is not None:
        expected["POSITIVE"] = args.positive
    if args.negative is not None:
        expected["NEGATIVE"] = args.negative
    if expected:
        actual = report["label_counts"]
        for label, count in expected.items():
            if actual.get(label, 0) != count:
                report["errors"].append(
                    f"Unexpected {label} count: {actual.get(label, 0)} != {count}"
                )
                report["valid"] = False

    print(f"Count: {report['count']}")
    print(f"Labels: {Counter(item.label for item in main_items)}")
    print(f"Valid: {report['valid']}")
    for error in report["errors"]:
        print(f"- {error}")

    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
