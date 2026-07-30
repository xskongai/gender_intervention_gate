#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

from gender_gate.data import load_items, validate_items


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    main_items = load_items(root / "data/processed/main.jsonl")
    report = validate_items(main_items)

    expected = {"POSITIVE": 734, "NEGATIVE": 798}
    if report["label_counts"] != expected:
        report["errors"].append(
            f"Unexpected label counts: {report['label_counts']} != {expected}"
        )
        report["valid"] = False

    split_dir = root / "data/splits/iid_v1"
    split_names = ["exemplar_pool", "dev", "test"]
    split_ids = {}
    for name in split_names:
        items = load_items(split_dir / f"{name}.jsonl")
        split_ids[name] = {item.id for item in items}

    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                report["errors"].append(
                    f"Split overlap between {left} and {right}: {len(overlap)}"
                )
                report["valid"] = False

    pilot_ids = {
        item.id
        for item in load_items(split_dir / "dev_pilot_60.jsonl")
    }
    if not pilot_ids <= split_ids["dev"]:
        report["errors"].append("Pilot is not a subset of dev.")
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
