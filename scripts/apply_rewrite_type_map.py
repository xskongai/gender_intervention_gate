#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ALLOWED_TYPES = {"LOCAL_REPAIR", "PROPOSITION_RECONSTRUCTION"}


def normalize_rewrite_type(value: str) -> str:
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized not in ALLOWED_TYPES:
        raise ValueError(
            f"Unknown rewrite_type {value!r}; expected LOCAL_REPAIR "
            "or PROPOSITION_RECONSTRUCTION"
        )
    return normalized


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a fixed rewrite-type map to a prepared Judge input CSV."
    )
    parser.add_argument("--input", required=True, help="Prepared Judge input CSV")
    parser.add_argument("--type-map", required=True, help="CSV with id and rewrite_type")
    parser.add_argument("--output", required=True, help="Filled output CSV")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    map_path = Path(args.type_map).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    fields, rows = read_rows(input_path)
    map_fields, map_rows = read_rows(map_path)

    required_input = {"id", "text", "rewrite_type"}
    required_map = {"id", "rewrite_type"}
    if not required_input.issubset(fields):
        raise ValueError(f"Input missing fields: {sorted(required_input - set(fields))}")
    if not required_map.issubset(map_fields):
        raise ValueError(f"Type map missing fields: {sorted(required_map - set(map_fields))}")

    mapping: dict[str, dict[str, str]] = {}
    for row in map_rows:
        item_id = row["id"].strip()
        if not item_id:
            raise ValueError("Blank id in type map")
        if item_id in mapping:
            raise ValueError(f"Duplicate id in type map: {item_id}")
        normalize_rewrite_type(row["rewrite_type"])
        mapping[item_id] = row

    if "type_note" not in fields:
        fields.append("type_note")

    missing: list[str] = []
    text_mismatches: list[str] = []
    for row in rows:
        item_id = row["id"].strip()
        mapped = mapping.get(item_id)
        if mapped is None:
            missing.append(item_id)
            continue
        if mapped.get("text", "").strip() and mapped["text"].strip() != row["text"].strip():
            text_mismatches.append(item_id)
        row["rewrite_type"] = normalize_rewrite_type(mapped["rewrite_type"])
        row["type_note"] = mapped.get("type_note", "")

    if missing:
        raise ValueError(f"Type map missing {len(missing)} ids; first: {missing[:5]}")
    if text_mismatches:
        raise ValueError(
            f"Text mismatch for {len(text_mismatches)} ids; first: {text_mismatches[:5]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    local = sum(row["rewrite_type"] == "LOCAL_REPAIR" for row in rows)
    reconstruction = sum(
        row["rewrite_type"] == "PROPOSITION_RECONSTRUCTION" for row in rows
    )
    print(f"Saved: {output_path}")
    print(f"Rows: {len(rows)}")
    print(f"LOCAL_REPAIR: {local}")
    print(f"PROPOSITION_RECONSTRUCTION: {reconstruction}")


if __name__ == "__main__":
    main()
