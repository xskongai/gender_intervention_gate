#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gender_gate.cross_model import build_judge_input


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create GPT-4o Judge input by joining rewriter outputs with the frozen rewrite-type map."
    )
    parser.add_argument("rewriter_run")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--type-map",
        default="data/review/rewrite_type_map_dev219.csv",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    count = build_judge_input(
        resolve(root, args.rewriter_run),
        resolve(root, args.type_map),
        resolve(root, args.output),
    )
    print(f"Saved {count} Judge rows to {resolve(root, args.output)}")


if __name__ == "__main__":
    main()
