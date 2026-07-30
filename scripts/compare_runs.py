#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    rows = []
    for value in args.run_dirs:
        run_dir = Path(value)
        metrics = json.loads(
            (run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        rows.append((run_dir.name, metrics))

    print(
        "| Run | Pos Recall | Neg Recall | "
        "Balanced Acc | Macro-F1 | Pass 90% |"
    )
    print("|---|---:|---:|---:|---:|---|")
    for name, metrics in rows:
        print(
            f"| {name} | {metrics['positive_recall']:.4f} | "
            f"{metrics['negative_recall']:.4f} | "
            f"{metrics['balanced_accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | "
            f"{metrics['passes_90_target']} |"
        )


if __name__ == "__main__":
    main()
