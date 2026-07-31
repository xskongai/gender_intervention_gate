#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_manifest(path_value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path.parent, json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("direct_run")
    parser.add_argument("gated_run")
    parser.add_argument("--output")
    args = parser.parse_args()

    direct_dir, direct = load_manifest(args.direct_run)
    gated_dir, gated = load_manifest(args.gated_run)
    if direct["split_sha256"] != gated["split_sha256"]:
        raise ValueError("Runs use different dataset splits.")
    if direct["prompt_sha256"] != gated["prompt_sha256"]:
        raise ValueError("Runs use different rewrite prompts.")
    if direct["model"] != gated["model"]:
        raise ValueError("Runs use different rewrite models.")

    dm = direct["metrics"]
    gm = gated["metrics"]
    rows = [
        ("Positive intervention rate", dm["positive_intervention_rate"], gm["positive_intervention_rate"]),
        ("Under-edit rate", dm["under_edit_rate"], gm["under_edit_rate"]),
        ("Negative preservation", dm["negative_preservation"], gm["negative_preservation"]),
        ("Over-edit rate", dm["over_edit_rate"], gm["over_edit_rate"]),
        ("Rewrite call rate", dm["rewrite_call_rate"], gm["rewrite_call_rate"]),
        ("Error rate", dm["error_rate"], gm["error_rate"]),
    ]

    lines = [
        "# Direct Rewrite vs Gate + Rewrite",
        "",
        f"- Direct run: `{direct_dir}`",
        f"- Gated run: `{gated_dir}`",
        f"- Count: {dm['count']}",
        "",
        "| Metric | Direct | Gated | Gated − Direct |",
        "|---|---:|---:|---:|",
    ]
    for name, direct_value, gated_value in rows:
        lines.append(
            f"| {name} | {pct(direct_value)} | {pct(gated_value)} | "
            f"{(gated_value-direct_value)*100:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## API usage",
            "",
            f"- Direct rewrite calls: {dm['rewrite_calls']}",
            f"- Gated rewrite calls: {gm['rewrite_calls']}",
            f"- Calls saved by Gate: {dm['rewrite_calls'] - gm['rewrite_calls']}",
        ]
    )
    output = "\n".join(lines) + "\n"
    print(output)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
