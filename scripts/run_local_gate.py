#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

LOCAL_MODELS = {
    "qwen": "qwen3_5_9b_ollama",
    "deepseek": "deepseek_r1_8b_ollama",
    "glm": "glm4_9b_ollama",
    "gemma": "gemma2_9b_ollama",
    "llama": "llama3_1_8b_ollama",
    "mistral": "mistral_7b_ollama",
}

STAGES = {
    "smoke20": "data/splits/group_aware_v2.3/dev_smoke_20.jsonl",
    "pilot60": "data/splits/group_aware_v2.3/dev_pilot_60.jsonl",
    "dev400": "data/splits/group_aware_v2.3/dev.jsonl",
}

CONFIG = "configs/experiments/contrastive_fewshot_rule_first.yaml"


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Rule-first Gate with local Ollama 7B-9B profiles. "
            "Reasoning is disabled for Qwen3.5 and DeepSeek-R1."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", *LOCAL_MODELS],
        help="Local model aliases to run sequentially.",
    )
    parser.add_argument("--stage", choices=STAGES, default="smoke20")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.concurrency <= 0 or args.progress_every <= 0:
        parser.error("--concurrency and --progress-every must be positive")

    selected = list(LOCAL_MODELS) if "all" in args.models else args.models
    root = Path(__file__).resolve().parents[1]
    profiles = yaml.safe_load(
        (root / "configs/models.yaml").read_text(encoding="utf-8")
    )["models"]

    for alias in selected:
        model_key = LOCAL_MODELS[alias]
        profile = profiles[model_key]
        model = profile["model"]
        max_tokens = int(profile["max_tokens"])
        name = f"{model_key}_gate_{args.stage}_nothink"
        command = [
            sys.executable,
            "-u",
            "scripts/run_experiment.py",
            "--config",
            CONFIG,
            "--split",
            STAGES[args.stage],
            "--model-key",
            model_key,
            "--rule-first",
            "--concurrency",
            str(args.concurrency),
            "--progress-every",
            str(args.progress_every),
            "--max-output-tokens",
            str(max_tokens),
            "--name",
            name,
        ]
        print("=" * 72, flush=True)
        print(f"{alias}: {model} | max_tokens={max_tokens}", flush=True)
        print(f"$ {command_text(command)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
