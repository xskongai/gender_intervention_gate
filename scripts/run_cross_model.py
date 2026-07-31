#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml
from dotenv import load_dotenv

from gender_gate.clients import ClientConfigurationError, resolve_model_name
from gender_gate.cross_model import (
    STAGES,
    StageSpec,
    build_judge_input,
    summarize_suite,
    summary_markdown,
)


TARGET_PROVIDERS = ("openai", "gemini", "deepseek", "qwen", "glm", "local")
GATE_CONFIG = "configs/experiments/contrastive_fewshot_rule_first.yaml"
REWRITER_CONFIG = "configs/rewriter/rewriter_v02_gpt4o.yaml"
JUDGE_CONFIG = "configs/judge/rewrite_judge_v04_balanced_gpt4o.yaml"
TYPE_MAP = "data/review/rewrite_type_map_dev219.csv"
GATE_MAX_OUTPUT_TOKENS = 2048
REWRITER_MAX_OUTPUT_TOKENS = 4096
QWEN_GATE_THINKING_BUDGET = 512
QWEN_REWRITER_THINKING_BUDGET = 1536


def command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_and_find_directory(
    command: list[str], root: Path, dry_run: bool = False
) -> Path | None:
    print(f"\n$ {command_text(command)}", flush=True)
    if dry_run:
        return None

    process = subprocess.Popen(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    run_dir: Path | None = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith("Run directory:"):
            value = line.split(":", 1)[1].strip()
            candidate = Path(value)
            run_dir = candidate if candidate.is_absolute() else root / candidate
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if run_dir is None:
        raise RuntimeError("Child process succeeded but did not print its run directory")
    return run_dir.resolve()


def load_models(root: Path) -> dict:
    path = root / "configs/models.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["models"]


def resolve_provider_model(
    provider: str, model_override: str | None, models: dict, require_key: bool = True
) -> str:
    if provider not in models:
        raise KeyError(f"Unknown provider/model key: {provider}")
    request = {"model": model_override} if model_override else {}
    model = resolve_model_name(models[provider], request)
    if not model:
        model_env = models[provider].get("model_env")
        raise ClientConfigurationError(
            f"Missing target model for {provider}. Set {model_env} or pass --model."
        )
    api_key_env = models[provider].get("api_key_env")
    if require_key and not models[provider].get("allow_missing_key") and not os.getenv(
        str(api_key_env), ""
    ):
        raise ClientConfigurationError(
            f"Missing API key for {provider}. Set {api_key_env}."
        )
    return model


def check_judge(
    root: Path, models: dict, skip_judge: bool, require_key: bool = True
) -> None:
    if skip_judge:
        return
    judge_config = yaml.safe_load(
        (root / JUDGE_CONFIG).read_text(encoding="utf-8")
    )
    judge_key = str(judge_config["model_key"])
    if judge_key != "openai_judge":
        raise ValueError(
            f"Judge config must use openai_judge, found {judge_key}"
        )
    model = resolve_model_name(models[judge_key], judge_config)
    if model != "gpt-4o":
        raise ValueError(f"Judge must remain fixed at gpt-4o, found {model}")
    key_env = str(models[judge_key]["api_key_env"])
    if require_key and not os.getenv(key_env, ""):
        raise ClientConfigurationError(
            f"Missing GPT-4o Judge API key. Set {key_env}."
        )


def run_provider(
    root: Path,
    provider: str,
    stage: StageSpec,
    model: str,
    concurrency: int | None,
    skip_judge: bool,
    dry_run: bool,
) -> dict | None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_dir = root / "runs" / "cross_model" / f"{stamp}_{provider}_{stage.name}"
    if not dry_run:
        suite_dir.mkdir(parents=True, exist_ok=False)

    common_model_args = ["--model-key", provider, "--model", model]
    gate_name = f"crossmodel_{provider}_{stage.name}_gate"
    gate_command = [
        sys.executable,
        "scripts/run_experiment.py",
        "--config",
        GATE_CONFIG,
        "--split",
        stage.gate_split,
        *common_model_args,
        "--name",
        gate_name,
        "--rule-first",
        "--max-output-tokens",
        str(GATE_MAX_OUTPUT_TOKENS),
    ]
    if provider == "qwen":
        gate_command.extend(["--thinking-budget", str(QWEN_GATE_THINKING_BUDGET)])
    if concurrency is not None:
        gate_command.extend(["--concurrency", str(concurrency)])

    rewriter_name = f"crossmodel_{provider}_{stage.name}"
    rewriter_command = [
        sys.executable,
        "scripts/run_rewriter_experiment.py",
        "--config",
        REWRITER_CONFIG,
        "--split",
        stage.rewriter_split,
        *common_model_args,
        "--name",
        rewriter_name,
        "--max-output-tokens",
        str(REWRITER_MAX_OUTPUT_TOKENS),
    ]
    if provider == "qwen":
        rewriter_command.extend(
            ["--thinking-budget", str(QWEN_REWRITER_THINKING_BUDGET)]
        )
    if concurrency is not None:
        rewriter_command.extend(["--concurrency", str(concurrency)])

    gate_run = run_and_find_directory(gate_command, root, dry_run)
    rewriter_run = run_and_find_directory(rewriter_command, root, dry_run)

    judge_run: Path | None = None
    if dry_run:
        judge_input = suite_dir / "judge_input.csv"
    else:
        assert gate_run is not None and rewriter_run is not None
        judge_input = suite_dir / "judge_input.csv"
        count = build_judge_input(
            rewriter_run,
            root / TYPE_MAP,
            judge_input,
        )
        if count != stage.positive_count:
            raise ValueError(
                f"Expected {stage.positive_count} Judge rows, created {count}"
            )
        print(f"Prepared GPT-4o Judge input: {judge_input} ({count} rows)")

    if not skip_judge:
        judge_command = [
            sys.executable,
            "scripts/run_rewrite_judge.py",
            "--config",
            JUDGE_CONFIG,
            "--input",
            str(judge_input),
            "--name",
            f"crossmodel_{provider}_{stage.name}_gpt4o",
        ]
        if concurrency is not None:
            judge_command.extend(["--concurrency", str(concurrency)])
        judge_run = run_and_find_directory(judge_command, root, dry_run)

    if dry_run:
        return None

    assert gate_run is not None and rewriter_run is not None
    result = summarize_suite(
        provider=provider,
        stage=stage,
        target_model=model,
        gate_run=gate_run,
        rewriter_run=rewriter_run,
        judge_run=judge_run,
    )
    (suite_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (suite_dir / "summary.md").write_text(
        summary_markdown(result), encoding="utf-8"
    )
    print(f"\nSuite directory: {suite_dir}")
    print(summary_markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Rule-first + Frozen LLM Gate and the frozen constrained Rewriter "
            "with one or more target model providers; score rewrites with fixed GPT-4o."
        )
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        required=True,
        choices=TARGET_PROVIDERS,
        help="Target providers to run sequentially.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=tuple(STAGES),
        help="smoke20 -> pilot60 -> dev400",
    )
    parser.add_argument(
        "--model",
        help="Exact target model id. Allowed only when one provider is selected.",
    )
    parser.add_argument("--concurrency", type=int)
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Run Gate and Rewriter only. Normal paper runs should keep GPT-4o Judge enabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print all commands without API calls.",
    )
    args = parser.parse_args()

    if args.model and len(args.providers) != 1:
        parser.error("--model can be used only with exactly one provider")
    if args.concurrency is not None and args.concurrency <= 0:
        parser.error("--concurrency must be positive")

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    models = load_models(root)
    check_judge(root, models, args.skip_judge, require_key=not args.dry_run)
    stage = STAGES[args.stage]

    print("Cross-model experiment")
    print(f"Stage: {stage.name}")
    print(f"Gate items: {stage.gate_count}")
    print(f"Positive rewrites/Judge items: {stage.positive_count}")
    print("Gate: Rule-first + Frozen LLM Gate")
    print("Rewriter: frozen v02 semantic-preserving")
    print("Judge: fixed GPT-4o v04 Balanced")
    print(f"Gate max output tokens: {GATE_MAX_OUTPUT_TOKENS}")
    print(f"Rewriter max output tokens: {REWRITER_MAX_OUTPUT_TOKENS}")
    print(
        "Qwen thinking budgets: "
        f"Gate={QWEN_GATE_THINKING_BUDGET}, "
        f"Rewriter={QWEN_REWRITER_THINKING_BUDGET}"
    )

    all_results: list[dict] = []
    for provider in args.providers:
        target_model = resolve_provider_model(
            provider,
            args.model if len(args.providers) == 1 else None,
            models,
            require_key=not args.dry_run,
        )
        print(f"\n=== {provider}: {target_model} ===")
        result = run_provider(
            root=root,
            provider=provider,
            stage=stage,
            model=target_model,
            concurrency=args.concurrency,
            skip_judge=args.skip_judge,
            dry_run=args.dry_run,
        )
        if result is not None:
            all_results.append(result)

    if all_results:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        matrix_path = (
            root
            / "runs"
            / "cross_model"
            / f"{stamp}_{stage.name}_matrix_summary.json"
        )
        matrix_path.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        csv_path = matrix_path.with_suffix(".csv")
        rows = []
        for result in all_results:
            gate = result["gate"]
            rewriter = result["rewriter"]
            judge = result.get("judge") or {}
            rows.append(
                {
                    "provider": result["provider"],
                    "stage": result["stage"],
                    "target_model": result["target_model"],
                    "positive_recall": gate["positive_recall"],
                    "negative_preservation": gate["negative_preservation"],
                    "balanced_accuracy": gate["balanced_accuracy"],
                    "rule_coverage": gate["rule_coverage"],
                    "llm_call_rate": gate["llm_call_rate"],
                    "gate_format_error_rate": gate["format_error_rate"],
                    "rewriter_error_rate": rewriter["error_rate"],
                    "overall_quality": judge.get("overall_quality"),
                    "macro_quality": judge.get("macro_quality"),
                    "debiasing": judge.get("debiasing"),
                    "naturalness": judge.get("naturalness"),
                    "type_specific": judge.get("type_specific"),
                    "pass_rate": judge.get("pass_rate"),
                    "judge_model": result["judge_model"],
                }
            )
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Matrix summary: {matrix_path}")
        print(f"Matrix CSV: {csv_path}")


if __name__ == "__main__":
    main()
