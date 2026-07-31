#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from gender_gate.clients import OpenAICompatibleClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Make one minimal provider access check.")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model")
    parser.add_argument("--thinking-budget", type=int)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    models = yaml.safe_load(
        (root / "configs/models.yaml").read_text(encoding="utf-8")
    )["models"]
    if args.provider not in models:
        raise KeyError(f"Unknown provider: {args.provider}")

    profile = models[args.provider]
    request_config = {
        "model": args.model,
        "temperature": profile.get("temperature", 0),
        "max_output_tokens": int(profile.get("max_tokens", 2048)),
        "max_tokens_field": profile.get("max_tokens_field", "max_tokens"),
        "retries": 1,
    }
    if args.thinking_budget is not None:
        if args.thinking_budget <= 0:
            raise ValueError("--thinking-budget must be positive")
        request_config["extra_body"] = {"thinking_budget": args.thinking_budget}
    elif args.provider == "qwen":
        request_config["extra_body"] = {"thinking_budget": 128}
    client = OpenAICompatibleClient(models[args.provider], request_config)
    output = client.complete(
        [{"role": "user", "content": "只输出两个大写字母：OK"}]
    )
    print(f"Provider: {args.provider}")
    print(f"Model: {client.model}")
    print(f"Output: {output.strip()}")


if __name__ == "__main__":
    main()
