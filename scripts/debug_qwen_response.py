#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from gender_gate.clients import OpenAICompatibleClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call Qwen once and print final-content/reasoning diagnostics."
    )
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--thinking-budget", type=int, default=512)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    args = parser.parse_args()

    if args.thinking_budget <= 0 or args.max_output_tokens <= 0:
        raise ValueError("Token budgets must be positive")

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    models = yaml.safe_load(
        (root / "configs/models.yaml").read_text(encoding="utf-8")
    )["models"]

    request_config = {
        "model": args.model,
        "temperature": 0,
        "max_output_tokens": args.max_output_tokens,
        "extra_body": {"thinking_budget": args.thinking_budget},
        "retries": 1,
    }
    wrapper = OpenAICompatibleClient(models["qwen"], request_config)

    request = {
        "model": wrapper.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "判断下句是否需要性别包容改写。只输出 POSITIVE 或 NEGATIVE。\n"
                    "句子：男人有泪不轻弹。"
                ),
            }
        ],
        wrapper.max_tokens_field: wrapper.max_output_tokens,
        "extra_body": wrapper.extra_body,
    }
    if wrapper.temperature is not None:
        request["temperature"] = wrapper.temperature

    response = wrapper.client.chat.completions.create(**request)
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None)

    print(f"Provider: {wrapper.provider}")
    print(f"Model: {wrapper.model}")
    print(f"Token field: {wrapper.max_tokens_field}")
    print(f"Max output tokens: {wrapper.max_output_tokens}")
    print(f"Extra body: {wrapper.extra_body}")
    print(f"Finish reason: {getattr(choice, 'finish_reason', None)!r}")
    print(f"Reasoning chars: {len(str(reasoning)) if reasoning else 0}")
    print(f"Final content: {content!r}")
    usage = getattr(response, "usage", None)
    if usage is not None:
        print(f"Usage: {usage}")


if __name__ == "__main__":
    main()
