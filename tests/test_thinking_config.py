from __future__ import annotations

from pathlib import Path

import yaml

from gender_gate.cache import request_key


ROOT = Path(__file__).resolve().parents[1]


def test_requested_models_have_explicit_thinking_enabled() -> None:
    models = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8"))["models"]
    assert models["deepseek"]["extra_body"] == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    assert models["qwen"]["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": 512,
    }
    assert models["glm"]["extra_body"] == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }


def test_thinking_and_non_thinking_requests_use_different_cache_keys() -> None:
    base = {"provider": "qwen", "model": "qwen3.7-plus", "messages": [{"role": "user", "content": "x"}]}
    non_thinking = {**base, "extra_body": {"enable_thinking": False}}
    thinking = {**base, "extra_body": {"enable_thinking": True}}
    assert request_key(non_thinking) != request_key(thinking)


def test_thinking_token_limits_are_not_tiny() -> None:
    gate = yaml.safe_load((ROOT / "configs/experiments/contrastive_fewshot_rule_first.yaml").read_text(encoding="utf-8"))
    rewriter = yaml.safe_load((ROOT / "configs/rewriter/rewriter_v02_gpt4o.yaml").read_text(encoding="utf-8"))
    assert gate["max_output_tokens"] >= 2048
    assert rewriter["max_output_tokens"] >= 4096
