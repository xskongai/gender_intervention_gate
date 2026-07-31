from pathlib import Path
import sys
import types

import yaml

from gender_gate.clients import OpenAICompatibleClient


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_uses_completion_budget_and_bounded_thinking() -> None:
    models = yaml.safe_load(
        (ROOT / "configs/models.yaml").read_text(encoding="utf-8")
    )["models"]
    qwen = models["qwen"]
    assert qwen["max_tokens_field"] == "max_completion_tokens"
    assert qwen["extra_body"]["enable_thinking"] is True
    assert qwen["extra_body"]["thinking_budget"] == 512


def test_extra_body_is_merged(monkeypatch) -> None:
    monkeypatch.setenv("TEST_QWEN_KEY", "dummy")

    class DummyOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_openai = types.SimpleNamespace(OpenAI=DummyOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    client = OpenAICompatibleClient(
        {
            "provider": "qwen",
            "api_key_env": "TEST_QWEN_KEY",
            "model": "qwen3.7-plus",
            "max_tokens_field": "max_completion_tokens",
            "extra_body": {
                "enable_thinking": True,
                "thinking_budget": 512,
            },
        },
        {
            "max_output_tokens": 2048,
            "extra_body": {"thinking_budget": 256},
        },
    )
    assert client.max_tokens_field == "max_completion_tokens"
    assert client.extra_body == {
        "enable_thinking": True,
        "thinking_budget": 256,
    }
