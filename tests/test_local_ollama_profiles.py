from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_local_profiles_are_present() -> None:
    models = yaml.safe_load(
        (ROOT / "configs/models.yaml").read_text(encoding="utf-8")
    )["models"]
    keys = {
        "qwen3_5_9b_ollama",
        "deepseek_r1_8b_ollama",
        "glm4_9b_ollama",
        "gemma2_9b_ollama",
        "llama3_1_8b_ollama",
        "mistral_7b_ollama",
    }
    assert keys <= models.keys()
    for key in keys:
        profile = models[key]
        assert profile["allow_missing_key"] is True
        assert profile["runtime"] == "ollama"
        assert profile["temperature"] == 0.0


def test_qwen_uses_openai_compat_and_disables_thinking() -> None:
    models = yaml.safe_load(
        (ROOT / "configs/models.yaml").read_text(encoding="utf-8")
    )["models"]
    qwen = models["qwen3_5_9b_ollama"]
    assert qwen["provider"] == "openai_compatible"
    assert qwen["base_url"] == "http://127.0.0.1:11434/v1"
    assert qwen["extra_body"] == {"reasoning_effort": "none"}


def test_deepseek_uses_native_plain_with_large_budget() -> None:
    models = yaml.safe_load(
        (ROOT / "configs/models.yaml").read_text(encoding="utf-8")
    )["models"]
    profile = models["deepseek_r1_8b_ollama"]
    assert profile["provider"] == "ollama_native"
    assert profile["base_url"] == "http://127.0.0.1:11434"
    assert profile["think"] is False
    assert profile["structured_output"] is False
    assert profile["max_tokens"] == 2048


def test_llama_uses_native_structured_output() -> None:
    models = yaml.safe_load(
        (ROOT / "configs/models.yaml").read_text(encoding="utf-8")
    )["models"]
    profile = models["llama3_1_8b_ollama"]
    assert profile["provider"] == "ollama_native"
    assert profile["base_url"] == "http://127.0.0.1:11434"
    assert profile["structured_output"] is True
