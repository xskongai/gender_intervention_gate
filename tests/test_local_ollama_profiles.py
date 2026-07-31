from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_local_profiles_are_present_and_use_local_endpoint() -> None:
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
        assert profile["base_url"] == "http://127.0.0.1:11434/v1"
        assert profile["allow_missing_key"] is True
        assert profile["runtime"] == "ollama"
        assert profile["temperature"] == 0.0


def test_reasoning_capable_local_profiles_disable_thinking() -> None:
    models = yaml.safe_load(
        (ROOT / "configs/models.yaml").read_text(encoding="utf-8")
    )["models"]
    assert models["qwen3_5_9b_ollama"]["extra_body"] == {
        "reasoning_effort": "none"
    }
    assert models["deepseek_r1_8b_ollama"]["extra_body"] == {
        "reasoning_effort": "none"
    }
