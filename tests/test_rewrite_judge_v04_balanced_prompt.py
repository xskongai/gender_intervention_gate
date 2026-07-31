from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_balanced_judge_config_uses_combined_prompt() -> None:
    config_path = ROOT / "configs/judge/rewrite_judge_v04_balanced_gpt4o.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["name"] == "rewrite_judge_v04_balanced"
    assert "judge_mode" not in config
    assert (ROOT / config["prompt"]).exists()


def test_balanced_prompt_prevents_systematic_over_penalization() -> None:
    prompt = (ROOT / "prompts/rewrite_judge_v04_balanced.txt").read_text(
        encoding="utf-8"
    )
    assert "3 分表示改写已经合理、正确、可接受" in prompt
    assert "同一个问题不要在多个维度重复扣分" in prompt
    assert "不得被视为信息损失" in prompt
    assert "不要为了严格而寻找细小缺点" in prompt
    assert "不要求再次明确提到性别" in prompt
