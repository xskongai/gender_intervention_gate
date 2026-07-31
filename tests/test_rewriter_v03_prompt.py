from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts" / "rewriter_v03_minimal_sufficient.txt"
CONFIG = ROOT / "configs" / "rewriter" / "rewriter_v03_gpt4o.yaml"


def test_v03_prompt_contains_minimal_sufficient_constraints() -> None:
    text = PROMPT.read_text(encoding="utf-8")
    assert "完整消除核心性别偏见命题" in text
    assert "尽量减少对原句其他内容的改动" in text
    assert "局部修复" in text
    assert "最小命题重构" in text
    assert "不能把有害规范从某一性别泛化给所有人" in text
    assert "不要新增健康、安全、道德或价值判断" in text
    assert "{{TEXT}}" in text


def test_v03_prompt_does_not_copy_exact_dev_diagnostic() -> None:
    text = PROMPT.read_text(encoding="utf-8")
    assert "一个大老爷们，怎么连家都养不起。" not in text


def test_v03_config_points_to_new_prompt_and_full_dev() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["name"] == "rewriter_v03_minimal_sufficient"
    assert config["prompt"] == "prompts/rewriter_v03_minimal_sufficient.txt"
    assert config["split"] == "data/splits/group_aware_v2.3/dev_positive_219.jsonl"
    assert config["temperature"] == 0
