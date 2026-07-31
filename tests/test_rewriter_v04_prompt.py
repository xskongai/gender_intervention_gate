from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts" / "rewriter_v04_branch_aware.txt"
CONFIG = ROOT / "configs" / "rewriter" / "rewriter_v04_gpt4o.yaml"


def test_v04_prompt_contains_branch_aware_constraints() -> None:
    text = PROMPT.read_text(encoding="utf-8")
    assert "局部修复充分性测试" in text
    assert "必须采用【局部修复】" in text
    assert "只有局部修复不充分时" in text
    assert "最小命题重构" in text
    assert "普通的负面评价不自动等于性别偏见" in text
    assert "普通的正面评价也不自动隐含性别偏见" in text
    assert "与性别无关的原有内容应尽量保留" in text
    assert "不得为了显得积极而新增" in text
    assert "{{TEXT}}" in text


def test_v04_prompt_distinguishes_residual_gender_norm_from_general_criticism() -> None:
    text = PROMPT.read_text(encoding="utf-8")
    assert "剩余赞美、批评、质问或描述可以独立适用于任何性别" in text
    assert "延续原来的性别角色义务、身份羞辱或行为禁令" in text
    assert "把原来的有害性别规范扩大" in text


def test_v04_prompt_does_not_copy_exact_dev_diagnostic() -> None:
    text = PROMPT.read_text(encoding="utf-8")
    assert "一个大老爷们，怎么连家都养不起。" not in text


def test_v04_config_points_to_new_prompt_and_full_dev() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["name"] == "rewriter_v04_branch_aware"
    assert config["prompt"] == "prompts/rewriter_v04_branch_aware.txt"
    assert config["split"] == "data/splits/group_aware_v2.3/dev_positive_219.jsonl"
    assert config["temperature"] == 0
