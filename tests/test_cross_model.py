from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from gender_gate.clients import resolve_model_name
from gender_gate.cross_model import STAGES, build_judge_input
from gender_gate.deterministic_rules import deterministic_label


ROOT = Path(__file__).resolve().parents[1]


def test_model_resolution_precedence(monkeypatch) -> None:
    config = {"model_env": "TEST_PROVIDER_MODEL", "model": "fixed-default"}
    monkeypatch.setenv("TEST_PROVIDER_MODEL", "from-env")
    assert resolve_model_name(config, {}) == "from-env"
    assert resolve_model_name(config, {"model": "from-cli"}) == "from-cli"
    monkeypatch.delenv("TEST_PROVIDER_MODEL")
    assert resolve_model_name(config, {}) == "fixed-default"


def test_judge_is_fixed_to_gpt4o() -> None:
    judge_config = yaml.safe_load(
        (ROOT / "configs/judge/rewrite_judge_v04_balanced_gpt4o.yaml").read_text(
            encoding="utf-8"
        )
    )
    models = yaml.safe_load(
        (ROOT / "configs/models.yaml").read_text(encoding="utf-8")
    )["models"]
    assert judge_config["model_key"] == "openai_judge"
    assert resolve_model_name(models["openai_judge"], judge_config) == "gpt-4o"


def test_stage_sizes_and_smoke_balance() -> None:
    assert STAGES["smoke20"].gate_count == 20
    assert STAGES["pilot60"].gate_count == 60
    assert STAGES["dev400"].gate_count == 400

    path = ROOT / STAGES["smoke20"].gate_split
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    assert sum(row["label"] == "POSITIVE" for row in rows) == 10
    assert sum(row["label"] == "NEGATIVE" for row in rows) == 10
    rule_count = sum(deterministic_label(row["text"]) is not None for row in rows)
    assert 0 < rule_count < len(rows)


def test_build_judge_input_joins_frozen_type_map(tmp_path: Path) -> None:
    rewriter_run = tmp_path / "rewriter"
    rewriter_run.mkdir()
    predictions = [
        {
            "id": "POS-1182",
            "text": "男怕入错行，女怕嫁错郎。",
            "final_output": "每个人都怕选错职业和伴侣。",
        }
    ]
    (rewriter_run / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "judge.csv"
    count = build_judge_input(
        rewriter_run,
        ROOT / "data/review/rewrite_type_map_dev219.csv",
        output,
    )
    assert count == 1
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["rewrite_type"] == "PROPOSITION_RECONSTRUCTION"
