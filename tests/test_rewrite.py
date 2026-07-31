from gender_gate.rewrite import (
    MockOracleRewriter,
    build_rewrite_prediction,
    build_skipped_prediction,
)
from gender_gate.schema import DatasetItem


def item(label: str = "POSITIVE") -> DatasetItem:
    return DatasetItem(
        id="X-1",
        text="女人都是路痴。",
        label=label,  # type: ignore[arg-type]
        meta={"reference_output": "有些人方向感较弱。"},
    )


def test_mock_rewriter_uses_reference_output() -> None:
    prediction = build_rewrite_prediction(
        item(), "direct", MockOracleRewriter()
    )
    assert prediction.rewrite_called is True
    assert prediction.final_output == "有些人方向感较弱。"
    assert prediction.changed is True


def test_gated_skip_preserves_original() -> None:
    prediction = build_skipped_prediction(
        item(),
        mode="gated",
        model="mock",
        prompt_version="test",
        gate_prediction={"predicted": "NEGATIVE", "raw_output": "NEGATIVE"},
    )
    assert prediction.rewrite_called is False
    assert prediction.final_output == prediction.text
    assert prediction.changed is False
