import pytest

from gender_gate.rewriter import MockReferenceRewriter, build_rewriter_prediction
from gender_gate.schema import DatasetItem


def positive_item() -> DatasetItem:
    return DatasetItem(
        id="POS-X",
        text="她跑步比男的还快。",
        label="POSITIVE",
        meta={"reference_output": "她跑步很快。"},
    )


def negative_item() -> DatasetItem:
    return DatasetItem(
        id="NEG-X",
        text="她是医生。",
        label="NEGATIVE",
        meta={"reference_output": "她是医生。"},
    )


def test_mock_reference_rewriter_uses_reference() -> None:
    prediction = build_rewriter_prediction(positive_item(), MockReferenceRewriter())
    assert prediction.final_output == "她跑步很快。"
    assert prediction.changed is True
    assert prediction.gold == "POSITIVE"


def test_independent_rewriter_rejects_negative() -> None:
    with pytest.raises(ValueError, match="POSITIVE items only"):
        build_rewriter_prediction(negative_item(), MockReferenceRewriter())
