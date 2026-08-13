from pathlib import Path

from gender_gate.rewrite_type_router import RewriteTypeRouter, parse_rewrite_type_output
from gender_gate.schema import DatasetItem


class FakeClient:
    provider = "fake"
    model = "fake-router"
    temperature = 0
    max_output_tokens = 32

    def __init__(self, output: str):
        self.output = output

    def complete(self, messages):
        assert messages[0]["role"] == "user"
        assert "LOCAL_REPAIR" in messages[0]["content"]
        return self.output


def test_parse_bare_labels():
    assert parse_rewrite_type_output("LOCAL_REPAIR") == "LOCAL_REPAIR"
    assert (
        parse_rewrite_type_output("PROPOSITION_RECONSTRUCTION")
        == "PROPOSITION_RECONSTRUCTION"
    )


def test_parse_json_and_small_extra_text():
    assert parse_rewrite_type_output('{"rewrite_type":"LOCAL_REPAIR"}') == "LOCAL_REPAIR"
    assert (
        parse_rewrite_type_output("Final: PROPOSITION_RECONSTRUCTION")
        == "PROPOSITION_RECONSTRUCTION"
    )


def test_router_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config = {
        "prompt": "prompts/rewrite_type_router_v01.txt",
        "cache_db": str(tmp_path / "router.sqlite"),
    }
    item = DatasetItem(
        id="POS-X",
        text="她比男人还细心。",
        label="POSITIVE",
        meta={},
    )
    router = RewriteTypeRouter(
        {},
        config,
        root,
        client=FakeClient("LOCAL_REPAIR"),
    )
    prediction = router.predict(item)
    assert prediction.predicted_type == "LOCAL_REPAIR"
    assert prediction.error is None
