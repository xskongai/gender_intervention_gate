import json
import subprocess
import sys
from pathlib import Path


def test_compare_rejects_different_splits(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    gated = tmp_path / "gated"
    direct.mkdir()
    gated.mkdir()
    base = {
        "prompt_sha256": "p",
        "model": "m",
        "metrics": {
            "count": 1,
            "positive_intervention_rate": 1.0,
            "under_edit_rate": 0.0,
            "negative_preservation": 1.0,
            "over_edit_rate": 0.0,
            "rewrite_call_rate": 1.0,
            "error_rate": 0.0,
            "rewrite_calls": 1,
        },
    }
    (direct / "manifest.json").write_text(
        json.dumps({**base, "split_sha256": "a"}), encoding="utf-8"
    )
    (gated / "manifest.json").write_text(
        json.dumps({**base, "split_sha256": "b"}), encoding="utf-8"
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/compare_rewrite_runs.py"),
            str(direct),
            str(gated),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "different dataset splits" in result.stderr
