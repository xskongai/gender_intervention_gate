import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "apply_rewrite_type_map.py"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_apply_rewrite_type_map(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    map_path = tmp_path / "map.csv"
    output_path = tmp_path / "output.csv"

    write_csv(
        input_path,
        ["id", "text", "output", "rewrite_type", "type_note"],
        [
            {"id": "A", "text": "甲", "output": "甲改", "rewrite_type": "", "type_note": ""},
            {"id": "B", "text": "乙", "output": "乙改", "rewrite_type": "", "type_note": ""},
        ],
    )
    write_csv(
        map_path,
        ["id", "text", "rewrite_type", "type_note"],
        [
            {"id": "A", "text": "甲", "rewrite_type": "LOCAL_REPAIR", "type_note": "local"},
            {"id": "B", "text": "乙", "rewrite_type": "PROPOSITION_RECONSTRUCTION", "type_note": "recon"},
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_path),
            "--type-map",
            str(map_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["rewrite_type"] == "LOCAL_REPAIR"
    assert rows[1]["rewrite_type"] == "PROPOSITION_RECONSTRUCTION"
    assert rows[1]["type_note"] == "recon"
