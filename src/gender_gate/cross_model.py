from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    name: str
    gate_split: str
    rewriter_split: str
    gate_count: int
    positive_count: int


STAGES: dict[str, StageSpec] = {
    "smoke20": StageSpec(
        name="smoke20",
        gate_split="data/splits/group_aware_v2.3/dev_smoke_20.jsonl",
        rewriter_split="data/splits/group_aware_v2.3/dev_smoke_positive_10.jsonl",
        gate_count=20,
        positive_count=10,
    ),
    "pilot60": StageSpec(
        name="pilot60",
        gate_split="data/splits/group_aware_v2.3/dev_pilot_60.jsonl",
        rewriter_split="data/splits/group_aware_v2.3/dev_pilot_positive_33.jsonl",
        gate_count=60,
        positive_count=33,
    ),
    "dev400": StageSpec(
        name="dev400",
        gate_split="data/splits/group_aware_v2.3/dev.jsonl",
        rewriter_split="data/splits/group_aware_v2.3/dev_positive_219.jsonl",
        gate_count=400,
        positive_count=219,
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_judge_input(
    rewriter_run: Path,
    type_map_path: Path,
    output_path: Path,
) -> int:
    predictions_path = rewriter_run / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    predictions = load_jsonl(predictions_path)

    with type_map_path.open(encoding="utf-8-sig", newline="") as handle:
        type_rows = list(csv.DictReader(handle))
    rewrite_types = {
        str(row["id"]).strip(): str(row["rewrite_type"]).strip()
        for row in type_rows
        if str(row.get("id", "")).strip()
    }

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for prediction in predictions:
        item_id = str(prediction["id"])
        rewrite_type = rewrite_types.get(item_id, "")
        if not rewrite_type:
            missing.append(item_id)
            continue
        rows.append(
            {
                "id": item_id,
                "text": str(prediction["text"]),
                "output": str(prediction["final_output"]),
                "rewrite_type": rewrite_type,
                "type_note": "",
            }
        )

    if missing:
        raise ValueError(
            "Rewrite type map is missing ids: " + ", ".join(missing[:10])
        )
    if not rows:
        raise ValueError("No rewriter predictions were available for Judge input")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def summarize_suite(
    provider: str,
    stage: StageSpec,
    target_model: str,
    gate_run: Path,
    rewriter_run: Path,
    judge_run: Path | None,
) -> dict[str, Any]:
    gate_metrics = load_json(gate_run / "metrics.json")
    rewriter_metrics = load_json(rewriter_run / "metrics.json")
    gate_manifest = load_json(gate_run / "manifest.json")
    rewriter_manifest = load_json(rewriter_run / "manifest.json")

    result: dict[str, Any] = {
        "provider": provider,
        "stage": stage.name,
        "target_model": target_model,
        "judge_model": "gpt-4o",
        "gate_run": str(gate_run),
        "rewriter_run": str(rewriter_run),
        "judge_run": str(judge_run) if judge_run else None,
        "gate": {
            "count": gate_metrics["count"],
            "positive_recall": gate_metrics["positive_recall"],
            "negative_preservation": gate_metrics["negative_recall"],
            "balanced_accuracy": gate_metrics["balanced_accuracy"],
            "accuracy": gate_metrics["accuracy"],
            "rule_coverage": gate_metrics.get("routing", {}).get("rule_coverage", 0.0),
            "llm_call_rate": gate_metrics.get("routing", {}).get("llm_call_rate", 1.0),
            "format_error_rate": gate_metrics.get("format_error_rate", 0.0),
            "model": gate_manifest.get("model"),
        },
        "rewriter": {
            "count": rewriter_metrics["count"],
            "intervention_rate": rewriter_metrics["intervention_rate"],
            "under_edit_rate": rewriter_metrics["under_edit_rate"],
            "error_rate": rewriter_metrics["error_rate"],
            "model": rewriter_manifest.get("model"),
        },
    }

    if judge_run:
        judge_metrics = load_json(judge_run / "metrics.json")
        judge_manifest = load_json(judge_run / "manifest.json")
        result["judge"] = {
            "count": judge_metrics["overall"]["count"],
            "overall_quality": judge_metrics["overall"]["quality_score"],
            "macro_quality": judge_metrics["macro_quality_score"],
            "debiasing": judge_metrics["overall"]["debiasing_score"],
            "naturalness": judge_metrics["overall"]["naturalness_score"],
            "type_specific": judge_metrics["overall"]["type_specific_score"],
            "pass_rate": judge_metrics["overall"]["pass_rate"],
            "error_count": judge_metrics["overall"]["error_count"],
            "model": judge_manifest.get("model"),
        }
    return result


def summary_markdown(result: dict[str, Any]) -> str:
    gate = result["gate"]
    rewriter = result["rewriter"]
    lines = [
        f"# Cross-model run: {result['provider']} / {result['stage']}",
        "",
        f"- Target model: `{result['target_model']}`",
        "- Judge: `gpt-4o` (frozen v04 Balanced)",
        "- Gate: Rule-first + Frozen LLM Gate",
        "",
        "## Gate",
        "",
        f"- Count: {gate['count']}",
        f"- Positive Recall: {gate['positive_recall']:.4f}",
        f"- Negative Preservation: {gate['negative_preservation']:.4f}",
        f"- Balanced Accuracy: {gate['balanced_accuracy']:.4f}",
        f"- Rule Coverage: {gate['rule_coverage']:.4f}",
        f"- LLM Call Rate: {gate['llm_call_rate']:.4f}",
        f"- Format Error Rate: {gate['format_error_rate']:.4f}",
        "",
        "## Rewriter",
        "",
        f"- Count: {rewriter['count']}",
        f"- Intervention Rate: {rewriter['intervention_rate']:.4f}",
        f"- Under-edit Rate: {rewriter['under_edit_rate']:.4f}",
        f"- Error Rate: {rewriter['error_rate']:.4f}",
    ]
    if "judge" in result:
        judge = result["judge"]
        lines.extend(
            [
                "",
                "## GPT-4o Judge",
                "",
                f"- Overall Quality: {judge['overall_quality']:.4f}",
                f"- Macro Quality: {judge['macro_quality']:.4f}",
                f"- Debiasing: {judge['debiasing']:.4f}",
                f"- Naturalness: {judge['naturalness']:.4f}",
                f"- Type-specific: {judge['type_specific']:.4f}",
                f"- Pass Rate: {judge['pass_rate']:.4f}",
                f"- Judge Errors: {judge['error_count']}",
            ]
        )
    return "\n".join(lines) + "\n"
