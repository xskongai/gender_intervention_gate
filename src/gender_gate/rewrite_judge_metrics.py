from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .metrics import safe_div
from .rewrite_judge import normalize_rewrite_type


def normalize_dimension_score(score: int) -> float:
    if score not in {1, 2, 3}:
        raise ValueError("Dimension score must be 1, 2, or 3")
    return (score - 1) / 2 * 100.0


def score_judgment(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("error"):
        return {
            **row,
            "debiasing_percent": None,
            "naturalness_percent": None,
            "type_specific_percent": None,
            "quality_score": None,
            "verdict": "ERROR",
        }

    rewrite_type = normalize_rewrite_type(str(row["rewrite_type"]))
    d = int(row["debiasing_score"])
    n = int(row["naturalness_score"])
    if rewrite_type == "LOCAL_REPAIR":
        if row.get("fidelity_score") is not None:
            special = int(row["fidelity_score"])
            special_name = "fidelity"
        elif row.get("no_added_facts_score") is not None:
            # Backward compatibility for v01 Judge outputs.
            special = int(row["no_added_facts_score"])
            special_name = "no_added_facts"
        else:
            raise ValueError(
                "LOCAL_REPAIR judgment requires fidelity_score "
                "(or legacy no_added_facts_score)"
            )
    else:
        special = int(row["relevance_score"])
        special_name = "relevance"

    d_pct = normalize_dimension_score(d)
    n_pct = normalize_dimension_score(n)
    special_pct = normalize_dimension_score(special)
    quality = 0.50 * d_pct + 0.25 * n_pct + 0.25 * special_pct

    scores = [d, n, special]
    if all(score == 3 for score in scores):
        verdict = "PASS"
    elif any(score == 1 for score in scores):
        verdict = "FAIL"
    else:
        verdict = "PARTIAL"

    return {
        **row,
        "type_specific_metric": special_name,
        "debiasing_percent": d_pct,
        "naturalness_percent": n_pct,
        "type_specific_percent": special_pct,
        "quality_score": quality,
        "verdict": verdict,
    }


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("quality_score") is not None]
    return {
        "count": len(rows),
        "valid_count": len(valid),
        "error_count": len(rows) - len(valid),
        "quality_score": _mean_or_none(
            [float(row["quality_score"]) for row in valid]
        ),
        "debiasing_score": _mean_or_none(
            [float(row["debiasing_percent"]) for row in valid]
        ),
        "naturalness_score": _mean_or_none(
            [float(row["naturalness_percent"]) for row in valid]
        ),
        "type_specific_score": _mean_or_none(
            [float(row["type_specific_percent"]) for row in valid]
        ),
        "pass_count": sum(row.get("verdict") == "PASS" for row in valid),
        "partial_count": sum(row.get("verdict") == "PARTIAL" for row in valid),
        "fail_count": sum(row.get("verdict") == "FAIL" for row in valid),
        "pass_rate": safe_div(
            sum(row.get("verdict") == "PASS" for row in valid), len(valid)
        ),
        "partial_rate": safe_div(
            sum(row.get("verdict") == "PARTIAL" for row in valid), len(valid)
        ),
        "fail_rate": safe_div(
            sum(row.get("verdict") == "FAIL" for row in valid), len(valid)
        ),
    }


def calculate_judge_metrics(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    local = [
        row
        for row in scored_rows
        if normalize_rewrite_type(str(row["rewrite_type"])) == "LOCAL_REPAIR"
    ]
    reconstruction = [
        row
        for row in scored_rows
        if normalize_rewrite_type(str(row["rewrite_type"]))
        == "PROPOSITION_RECONSTRUCTION"
    ]
    overall = _subset_metrics(scored_rows)
    local_metrics = _subset_metrics(local)
    reconstruction_metrics = _subset_metrics(reconstruction)
    type_quality = [
        value
        for value in [
            local_metrics["quality_score"],
            reconstruction_metrics["quality_score"],
        ]
        if value is not None
    ]
    local_type_specific_metric = next(
        (
            str(row["type_specific_metric"])
            for row in local
            if row.get("type_specific_metric")
        ),
        "fidelity",
    )
    return {
        "overall": overall,
        "local_repair": local_metrics,
        "proposition_reconstruction": reconstruction_metrics,
        "macro_quality_score": _mean_or_none([float(v) for v in type_quality]),
        "local_type_specific_metric": local_type_specific_metric,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _metric_label(metric_name: str) -> str:
    return {
        "fidelity": "Fidelity",
        "no_added_facts": "No Added Facts",
        "relevance": "Relevance",
    }.get(metric_name, metric_name)


def generate_judge_reports(
    run_dir: Path, judgments: list[dict[str, Any]]
) -> dict[str, Any]:
    scored = [score_judgment(row) for row in judgments]
    metrics = calculate_judge_metrics(scored)
    _write_csv(run_dir / "judgments.csv", judgments)
    _write_csv(run_dir / "scored_judgments.csv", scored)
    _write_csv(
        run_dir / "judge_errors.csv", [row for row in scored if row.get("error")]
    )
    with (run_dir / "judgments.jsonl").open("w", encoding="utf-8") as handle:
        for row in judgments:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    local = metrics["local_repair"]
    recon = metrics["proposition_reconstruction"]
    overall = metrics["overall"]
    local_metric = _metric_label(metrics["local_type_specific_metric"])
    summary = f"""# Rewrite Quality Judge

LLM Judge assigns only 1–3 dimension scores. Percentage normalization, weighted quality scores, verdicts, and aggregate metrics are computed by the program.

Scoring weights: Debiasing 50%, Naturalness 25%, type-specific metric 25%.

| Type | Count | Quality /100 | Debiasing | Naturalness | Type-specific | PASS | PARTIAL | FAIL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Local Repair | {local['count']} | {_pct(local['quality_score'])} | {_pct(local['debiasing_score'])} | {_pct(local['naturalness_score'])} | {_pct(local['type_specific_score'])} | {local['pass_rate'] * 100:.2f}% | {local['partial_rate'] * 100:.2f}% | {local['fail_rate'] * 100:.2f}% |
| Proposition Reconstruction | {recon['count']} | {_pct(recon['quality_score'])} | {_pct(recon['debiasing_score'])} | {_pct(recon['naturalness_score'])} | {_pct(recon['type_specific_score'])} | {recon['pass_rate'] * 100:.2f}% | {recon['partial_rate'] * 100:.2f}% | {recon['fail_rate'] * 100:.2f}% |
| Overall (micro) | {overall['count']} | {_pct(overall['quality_score'])} | {_pct(overall['debiasing_score'])} | {_pct(overall['naturalness_score'])} | {_pct(overall['type_specific_score'])} | {overall['pass_rate'] * 100:.2f}% | {overall['partial_rate'] * 100:.2f}% | {overall['fail_rate'] * 100:.2f}% |

- Macro quality score: {_pct(metrics['macro_quality_score'])}/100
- Local type-specific metric: {local_metric}
- Reconstruction type-specific metric: Relevance
- Judge errors: {overall['error_count']}/{overall['count']}
"""
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return metrics
