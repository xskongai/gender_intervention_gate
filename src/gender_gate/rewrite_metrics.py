from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from .metrics import safe_div
from .rewrite import normalize_text


def calculate_rewrite_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [p for p in predictions if p["gold"] == "POSITIVE"]
    negatives = [p for p in predictions if p["gold"] == "NEGATIVE"]
    positive_changed = sum(bool(p.get("changed")) for p in positives)
    negative_changed = sum(bool(p.get("changed")) for p in negatives)
    rewrite_calls = sum(bool(p.get("rewrite_called")) for p in predictions)
    errors = sum(bool(p.get("error")) for p in predictions)
    cache_hits = sum(bool(p.get("cache_hit")) for p in predictions)

    exact_reference_matches = 0
    reference_count = 0
    for prediction in positives:
        reference = prediction.get("reference_output")
        if reference is None:
            continue
        reference_count += 1
        exact_reference_matches += (
            normalize_text(str(prediction.get("final_output") or ""))
            == normalize_text(str(reference))
        )

    return {
        "count": len(predictions),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "rewrite_calls": rewrite_calls,
        "rewrite_call_rate": safe_div(rewrite_calls, len(predictions)),
        "rewrite_calls_saved": len(predictions) - rewrite_calls,
        "cache_hits": cache_hits,
        "error_count": errors,
        "error_rate": safe_div(errors, len(predictions)),
        "positive_intervention_rate": safe_div(positive_changed, len(positives)),
        "under_edit_rate": 1.0 - safe_div(positive_changed, len(positives)),
        "negative_preservation": 1.0 - safe_div(negative_changed, len(negatives)),
        "over_edit_rate": safe_div(negative_changed, len(negatives)),
        "overall_change_rate": safe_div(positive_changed + negative_changed, len(predictions)),
        "reference_count": reference_count,
        "exact_reference_match_rate": safe_div(exact_reference_matches, reference_count),
        "counts": {
            "positive_changed": positive_changed,
            "positive_unchanged": len(positives) - positive_changed,
            "negative_preserved": len(negatives) - negative_changed,
            "negative_changed": negative_changed,
        },
    }


def semantic_review_reasons(prediction: dict[str, Any]) -> list[str]:
    """Return conservative review flags, not claims of semantic violation."""
    if not prediction.get("rewrite_called") or prediction.get("error"):
        return []

    source = normalize_text(str(prediction.get("text") or ""))
    output = normalize_text(str(prediction.get("final_output") or ""))
    if not source or not output or source == output:
        return []

    reasons: list[str] = []
    similarity = SequenceMatcher(None, source, output).ratio()
    length_ratio = len(output) / max(len(source), 1)

    if similarity < 0.25:
        reasons.append("LOW_TEXT_SIMILARITY")
    if length_ratio > 2.2:
        reasons.append("OUTPUT_MUCH_LONGER")
    if length_ratio < 0.35:
        reasons.append("OUTPUT_MUCH_SHORTER")
    if "原句" in output or "改写" in output or "性别包容" in output:
        reasons.append("POSSIBLE_EXPLANATORY_OUTPUT")
    if "\n" in output:
        reasons.append("MULTILINE_OUTPUT")
    return reasons
