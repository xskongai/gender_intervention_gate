from __future__ import annotations

from gender_gate.adaptive_rewriter import (
    AdaptiveCandidate,
    AdaptivePolicy,
    candidate_passes,
    choose_repair_route,
    select_best_candidate,
)


def make_candidate(
    *,
    round_id: int = 1,
    quality: float | None = 75.0,
    debiasing: int | None = 2,
    naturalness: int | None = 3,
    type_score: int | None = 3,
    type_name: str = "fidelity",
    edit_distance: int = 2,
    verifier_error: str | None = None,
) -> AdaptiveCandidate:
    return AdaptiveCandidate(
        round_id=round_id,
        route="initial",
        raw_output="raw",
        output=f"candidate-{round_id}",
        generator_prompt_version="v02",
        generator_latency_ms=10,
        generator_cache_hit=False,
        generator_error=None,
        verifier_raw_output="{}",
        verifier_prompt_version="judge",
        verifier_latency_ms=10,
        verifier_cache_hit=False,
        verifier_error=verifier_error,
        debiasing_score=debiasing,
        debiasing_reason="d",
        naturalness_score=naturalness,
        naturalness_reason="n",
        type_specific_name=type_name,
        type_specific_score=type_score,
        type_specific_reason="t",
        quality_score=quality,
        verdict="PARTIAL",
        accepted=False,
        edit_distance=edit_distance,
    )


def test_threshold_acceptance_is_configurable() -> None:
    candidate = make_candidate(quality=87.5, debiasing=3, naturalness=2, type_score=3)
    loose = AdaptivePolicy(threshold=80, max_rounds=3)
    strict = AdaptivePolicy(
        threshold=90,
        max_rounds=3,
        debiasing_min_score=3,
        naturalness_min_score=2,
        type_specific_min_score=2,
    )
    assert candidate_passes(candidate, loose)
    assert not candidate_passes(candidate, strict)


def test_route_prioritizes_weighted_debiasing_deficit() -> None:
    candidate = make_candidate(
        quality=50,
        debiasing=1,
        naturalness=1,
        type_score=2,
    )
    assert choose_repair_route(candidate) == "debiasing"


def test_route_uses_fidelity_or_relevance() -> None:
    local = make_candidate(
        quality=75,
        debiasing=3,
        naturalness=3,
        type_score=1,
        type_name="fidelity",
    )
    reconstruction = make_candidate(
        quality=75,
        debiasing=3,
        naturalness=3,
        type_score=1,
        type_name="relevance",
    )
    assert choose_repair_route(local) == "fidelity"
    assert choose_repair_route(reconstruction) == "relevance"


def test_best_of_trajectory_does_not_force_last_round() -> None:
    candidates = [
        make_candidate(round_id=1, quality=75, debiasing=2),
        make_candidate(round_id=2, quality=100, debiasing=3),
        make_candidate(round_id=3, quality=87.5, debiasing=3),
    ]
    assert select_best_candidate(candidates).round_id == 2


def test_tie_prefers_smaller_edit_after_dimension_ties() -> None:
    larger = make_candidate(
        round_id=1,
        quality=87.5,
        debiasing=3,
        naturalness=3,
        type_score=2,
        edit_distance=8,
    )
    smaller = make_candidate(
        round_id=2,
        quality=87.5,
        debiasing=3,
        naturalness=3,
        type_score=2,
        edit_distance=3,
    )
    assert select_best_candidate([larger, smaller]).round_id == 2

import json

from gender_gate.adaptive_rewriter import AdaptiveRewriterRunner
from gender_gate.schema import DatasetItem


class StubInitialRewriter:
    model = "stub-rewriter"
    prompt_version = "initial"

    def rewrite(self, item: DatasetItem):
        return "第一版", 1, False, None


class StubRepairRewriter:
    model = "stub-rewriter"

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        round_id = kwargs["round_id"]
        route = kwargs["route"]
        self.calls.append((round_id, route))
        output = "第二版" if round_id == 2 else "第三版"
        return output, f"repair_{route}", 1, False, None


class StubVerifier:
    model = "stub-verifier"
    prompt_version = "judge"

    def judge(self, *, item_id, rewrite_type, text, output):
        if output == "第一版":
            scores = (2, 3, 3)
        elif output == "第二版":
            scores = (3, 2, 3)
        else:
            scores = (3, 3, 3)
        d, n, t = scores
        payload = {
            "debiasing": {"score": d, "reason": "d"},
            "naturalness": {"score": n, "reason": "n"},
            "fidelity": {"score": t, "reason": "f"},
            "relevance": None,
        }
        return json.dumps(payload, ensure_ascii=False), 1, False, None


def test_runner_uses_feedback_then_stops_on_threshold() -> None:
    repair = StubRepairRewriter()
    runner = AdaptiveRewriterRunner(
        initial_rewriter=StubInitialRewriter(),
        repair_rewriter=repair,
        verifier=StubVerifier(),
        rewrite_types={"POS-1": "LOCAL_REPAIR"},
        policy=AdaptivePolicy(threshold=80, max_rounds=3),
    )
    item = DatasetItem(id="POS-1", text="原句", label="POSITIVE", meta={})
    trajectory = runner.run_item(item)
    assert len(trajectory.candidates) == 2
    assert repair.calls == [(2, "debiasing")]
    assert trajectory.selected_round == 2
    assert trajectory.final_output == "第二版"
    assert trajectory.first_pass_round == 2
