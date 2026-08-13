#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import finalize_full_rewriter_results as finalizer


ROOT = Path.cwd()

WORKSPACES = {
    "Manual Type": ROOT / "refinement_workspace/dev219/qwen_manual_type/models/qwen_api",
    "Auto Type": ROOT / "refinement_workspace/dev219/qwen_auto_type/models/qwen_api",
}

OUTPUT_DIR = ROOT / "refinement_workspace/dev219/type_ab_summary"


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def get_summary(label: str, model_dir: Path) -> dict:
    _, s = finalizer.finalize_model("qwen_api", model_dir)

    # finalizer currently leaves v2_quality=None in this Dev219 setup,
    # but quality_gain = final_quality - initial_quality.
    v2_quality = s.get("v2_quality")
    if v2_quality is None:
        v2_quality = float(s["final_quality"]) - float(s["quality_gain"])

    n = int(s["v2_success_n"])

    return {
        "label": label,
        "n": n,
        "v2_quality": float(v2_quality),
        "accepted_r1": int(s["accepted_round1_n"]),
        "triggered_r2": int(s["round2_triggered_n"]),
        "accepted_r2": int(s["accepted_round2_n"]),
        "queue_r3": int(s["round3_queue_n"]),
        "accepted_r3": int(s["accepted_round3_candidate_n"]),
        "final_accepted": int(s["final_accepted_n"]),
        "fallback": int(s["highest_score_fallback_n"]),
        "final_quality": float(s["final_quality"]),
        "quality_gain": float(s["quality_gain"]),
    }


def main():
    results = [
        get_summary(label, path)
        for label, path in WORKSPACES.items()
    ]

    manual = results[0]
    auto = results[1]

    rows = [
        (
            "Round-1 Quality",
            manual["v2_quality"],
            auto["v2_quality"],
            auto["v2_quality"] - manual["v2_quality"],
            "score",
        ),
        (
            "Round-1 Accepted",
            manual["accepted_r1"],
            auto["accepted_r1"],
            auto["accepted_r1"] - manual["accepted_r1"],
            "count",
        ),
        (
            "Round-2 Triggered",
            manual["triggered_r2"],
            auto["triggered_r2"],
            auto["triggered_r2"] - manual["triggered_r2"],
            "count",
        ),
        (
            "Round-2 Accepted",
            manual["accepted_r2"],
            auto["accepted_r2"],
            auto["accepted_r2"] - manual["accepted_r2"],
            "count",
        ),
        (
            "Round-3 Queue",
            manual["queue_r3"],
            auto["queue_r3"],
            auto["queue_r3"] - manual["queue_r3"],
            "count",
        ),
        (
            "Round-3 Accepted",
            manual["accepted_r3"],
            auto["accepted_r3"],
            auto["accepted_r3"] - manual["accepted_r3"],
            "count",
        ),
        (
            "Final Accepted (%)",
            pct(manual["final_accepted"], manual["n"]),
            pct(auto["final_accepted"], auto["n"]),
            pct(auto["final_accepted"], auto["n"])
            - pct(manual["final_accepted"], manual["n"]),
            "score",
        ),
        (
            "Fallback",
            manual["fallback"],
            auto["fallback"],
            auto["fallback"] - manual["fallback"],
            "count",
        ),
        (
            "Final Quality",
            manual["final_quality"],
            auto["final_quality"],
            auto["final_quality"] - manual["final_quality"],
            "score",
        ),
        (
            "Refinement Gain",
            manual["quality_gain"],
            auto["quality_gain"],
            auto["quality_gain"] - manual["quality_gain"],
            "score",
        ),
    ]

    # ============================================================
    # Terminal output
    # ============================================================

    output_lines = []

    output_lines.append("=" * 88)
    output_lines.append("DEV219 — MANUAL TYPE vs AUTO TYPE")
    output_lines.append("=" * 88)

    output_lines.append(
        f"{'Metric':<28}"
        f"{'Manual Type':>18}"
        f"{'Auto Type':>18}"
        f"{'Auto - Manual':>18}"
    )

    output_lines.append("-" * 88)

    for name, m, a, delta, kind in rows:
        if kind == "count":
            line = f"{name:<28}{m:>18d}{a:>18d}{delta:>+18d}"
        else:
            line = f"{name:<28}{m:>18.2f}{a:>18.2f}{delta:>+18.2f}"

        output_lines.append(line)

    output_lines.append("-" * 88)

    manual_accept_pct = pct(
        manual["final_accepted"],
        manual["n"],
    )

    auto_accept_pct = pct(
        auto["final_accepted"],
        auto["n"],
    )

    output_lines.append(
        f"Manual final accepted: "
        f"{manual['final_accepted']}/{manual['n']} "
        f"({manual_accept_pct:.2f}%)"
    )

    output_lines.append(
        f"Auto final accepted:   "
        f"{auto['final_accepted']}/{auto['n']} "
        f"({auto_accept_pct:.2f}%)"
    )

    output_lines.append("")
    output_lines.append("Summary:")

    output_lines.append(
        f"- Round-1 quality change: "
        f"{auto['v2_quality'] - manual['v2_quality']:+.2f}"
    )

    output_lines.append(
        f"- Final quality change:   "
        f"{auto['final_quality'] - manual['final_quality']:+.2f}"
    )

    output_lines.append(
        f"- Final acceptance change: "
        f"{auto_accept_pct - manual_accept_pct:+.2f} pp"
    )

    terminal_text = "\n".join(output_lines)

    print(terminal_text)

    # ============================================================
    # Save output directory
    # ============================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Save TXT
    # Exactly the same summary as terminal output
    # ============================================================

    txt_path = OUTPUT_DIR / "manual_vs_auto_summary.txt"

    txt_path.write_text(
        terminal_text + "\n",
        encoding="utf-8",
    )

    # ============================================================
    # Save CSV
    # Same displayed precision as terminal
    # ============================================================

    csv_path = OUTPUT_DIR / "manual_vs_auto_summary.csv"

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Metric",
            "Manual Type",
            "Auto Type",
            "Auto - Manual",
        ])

        for name, m, a, delta, kind in rows:
            if kind == "count":
                writer.writerow([
                    name,
                    f"{m:d}",
                    f"{a:d}",
                    f"{delta:+d}",
                ])
            else:
                writer.writerow([
                    name,
                    f"{m:.2f}",
                    f"{a:.2f}",
                    f"{delta:+.2f}",
                ])

    # ============================================================
    # Save JSON
    # Keep full numerical precision
    # ============================================================

    json_path = OUTPUT_DIR / "manual_vs_auto_summary.json"

    json_data = {
        "experiment": "DEV219 Manual Type vs Auto Type",
        "model": "qwen_api",

        "manual_type": {
            **manual,
            "final_accepted_pct": manual_accept_pct,
        },

        "auto_type": {
            **auto,
            "final_accepted_pct": auto_accept_pct,
        },

        "delta_auto_minus_manual": {
            "round1_quality":
                auto["v2_quality"]
                - manual["v2_quality"],

            "round1_accepted":
                auto["accepted_r1"]
                - manual["accepted_r1"],

            "round2_triggered":
                auto["triggered_r2"]
                - manual["triggered_r2"],

            "round2_accepted":
                auto["accepted_r2"]
                - manual["accepted_r2"],

            "round3_queue":
                auto["queue_r3"]
                - manual["queue_r3"],

            "round3_accepted":
                auto["accepted_r3"]
                - manual["accepted_r3"],

            "final_accepted":
                auto["final_accepted"]
                - manual["final_accepted"],

            "final_accepted_pp":
                auto_accept_pct
                - manual_accept_pct,

            "fallback":
                auto["fallback"]
                - manual["fallback"],

            "final_quality":
                auto["final_quality"]
                - manual["final_quality"],

            "refinement_gain":
                auto["quality_gain"]
                - manual["quality_gain"],
        },
    }

    json_path.write_text(
        json.dumps(
            json_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # Saved paths
    # ============================================================

    print()
    print("Saved results:")
    print(f"TXT : {txt_path}")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()