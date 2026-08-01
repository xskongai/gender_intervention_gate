# Adaptive Verify–Repair Rewriter v01

This implementation turns the one-pass POSITIVE rewriter into a configurable
feedback-guided trajectory:

1. Generate an initial rewrite with the frozen v02 prompt.
2. Score it with the frozen v04 multidimensional verifier.
3. Stop when the quality threshold is met.
4. Otherwise route the dominant failure to a targeted repair prompt:
   - `debiasing`
   - `fidelity` for local repair
   - `relevance` for proposition reconstruction
   - `naturalness`
   - `generic` for generation/verifier failures
5. Repeat for at most `max_rounds` total candidates.
6. Return the highest-ranked valid candidate from the whole trajectory, not
   mechanically the last candidate.

## Important score granularity

The current frozen v04 Judge assigns 1–3 per dimension and the program computes:

- Debiasing: 50%
- Naturalness: 25%
- Fidelity/Relevance: 25%

Therefore the quality score is discrete. With `threshold: 80`, accepted outputs
normally have 87.5 or 100. This is intentional for the first validation. A
future 1–5 or 0–100 verifier can provide a finer threshold sweep.

## Fast paired validation

GLM is a useful first target because its one-pass Pilot33 quality was moderate,
so the adaptive loop has room to rescue failures:

```bash
python -u scripts/run_adaptive_rewriter.py \
  --model-key glm4_9b_ollama \
  --split data/splits/group_aware_v2.3/dev_pilot_positive_33.jsonl \
  --threshold 80 \
  --max-rounds 3 \
  --concurrency 1 \
  --max-output-tokens 256 \
  --name glm_adaptive_pilot33
```

Mistral stress test:

```bash
python -u scripts/run_adaptive_rewriter.py \
  --model-key mistral_7b_ollama \
  --split data/splits/group_aware_v2.3/dev_pilot_positive_33.jsonl \
  --threshold 80 \
  --max-rounds 3 \
  --concurrency 1 \
  --max-output-tokens 256 \
  --name mistral_adaptive_pilot33
```

DeepSeek-R1:

```bash
python -u scripts/run_adaptive_rewriter.py \
  --model-key deepseek_r1_8b_ollama \
  --split data/splits/group_aware_v2.3/dev_pilot_positive_33.jsonl \
  --threshold 80 \
  --max-rounds 3 \
  --concurrency 1 \
  --max-output-tokens 2048 \
  --name deepseek_adaptive_pilot33
```

The initial candidate and initial verifier result reuse the existing cache when
model, prompt, token budget, and output match previous v02 runs. Do not clear the
cache before the first validation.

## Output files

Each run writes:

- `trajectories.jsonl`: all candidates, scores, reasons, routes, and selection.
- `candidates.csv`: flat round-level audit table.
- `predictions.jsonl`: selected final outputs in the existing rewriter format.
- `predictions.csv`: selected final outputs.
- `metrics.json`: paired initial-vs-final quality and cost metrics.
- `summary.md`: readable summary.
- copied initial, repair, and verifier prompts.

Key metrics:

- Initial quality vs final quality
- Mean quality gain
- Initial pass rate vs final pass rate
- Refinement trigger rate
- Rescue@2 and Rescue@3
- Average rounds
- Trajectory regression rate
- Selected-not-last rate
- Generation/verifier calls and latency

## Final evaluation warning

The online verifier directly guides repairs. It must not be the only final
paper evaluator. Use a different model or blinded human evaluation on the final
`predictions.jsonl` to avoid optimizing and evaluating against the same judge.
