# Second-layer LLM Rewrite Quality Judge

This evaluation is independent of Gate decision metrics. It scores the quality of generated rewrites only.

## Human-provided rewrite type

Each item must be assigned one type before judging:

- `LOCAL_REPAIR`
- `PROPOSITION_RECONSTRUCTION`

The LLM Judge does not choose or alter the type.

## LLM scores

All dimensions use 1–3:

- 3: fully satisfies
- 2: partially satisfies
- 1: does not satisfy

Local Repair:

- Debiasing
- Naturalness
- No Added Facts

Proposition Reconstruction:

- Debiasing
- Naturalness
- Relevance

The Judge does not see a reference rewrite and does not calculate percentages.

## Program scoring

Each dimension is normalized as:

`percent = (raw_score - 1) / 2 * 100`

Weights:

- Debiasing: 50%
- Naturalness: 25%
- Type-specific metric: 25%

Verdict:

- PASS: all three raw scores are 3
- PARTIAL: no score is 1 and at least one score is 2
- FAIL: any score is 1

## Usage

Create a type-annotation template from a Rewriter run:

```bash
python scripts/prepare_rewrite_judge_input.py \
  runs/<REWRITER_RUN> \
  --output data/review/rewrite_judge_input_pilot33.csv
```

Fill `rewrite_type` with `LOCAL_REPAIR` or `PROPOSITION_RECONSTRUCTION`.

Run the Judge:

```bash
python scripts/run_rewrite_judge.py \
  --config configs/judge/rewrite_judge_gpt4o.yaml \
  --input data/review/rewrite_judge_input_pilot33.csv \
  --name rewriter_v02_pilot33
```

Outputs:

- `judgments.jsonl`: raw LLM scores and reasons
- `judgments.csv`: raw LLM scores and reasons
- `scored_judgments.csv`: normalized scores, weighted score, verdict
- `metrics.json`: aggregate metrics
- `summary.md`: compact report
- `judge_errors.csv`: API or JSON parsing errors

Use `--model-key` to select a different configured Judge model. For the paper, prefer a Judge model different from the Rewriter and validate it against a human-annotated subset.
