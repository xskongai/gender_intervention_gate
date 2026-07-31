# Second-layer LLM Rewrite Quality Judge v02

This evaluation is independent of Gate decision metrics. It scores generated rewrites only.

## Human-provided rewrite type

Each item must be assigned one type before judging:

- `LOCAL_REPAIR`
- `PROPOSITION_RECONSTRUCTION`

The LLM Judge does not choose or alter the type. The type must reflect what the original sentence requires, not what a candidate model happened to produce.

## LLM scores

All dimensions use 1–3:

- 3: fully satisfies
- 2: partially satisfies
- 1: does not satisfy

Local Repair:

- Debiasing
- Naturalness
- Fidelity

Fidelity includes the old “No Added Facts” requirement and additionally checks that the candidate preserves the original non-biased content and communicative function. It penalizes unsupported advice, evaluations, rules, causal claims, or other substantive changes.

Proposition Reconstruction:

- Debiasing
- Naturalness
- Relevance

The Judge does not see a reference rewrite and does not calculate percentages.

## v02 calibration rules

- Judge the underlying biased proposition, not only the presence or absence of gender words.
- Generalizing a harmful gender norm to everyone is not complete debiasing.
- Interpret idioms, proverbs, metaphors, and irony by their conventional meaning rather than literal wording.
- A direct rejection or reversal of the original shame/norm can receive full relevance.
- A broad positive statement that does not address the core biased relation should receive partial or low relevance.

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
  --output data/review/rewrite_judge_input.csv
```

Fill `rewrite_type` with `LOCAL_REPAIR` or `PROPOSITION_RECONSTRUCTION`.

Run Judge v02:

```bash
python scripts/run_rewrite_judge.py \
  --config configs/judge/rewrite_judge_v02_gpt4o.yaml \
  --input data/review/rewrite_judge_input.csv \
  --name rewrite_judge_v02
```

Outputs:

- `judgments.jsonl`: raw LLM scores and reasons
- `judgments.csv`: raw LLM scores and reasons
- `scored_judgments.csv`: normalized scores, weighted score, verdict
- `metrics.json`: aggregate metrics
- `summary.md`: compact report
- `judge_errors.csv`: API or JSON parsing errors

The legacy v01 prompt/config remains available for reproducibility. The parser and metrics code can still read its `no_added_facts` field, but new experiments should use v02 and `fidelity`.
