# Rewrite Judge v03: Split-Dimension Evaluation

Judge v03 separates rewrite quality evaluation into three independent LLM calls:

1. Debiasing only
2. Naturalness only
3. Type-specific quality only
   - Fidelity for `LOCAL_REPAIR`
   - Relevance for `PROPOSITION_RECONSTRUCTION`

The program merges the three 1–3 scores and retains the existing scoring formula:

- Debiasing: 50%
- Naturalness: 25%
- Type-specific metric: 25%

This design prevents one prompt from conflating debiasing with fidelity or relevance.
`rewrite_type` remains externally fixed and is not selected by any judge.

## Run

```bash
python scripts/run_rewrite_judge.py \
  --config configs/judge/rewrite_judge_v03_split_gpt4o.yaml \
  --input data/review/rewrite_judge_input_v04_dev219_filled.csv \
  --name rewriter_v04_dev219_judge_v03_split
```

Each sample uses three LLM calls. For 219 samples, a fresh uncached run uses 657 calls.
The three raw model outputs are retained inside the `raw_output` audit field.
