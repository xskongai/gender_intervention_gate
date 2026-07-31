# Rewrite Judge v04 Balanced

This version returns to one combined Judge call per item while keeping the three scoring dimensions conceptually separate.

Key changes:

- A score of 3 means reasonable, correct, and acceptable—not flawless.
- Minor wording preferences do not reduce the score.
- The same defect should not be penalized repeatedly across dimensions.
- Necessary removal or replacement of biased content is not a fidelity loss.
- Fidelity protects only non-biased content.
- Relevance does not require repeating gender wording.

Run the 33-item pilot:

```bash
python scripts/run_rewrite_judge.py \
  --config configs/judge/rewrite_judge_v04_balanced_gpt4o.yaml \
  --input data/review/rewrite_judge_input_v02_pilot33.csv \
  --name rewriter_v02_pilot33_judge_v04_balanced
```

This is one LLM call per item. Judge v02 and v03 Split remain available for comparison.
