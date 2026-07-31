# Rewriter v04: Branch-Aware Minimal Rewriting

## Goal

Rewriter v04 separates two internal branches:

1. **Local repair** when removing or replacing the gender-linked fragment fully removes the gender bias while preserving independently valid praise, criticism, questions, and descriptions.
2. **Minimal proposition reconstruction** only when the residual proposition still carries a gender-role obligation, identity-based humiliation, restriction, or a generalized version of the same harmful norm.

The central rule is:

> Complete gender debiasing first; among fully debiased candidates, preserve the maximum amount of non-gender content and discourse function.

## Important calibration

A negative statement is not automatically a gender-biased statement. For example, after removing a gender restriction, an ordinary criticism of loudness, lateness, or a concrete action may remain if it no longer relies on gender to be meaningful.

Conversely, a role expectation such as compulsory breadwinning, emotional suppression, compulsory marriage, or gendered family sacrifice cannot be repaired by simply replacing the gender group with “people”, “adults”, “parents”, or another broader group.

## Run

```bash
python scripts/run_rewriter_experiment.py \
  --config configs/rewriter/rewriter_v04_gpt4o.yaml \
  --name rewriter_v04_dev219
```

Then prepare the judge input and apply the fixed Dev-219 type map:

```bash
RUN=$(ls -dt runs/*rewriter_rewriter_v04_dev219 | head -1)

python scripts/prepare_rewrite_judge_input.py \
  "$RUN" \
  --output data/review/rewrite_judge_input_v04_dev219.csv

python scripts/apply_rewrite_type_map.py \
  --input data/review/rewrite_judge_input_v04_dev219.csv \
  --type-map data/review/rewrite_type_map_dev219.csv \
  --output data/review/rewrite_judge_input_v04_dev219_filled.csv
```

## Evaluation warning

Rewrite Judge v02 may over-penalize correct local repairs when it treats any remaining criticism or negative evaluation as residual gender bias. Therefore, compare v04 with prior systems using the same frozen judge for continuity, but audit low-scoring local-repair cases before interpreting the absolute score.
