# Update: Rewriter v03 — Minimal Sufficient Rewrite

## Objective

Rewriter v03 operationalizes the following constrained objective:

> Completely remove the core gender-biased proposition, then minimize changes to all other content.

This is not mechanical gender-term deletion. If the harmful norm remains after removing a gender marker, the proposition must be minimally reconstructed.

## What changed from v02

1. Explicit priority ordering: complete debiasing first, textual minimality second.
2. Silent choice between local repair and minimal proposition reconstruction.
3. Mandatory residual-norm check after rewriting.
4. Stronger preservation of topic, lexical anchors, syntax, tone, and speech act.
5. Explicit prohibition against adding advice, praise, health/safety framing, or generic moral slogans.
6. Explicit prohibition against universalizing a gendered harmful norm.
7. Idioms, metaphors, irony, and proverbs must be interpreted semantically rather than literally.

## Data isolation

The exact Dev diagnostic sentence discussed during development is intentionally not included in the prompt. The prompt uses structurally analogous examples. Dev is used for prompt development; the final claim must be confirmed on untouched Test data.

## Run on Dev Positive 219

```bash
python scripts/run_rewriter_experiment.py \
  --config configs/rewriter/rewriter_v03_gpt4o.yaml \
  --name rewriter_v03_dev219
```

## Evaluate with the frozen Judge v02

First prepare the Judge input:

```bash
python scripts/prepare_rewrite_judge_input.py \
  runs/<V03_RUN_DIRECTORY> \
  --output data/review/rewrite_judge_input_v03_dev219.csv
```

Apply the same fixed type map by ID. Do not infer type from v03 outputs:

```bash
python scripts/apply_rewrite_type_map.py \
  --input data/review/rewrite_judge_input_v03_dev219.csv \
  --type-map data/review/rewrite_type_map_dev219.csv \
  --output data/review/rewrite_judge_input_v03_dev219_filled.csv
```

Then run:

```bash
python scripts/run_rewrite_judge.py \
  --config configs/judge/rewrite_judge_v02_gpt4o.yaml \
  --input data/review/rewrite_judge_input_v03_dev219_filled.csv \
  --name rewriter_v03_dev219_judge_v02
```

## Success criteria

Primary target:

- Overall quality >= 85 on Dev, while not reducing naturalness or intervention rate materially.

Stretch target:

- Overall quality approaching 90, with gains driven by Debiasing and Fidelity/Relevance rather than by Judge or type changes.

The Judge v02 prompt, score weights, and rewrite-type map must remain frozen during this comparison.
