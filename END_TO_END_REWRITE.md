# End-to-End Rewrite Experiment

## Goal

Compare two systems on exactly the same split and with exactly the same rewrite model and prompt:

1. **Direct Rewrite**: every input is sent to the rewriter.
2. **Gate + Rewrite**: `NEGATIVE` Gate decisions return the original text; only `POSITIVE` decisions call the rewriter.

The frozen Gate remains `v03_contrastive`. This stage does not modify Gate labels, prompts, or the unresolved fixed-expression issue.

## Pilot commands

### Direct Rewrite

```bash
python scripts/run_rewrite_experiment.py \
  --config configs/rewrite/rewrite_gpt4o.yaml \
  --mode direct \
  --split data/splits/group_aware_v2.3/dev_pilot_60.jsonl \
  --name direct_rewrite_pilot_v23
```

### Gate + Rewrite

The Gate run may cover the full Dev set; it only needs predictions for all 60 Pilot IDs.

```bash
python scripts/run_rewrite_experiment.py \
  --config configs/rewrite/rewrite_gpt4o.yaml \
  --mode gated \
  --gate-run runs/20260731T012001Z_contrastive_dev_v23 \
  --split data/splits/group_aware_v2.3/dev_pilot_60.jsonl \
  --name gated_rewrite_pilot_v23
```

## Full Dev commands

After the Pilot files and outputs are checked, use the same frozen Gate run and replace the split with `dev.jsonl`:

```bash
python scripts/run_rewrite_experiment.py \
  --config configs/rewrite/rewrite_gpt4o.yaml \
  --mode direct \
  --split data/splits/group_aware_v2.3/dev.jsonl \
  --name direct_rewrite_dev_v23

python scripts/run_rewrite_experiment.py \
  --config configs/rewrite/rewrite_gpt4o.yaml \
  --mode gated \
  --gate-run runs/20260731T012001Z_contrastive_dev_v23 \
  --split data/splits/group_aware_v2.3/dev.jsonl \
  --name gated_rewrite_dev_v23
```

## Compare the runs

```bash
python scripts/compare_rewrite_runs.py \
  runs/<DIRECT_RUN_DIRECTORY> \
  runs/<GATED_RUN_DIRECTORY> \
  --output runs/rewrite_comparison_dev_v23.md
```

The comparison script refuses to compare runs with different split hashes, rewrite prompt hashes, or rewrite models.

## Automatic metrics

- `negative_preservation`: Negative inputs returned exactly unchanged.
- `over_edit_rate`: Negative inputs changed.
- `positive_intervention_rate`: Positive inputs changed.
- `under_edit_rate`: Positive inputs left unchanged.
- `rewrite_calls_saved`: calls avoided by the Gate.

These are endpoint behavior metrics. A changed Positive is not automatically a correct rewrite.

## Rewrite quality review

Each run creates `semantic_review_queue.csv`. Its flags are conservative triage signals based on text length, similarity, multiline output, and explanatory wording. They are **not** automatic semantic-violation labels.

For final rewrite quality, manually or independently judge at least:

- whether the target bias was removed;
- whether facts, responsibility, relationships, and stance were preserved;
- whether unsupported information was inserted;
- whether register and fluency remained acceptable.

## Fail-safe behavior

In gated mode, an invalid or missing Gate label keeps the original input and records an error. This prevents a Gate formatting failure from causing an uncontrolled rewrite.
