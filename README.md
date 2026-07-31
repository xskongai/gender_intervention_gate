# Gender Intervention Gate v2.3

A focused binary-classification project for Chinese gender-inclusive rewriting:

> Distinguish text that contains sufficient internal evidence for gender-inclusive intervention (`POSITIVE`) from text whose gender information must be preserved (`NEGATIVE`).

The acceptance target is:

```text
Positive Recall >= 0.90
Negative Recall >= 0.90
```

## Dataset v2.3

Only finalized main-set rows are included:

- Positive: 871
- Negative: 717
- Total: 1,588

The model input is always `text`. IDs, labels, categories, source, template groups, difficulty, and reference rewrites are evaluation metadata and must never be inserted into the prompt.

Canonical files:

```text
data/raw/source_workbooks/positive_v2.3_main_only_clean.xlsx
data/raw/source_workbooks/negative_v2.3_main_only_clean.xlsx
data/raw/positive_main.csv
data/raw/negative_main.csv
data/processed/main.jsonl
```

## Leakage-controlled split

The recommended split is:

```text
data/splits/group_aware_v2.3/
```

- exemplar pool: 80
- dev: 400
- fixed pilot: 60 (subset of dev)
- frozen test: 1,108

The 414 template-derived Positive samples belong to 34 template groups. A template group is kept entirely within exemplar, dev, or test, preventing near-identical template variants from leaking across splits.

For comparison only, the project also contains a row-level stratified split:

```text
data/splits/iid_v2.3/
```

Do not use IID results as the primary paper result because template leakage can inflate performance.

## Setup

```bash
cd gender_intervention_gate
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Add the required API keys and model names to `.env` and `configs/models.yaml`.

## Validate the updated dataset

```bash
pytest
python scripts/validate_data.py --positive 871 --negative 717
```

To reconstruct the unified JSONL from the two CSV files:

```bash
python scripts/prepare_data.py
```

To reconstruct both split variants:

```bash
python scripts/make_splits.py \
  --mode group-aware \
  --output-dir data/splits/group_aware_v2.3

python scripts/make_splits.py \
  --mode iid \
  --output-dir data/splits/iid_v2.3
```

## Run experiments

Pilot 60:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/baseline_zero_shot.yaml
```

Boundary prompt:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/boundary_zero_shot.yaml
```

Contrastive few-shot:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/contrastive_fewshot.yaml
```

Full dev:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/contrastive_fewshot.yaml \
  --split data/splits/group_aware_v2.3/dev.jsonl \
  --name contrastive_dev_v23
```

The test split should remain frozen until prompt, examples, model selection, and thresholds are finalized.

## Outputs

Each experiment writes a reproducible run directory containing the copied config and prompt, predictions, metrics, error slices, and SHA-256 hashes of the dataset and split.

## End-to-end rewrite comparison

Use the same rewrite model, prompt, split, temperature, and token settings for both systems. The only difference is whether the frozen Gate is applied before rewriting.

### 1. Direct Rewrite

```bash
python scripts/run_rewrite_experiment.py \
  --config configs/rewrite/rewrite_gpt4o.yaml \
  --mode direct \
  --split data/splits/group_aware_v2.3/dev_pilot_60.jsonl \
  --name direct_rewrite_pilot_v23
```

### 2. Gate + Rewrite

Pass the completed Gate run directory whose `predictions.jsonl` covers the same split:

```bash
python scripts/run_rewrite_experiment.py \
  --config configs/rewrite/rewrite_gpt4o.yaml \
  --mode gated \
  --gate-run runs/<GATE_RUN_DIRECTORY> \
  --split data/splits/group_aware_v2.3/dev_pilot_60.jsonl \
  --name gated_rewrite_pilot_v23
```

For the full development set, replace the split with:

```text
data/splits/group_aware_v2.3/dev.jsonl
```

Each rewrite run produces:

- `predictions.jsonl` and `predictions.csv`
- `metrics.json` and `summary.md`
- `positive_failures.csv`
- `negative_over_edits.csv`
- `semantic_review_queue.csv`
- `errors.csv`
- `manifest.json`

The automatic endpoint metrics are change-based:

- **Negative preservation**: proportion of Negative inputs returned unchanged.
- **Over-edit rate**: proportion of Negative inputs changed.
- **Positive intervention rate**: proportion of Positive inputs changed.
- **Under-edit rate**: proportion of Positive inputs left unchanged.

A changed Positive is not automatically a successful rewrite. Review `semantic_review_queue.csv` and score bias removal and semantic preservation separately before reporting final rewrite quality.

### 3. Compare two runs

```bash
python scripts/compare_rewrite_runs.py \
  runs/<DIRECT_RUN_DIRECTORY> \
  runs/<GATED_RUN_DIRECTORY> \
  --output runs/rewrite_comparison.md
```

### Offline plumbing smoke test

`--mock-oracle` is provided only to verify the pipeline without API calls. Its metrics must never be reported as experimental results.

