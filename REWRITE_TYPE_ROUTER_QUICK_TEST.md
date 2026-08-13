# Rewrite Type Router — Quick Test

## What was added

- `src/gender_gate/rewrite_type_router.py`
- `prompts/rewrite_type_router_v01.txt`
- `scripts/run_rewrite_type_router.py`
- `tests/test_rewrite_type_router.py`

The router predicts only one of:

- `LOCAL_REPAIR`
- `PROPOSITION_RECONSTRUCTION`

It does not rewrite the sentence. It uses the original sentence only.

## 1. Offline smoke test

```bash
pytest -q tests/test_rewrite_type_router.py
```

Expected: `3 passed`.

## 2. Fast real test: 20 balanced examples

Make sure `OPENAI_API_KEY` is set, then run:

```bash
python scripts/run_rewrite_type_router.py \
  --sample-size 20 \
  --balanced \
  --concurrency 5 \
  --output-dir runs/rewrite_type_router/quick20_v01
```

The 20-item sample is balanced against the existing gold type map: 10 Local Repair + 10 Proposition Reconstruction.

Outputs:

- `predictions.csv`: per-item gold/prediction/error
- `metrics.json`: Accuracy, Macro-F1, per-class Precision/Recall/F1
- `rewrite_type_map_auto.csv`: same 3-column schema as the existing frozen type map

## 3. If 20 looks good, test 100

```bash
python scripts/run_rewrite_type_router.py \
  --sample-size 100 \
  --balanced \
  --concurrency 5 \
  --output-dir runs/rewrite_type_router/quick100_v01
```

## 4. Full 871

```bash
python scripts/run_rewrite_type_router.py \
  --sample-size 871 \
  --concurrency 5 \
  --output-dir runs/rewrite_type_router/full871_v01
```

After full prediction, the generated map is:

```text
runs/rewrite_type_router/full871_v01/rewrite_type_map_auto.csv
```

To use automatic types in the existing Adaptive Rewriter, set its `type_map` config to that file. No Judge or Adaptive Rewriter code change is required.
