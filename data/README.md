# Data contract — v2.3

`data/processed/main.jsonl` is the single dataset entry point used by experiments.

Counts:

- POSITIVE: 871
- NEGATIVE: 717
- Total: 1,588

Each item contains `id`, `text`, `label`, and `meta`. The model is allowed to see only `text`; all metadata is reserved for splitting, evaluation, and error analysis.

Important metadata added in v2.3:

- `dataset_version`: always `v2.3`
- `reference_output`: inclusive rewrite for POSITIVE, unchanged text for NEGATIVE
- `template_group`: shared group ID for the 414 retained template-derived Positive samples
- `split_group`: template group when present, otherwise the item ID

Source chain:

```text
data/raw/source_workbooks/*.xlsx
  -> data/raw/positive_main.csv + negative_main.csv
  -> data/processed/main.jsonl
  -> data/splits/group_aware_v2.3/*
```

Use `group_aware_v2.3` for reported experiments. `iid_v2.3` is retained only as a leakage-prone comparison baseline.
