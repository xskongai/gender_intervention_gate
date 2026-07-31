# v2.3 project update

- Replaced the project source workbooks with the finalized main-only datasets.
- Positive: 871; Negative: 717; total: 1,588.
- Regenerated `data/raw/positive_main.csv` and `negative_main.csv`.
- Regenerated `data/processed/main.jsonl` with dataset version, reference output, template group, and split group metadata.
- Added leakage-controlled `data/splits/group_aware_v2.3` and comparison `data/splits/iid_v2.3`.
- Switched all default experiment configs to `group_aware_v2.3/dev_pilot_60.jsonl`.
- Updated contrastive examples so every referenced ID exists in v2.3.
- Updated tests, manifests, and documentation.
- Removed stale v2.1 review queue artifacts; retained a compact v2.3 change log.
- Validation result: 11 tests passed.

## End-to-end rewrite runner

Added in code version 0.3.0:

- `scripts/run_rewrite_experiment.py`
- `scripts/compare_rewrite_runs.py`
- `configs/rewrite/rewrite_gpt4o.yaml`
- `prompts/rewrite_v01.txt`
- change-based endpoint metrics and conservative semantic review queues
- offline `--mock-oracle` plumbing mode, explicitly excluded from experimental reporting
