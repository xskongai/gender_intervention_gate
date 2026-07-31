# Rule-first + Frozen LLM Gate

The frozen contrastive LLM Gate is unchanged. A conservative deterministic
front route can now be enabled or disabled from configuration or CLI.

## Configuration

```yaml
rule_first:
  enabled: true
  ruleset: deterministic_v01
```

- `enabled: true`: deterministic sufficient-condition rules run first; all
  unmatched items fall back to the same frozen LLM Gate.
- `enabled: false`: original LLM-only behavior.

## Ready-to-use configurations

LLM-only:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/contrastive_fewshot.yaml
```

Rule-first:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/contrastive_fewshot_rule_first.yaml
```

Full development set:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/contrastive_fewshot_rule_first.yaml \
  --split data/splits/group_aware_v2.3/dev.jsonl \
  --name contrastive_dev_v23_rule_first
```

## CLI override

Enable Rule-first without editing YAML:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/contrastive_fewshot.yaml \
  --rule-first
```

Disable it for an ablation:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/contrastive_fewshot_rule_first.yaml \
  --no-rule-first
```

## Outputs

Each prediction records:

- `route`: `RULE` or `LLM`
- `rule`: deterministic rule ID, or `null`
- `ruleset`: `deterministic_v01` when enabled

Each run additionally writes `rule_routed.csv`. `metrics.json` and `summary.md`
include:

- rule coverage
- LLM call rate
- observed rule accuracy
- per-rule counts

The manifest records whether Rule-first was enabled and the SHA-256 of the
frozen deterministic rules file.

## Offline rule audit

No API call is required:

```bash
python scripts/audit_rule_first.py \
  --split data/splits/group_aware_v2.3/dev.jsonl \
  --output runs/rule_first_audit_dev
```

Current frozen Dev result:

- 400 total rows
- 68 deterministically routed
- 17.0% rule coverage
- 0 Gold conflicts
- 332 LLM fallbacks
