# v07 Gate Thinking Hotfix

The failed smoke run was executed with `max_output_tokens: 8`. Thinking models can spend this budget before producing the final `POSITIVE` or `NEGATIVE` label, leaving `message.content` empty.

This hotfix makes `run_cross_model.py` pass token budgets explicitly:

- Gate: 2048 tokens
- Rewriter: 4096 tokens
- GPT-4o Judge: unchanged

It also adds `--max-output-tokens` to both child runners and reports a clear error when a provider returns empty final content.

Qwen's run failed independently with HTTP 401 (`invalid_api_key`); update `QWEN_API_KEY` before rerunning.

## Rerun only Gate first

```bash
python scripts/run_experiment.py \
  --config configs/experiments/contrastive_fewshot_rule_first.yaml \
  --split data/splits/group_aware_v2.3/dev_smoke_20.jsonl \
  --model-key deepseek \
  --model deepseek-v4-flash \
  --rule-first \
  --max-output-tokens 2048 \
  --name deepseek_gate_smoke20_v07
```

Or use the cross-model runner:

```bash
python scripts/run_cross_model.py --providers gemini deepseek glm --stage smoke20
```
