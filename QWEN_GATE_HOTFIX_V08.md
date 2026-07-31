# Qwen Gate Hotfix v08

## Root cause addressed

Qwen3.7 thinking was enabled without a bounded `thinking_budget`, while the
shared runner still sent the legacy `max_tokens` field. The Gate output showed
that every LLM-routed item failed, leaving only Rule-first decisions in the
metrics.

## Changes

- Qwen now uses `max_completion_tokens`.
- Qwen Gate reasoning is capped at 512 tokens.
- Qwen Rewriter reasoning is capped at 1536 tokens in cross-model runs.
- Provider and per-run `extra_body` dictionaries are merged safely.
- `--thinking-budget` is available in Gate and Rewriter runners.
- Gate output now prints `Format error rate`.
- `scripts/debug_qwen_response.py` prints finish reason, reasoning length, and
  final content for one safe diagnostic request.

## First validation

```bash
python scripts/check_model_access.py \
  --provider qwen \
  --model qwen3.7-plus \
  --thinking-budget 128

python scripts/debug_qwen_response.py \
  --model qwen3.7-plus \
  --thinking-budget 512 \
  --max-output-tokens 2048
```

Expected diagnostic:

```text
Finish reason: 'stop'
Final content: 'POSITIVE'
```

Then rerun Smoke20:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/contrastive_fewshot_rule_first.yaml \
  --split data/splits/group_aware_v2.3/dev_smoke_20.jsonl \
  --model-key qwen \
  --model qwen3.7-plus \
  --rule-first \
  --max-output-tokens 2048 \
  --thinking-budget 512 \
  --name qwen_gate_smoke20_v08
```

Do not move to Dev400 until `Format error rate` is `0.0000`.
