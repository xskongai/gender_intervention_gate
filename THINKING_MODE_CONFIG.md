# Thinking-mode target model configuration

Target models:

- DeepSeek: `deepseek-v4-flash`
- Qwen: `qwen3.7-plus`
- GLM: `glm-5.2`
- Judge: fixed `gpt-4o` (unchanged)

The provider-specific switches live in `configs/models.yaml`:

```yaml
deepseek:
  extra_body:
    thinking:
      type: enabled
    reasoning_effort: high

qwen:
  extra_body:
    enable_thinking: true

glm:
  extra_body:
    thinking:
      type: enabled
    reasoning_effort: high
```

`max_output_tokens` is raised to 2048 for Gate and 4096 for Rewriter because reasoning tokens share the output budget. The GPT-4o Judge configuration remains unchanged.

Run a 20-item smoke test:

```bash
python scripts/run_cross_model.py --providers deepseek --stage smoke20
python scripts/run_cross_model.py --providers qwen --stage smoke20
python scripts/run_cross_model.py --providers glm --stage smoke20
```

Run a configuration-only check:

```bash
python scripts/run_cross_model.py --providers deepseek qwen glm --stage smoke20 --dry-run
```
