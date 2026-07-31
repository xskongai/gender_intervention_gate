# Ollama Native Runtime Fix v11

## Why this patch exists

The Smoke20 logs showed two different failure modes:

- DeepSeek-R1 returned empty final content with `finish_reason='length'` even though
  `reasoning_effort: none` was sent through the OpenAI-compatible endpoint.
- Llama 3.1 returned Chinese refusal prose for five positive examples; this was
  not a parser failure because no valid label was present.

## Runtime repair

- DeepSeek-R1 now uses Ollama native `/api/chat` with `think: false`.
- DeepSeek-R1 and Llama 3.1 use a two-label JSON schema so the runtime can only
  return `POSITIVE` or `NEGATIVE` in the `label` field.
- The frozen Gate prompt and contrastive examples are unchanged.
- Qwen, GLM, Gemma and Mistral retain their previous OpenAI-compatible profiles.

## Validation

```bash
python -m pytest -q
python -u scripts/run_local_gate.py --models deepseek llama --stage smoke20
```

Do not reuse old Llama cache entries: the provider key changes from
`openai_compatible` to `ollama_native`, so the request cache key also changes.
