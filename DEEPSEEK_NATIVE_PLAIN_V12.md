# DeepSeek native plain-output fix v12

The DeepSeek-R1 local profile uses Ollama's native `/api/chat` endpoint with:

- `think: false`
- no JSON schema (`structured_output: false`)
- `num_predict: 256`
- temperature `0.0`

This avoids the observed interaction in which `deepseek-r1:8b` consumed the full
structured-output token budget in `message.thinking` and returned empty
`message.content`. The frozen Gate prompt, deterministic rules, labels, and
metrics are unchanged.

Llama 3.1 remains on native structured output because that configuration reduced
its format-error rate to zero.

## Run

```bash
python -u scripts/run_local_gate.py --models deepseek --stage smoke20
```

The run name ends in `native_nothink_plain`.
