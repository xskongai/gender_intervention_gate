# Local 7B–9B Gate profiles (no thinking)

The local Gate profiles use Ollama's OpenAI-compatible endpoint and keep the
frozen Rule-first + LLM Gate unchanged. Qwen3.5 and DeepSeek-R1 explicitly send
`reasoning_effort: none`; the other local models are non-reasoning profiles.

## Profiles

| Alias | model_key | Ollama model | Max tokens |
|---|---|---|---:|
| qwen | qwen3_5_9b_ollama | qwen3.5:9b | 256 |
| deepseek | deepseek_r1_8b_ollama | deepseek-r1:8b | 512 |
| glm | glm4_9b_ollama | glm4:9b | 512 |
| gemma | gemma2_9b_ollama | gemma2:9b | 512 |
| llama | llama3_1_8b_ollama | llama3.1:latest | 512 |
| mistral | mistral_7b_ollama | mistral:latest | 512 |

## Quick access check

```bash
python scripts/check_model_access.py --provider qwen3_5_9b_ollama
```

## Run one model

```bash
python -u scripts/run_local_gate.py --models qwen --stage smoke20
python -u scripts/run_local_gate.py --models qwen --stage pilot60
python -u scripts/run_local_gate.py --models qwen --stage dev400
```

## Run all six sequentially

```bash
python -u scripts/run_local_gate.py --models all --stage smoke20
```

The runner defaults to `concurrency=1` and prints every completed item. Models
run sequentially to avoid repeatedly loading several 5–7 GB models into unified
memory at the same time.
