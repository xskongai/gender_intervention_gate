# Multi-model Rule-first Gate + Rewriter

## Frozen experimental design

- Gate: `Rule-first + Frozen LLM Gate v03_contrastive`
- Rewriter: `v02 semantic-preserving`
- Target model: the same provider/model is used for Gate and Rewriter
- Judge: always `GPT-4o`, prompt `v04 Balanced`
- No provider-specific prompt tuning

Supported target providers:

- `openai`
- `gemini`
- `deepseek`
- `qwen`
- `glm`
- `local` (OpenAI-compatible endpoint)

## 1. Environment

```bash
cp .env.example .env
```

Fill the API key and exact model id for each provider that will be tested. The Judge reads `OPENAI_API_KEY` but its model is fixed in `configs/models.yaml` as `gpt-4o`; it does not reuse the target-model environment variable.

Install/update the project:

```bash
pip install -e '.[dev]'
python -m pytest -q
```

## 2. One-call provider check

```bash
python scripts/check_model_access.py --provider deepseek
python scripts/check_model_access.py --provider qwen
python scripts/check_model_access.py --provider glm
python scripts/check_model_access.py --provider gemini
python scripts/check_model_access.py --provider openai_judge
```

An exact model can be supplied without editing `.env`:

```bash
python scripts/check_model_access.py \
  --provider deepseek \
  --model YOUR_EXACT_MODEL_ID
```

## 3. Progressive experiments

### Stage 1: Smoke 20

- Gate: 20 items = 10 Positive + 10 Negative
- Rule-routed: 4
- Target-model Gate calls: 16
- Rewriter: 10 Positive items
- GPT-4o Judge: 10 items

Run one provider:

```bash
python scripts/run_cross_model.py \
  --providers deepseek \
  --stage smoke20
```

Run all new providers sequentially:

```bash
python scripts/run_cross_model.py \
  --providers gemini deepseek qwen glm \
  --stage smoke20
```

### Stage 2: Pilot 60

```bash
python scripts/run_cross_model.py \
  --providers deepseek \
  --stage pilot60
```

- Gate: 60 items
- Rule-routed: 10
- Target-model Gate calls: 50
- Rewriter/Judge: 33 Positive items

### Stage 3: Dev 400 / Positive 219

```bash
python scripts/run_cross_model.py \
  --providers deepseek \
  --stage dev400
```

- Gate: 400 items
- Rule-routed: 68
- Target-model Gate calls: 332
- Rewriter/Judge: 219 Positive items

Only promote a provider after its previous stage has zero or near-zero request/format errors and no obvious all-KEEP/all-EDIT collapse.

## Useful options

Exact model override for a single provider:

```bash
python scripts/run_cross_model.py \
  --providers qwen \
  --model YOUR_EXACT_QWEN_MODEL_ID \
  --stage smoke20
```

Lower concurrency when a provider rate-limits:

```bash
python scripts/run_cross_model.py \
  --providers gemini \
  --stage smoke20 \
  --concurrency 2
```

Print and validate commands without API calls:

```bash
python scripts/run_cross_model.py \
  --providers gemini \
  --model YOUR_EXACT_GEMINI_MODEL_ID \
  --stage smoke20 \
  --dry-run
```

Temporarily skip the Judge while debugging only:

```bash
python scripts/run_cross_model.py \
  --providers deepseek \
  --stage smoke20 \
  --skip-judge
```

## Outputs

Each provider/stage creates:

```text
runs/cross_model/<timestamp>_<provider>_<stage>/
  judge_input.csv
  summary.json
  summary.md
```

`summary.json` combines:

- Gate Positive Recall
- Negative Preservation
- Balanced Accuracy
- Rule Coverage
- LLM Call Rate
- Rewriter error/under-edit rates
- GPT-4o Judge Overall and Macro Quality
- Debiasing, Naturalness, Type-specific score, and Pass Rate

The original detailed Gate, Rewriter, and Judge run directories are preserved and referenced from this summary.

A multi-provider invocation also writes a compact paper-table-ready matrix CSV next to the matrix JSON:

```text
runs/cross_model/<timestamp>_<stage>_matrix_summary.csv
```
