#!/usr/bin/env bash

# Run full Gender Intervention Gate evaluation on all 11 models.
# Dataset: data/processed/main.jsonl (1588 examples)
# Pipeline: Rule-first + Gate only
# No Rewriter. No Judge.
# DeepSeek API and local DeepSeek are intentionally placed last.

set -uo pipefail

# -----------------------------------------------------------------------------
# Resolve project root
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This script is intended to be stored under: scripts/run_all_11_gate_full.sh
# If it is launched from elsewhere, fall back to the current working directory.
if [[ -f "${SCRIPT_DIR}/../pyproject.toml" ]]; then
  ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "$(pwd)/pyproject.toml" ]]; then
  ROOT="$(pwd)"
else
  echo "ERROR: Cannot locate the project root containing pyproject.toml."
  echo "Save this file as scripts/run_all_11_gate_full.sh inside the project,"
  echo "or run it while your terminal is already in the project root."
  exit 1
fi

cd "$ROOT"

# -----------------------------------------------------------------------------
# Optional virtual-environment activation
# -----------------------------------------------------------------------------

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "Activated virtual environment: $ROOT/.venv"
else
  echo "WARNING: .venv/bin/activate was not found."
  echo "Using the currently active Python environment."
fi

# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------

CONFIG="configs/experiments/contrastive_fewshot_rule_first.yaml"
DATA="data/processed/main.jsonl"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/gate_full1588_${RUN_TAG}"
STATUS_FILE="${LOG_DIR}/status.tsv"

API_CONCURRENCY="${API_CONCURRENCY:-3}"
LOCAL_CONCURRENCY="${LOCAL_CONCURRENCY:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1}"

mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# Preflight checks
# -----------------------------------------------------------------------------

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Experiment config not found: $CONFIG"
  exit 1
fi

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: Dataset not found: $DATA"
  exit 1
fi

if [[ ! -f "scripts/run_experiment.py" ]]; then
  echo "ERROR: Gate runner not found: scripts/run_experiment.py"
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python command not found."
  exit 1
fi

printf "order\tmodel_key\trun_name\tstatus\texit_code\tlog_file\n" > "$STATUS_FILE"

# -----------------------------------------------------------------------------
# Gate runner
# -----------------------------------------------------------------------------

run_gate() {
  local order="$1"
  local model_key="$2"
  local run_name="$3"
  local concurrency="$4"
  shift 4

  local log_file="${LOG_DIR}/${order}_${run_name}.log"

  echo
  echo "================================================================================"
  echo "[$order/11] Running Gate model: $model_key"
  echo "Run name:       $run_name"
  echo "Dataset:        $DATA"
  echo "Configuration:  $CONFIG"
  echo "Pipeline:       Rule-first + Gate only"
  echo "Concurrency:    $concurrency"
  echo "Log file:       $log_file"
  echo "Started:        $(date)"
  echo "================================================================================"

  python -u scripts/run_experiment.py \
    --config "$CONFIG" \
    --split "$DATA" \
    --model-key "$model_key" \
    --rule-first \
    --concurrency "$concurrency" \
    --progress-every "$PROGRESS_EVERY" \
    --name "$run_name" \
    "$@" \
    2>&1 | tee "$log_file"

  local code=${PIPESTATUS[0]}

  if [[ "$code" -eq 0 ]]; then
    printf "%s\t%s\t%s\tSUCCESS\t%s\t%s\n" \
      "$order" "$model_key" "$run_name" "$code" "$log_file" >> "$STATUS_FILE"
    echo "SUCCESS: $model_key"
  else
    printf "%s\t%s\t%s\tFAILED\t%s\t%s\n" \
      "$order" "$model_key" "$run_name" "$code" "$log_file" >> "$STATUS_FILE"
    echo "FAILED: $model_key (exit code: $code)"
    echo "Continuing with the next model."
  fi
}

record_skipped() {
  local order="$1"
  local model_key="$2"
  local run_name="$3"
  local reason="$4"

  printf "%s\t%s\t%s\tSKIPPED\t-\t%s\n" \
    "$order" "$model_key" "$run_name" "$reason" >> "$STATUS_FILE"

  echo "SKIPPED: $model_key — $reason"
}

# -----------------------------------------------------------------------------
# Experiment header
# -----------------------------------------------------------------------------

echo "================================================================================"
echo "Gender Intervention Gate — Full 11-model evaluation"
echo "Started:             $(date)"
echo "Project root:        $ROOT"
echo "Dataset:             $DATA"
echo "API concurrency:     $API_CONCURRENCY"
echo "Local concurrency:   $LOCAL_CONCURRENCY"
echo "Progress interval:   $PROGRESS_EVERY"
echo "Output logs:         $LOG_DIR"
echo "Status file:         $STATUS_FILE"
echo "DeepSeek models:     positions 10 and 11"
echo "================================================================================"

# -----------------------------------------------------------------------------
# 1–4. Faster API models
# Model names are taken from the existing project config/.env.
# -----------------------------------------------------------------------------

run_gate \
  "01" \
  "openai" \
  "openai_gate_full1588" \
  "$API_CONCURRENCY"

run_gate \
  "02" \
  "gemini" \
  "gemini_gate_full1588" \
  "$API_CONCURRENCY"

run_gate \
  "03" \
  "qwen" \
  "qwen_api_gate_full1588" \
  "$API_CONCURRENCY"

run_gate \
  "04" \
  "glm" \
  "glm_api_gate_full1588" \
  "$API_CONCURRENCY"

# -----------------------------------------------------------------------------
# Check Ollama once before all local-model runs.
# Failure does not prevent the final DeepSeek API run.
# -----------------------------------------------------------------------------

LOCAL_AVAILABLE=1
LOCAL_UNAVAILABLE_REASON=""

echo
echo "================================================================================"
echo "Checking Ollama for local models"
echo "================================================================================"

if ! command -v ollama >/dev/null 2>&1; then
  LOCAL_AVAILABLE=0
  LOCAL_UNAVAILABLE_REASON="ollama command not found"
elif ! ollama list >/dev/null 2>&1; then
  LOCAL_AVAILABLE=0
  LOCAL_UNAVAILABLE_REASON="Ollama service is not running"
else
  echo "Ollama is available. Installed models:"
  ollama list
fi

# -----------------------------------------------------------------------------
# 5–9. Faster local Ollama models
# -----------------------------------------------------------------------------

if [[ "$LOCAL_AVAILABLE" -eq 1 ]]; then
  run_gate \
    "05" \
    "qwen3_5_9b_ollama" \
    "qwen3_5_9b_ollama_gate_full1588_nothink" \
    "$LOCAL_CONCURRENCY"

  run_gate \
    "06" \
    "glm4_9b_ollama" \
    "glm4_9b_ollama_gate_full1588_nothink" \
    "$LOCAL_CONCURRENCY"

  run_gate \
    "07" \
    "gemma2_9b_ollama" \
    "gemma2_9b_ollama_gate_full1588_nothink" \
    "$LOCAL_CONCURRENCY"

  run_gate \
    "08" \
    "llama3_1_8b_ollama" \
    "llama3_1_8b_ollama_gate_full1588_native_schema" \
    "$LOCAL_CONCURRENCY"

  run_gate \
    "09" \
    "mistral_7b_ollama" \
    "mistral_7b_ollama_gate_full1588_nothink" \
    "$LOCAL_CONCURRENCY"
else
  echo "Local models unavailable: $LOCAL_UNAVAILABLE_REASON"
  record_skipped "05" "qwen3_5_9b_ollama" \
    "qwen3_5_9b_ollama_gate_full1588_nothink" "$LOCAL_UNAVAILABLE_REASON"
  record_skipped "06" "glm4_9b_ollama" \
    "glm4_9b_ollama_gate_full1588_nothink" "$LOCAL_UNAVAILABLE_REASON"
  record_skipped "07" "gemma2_9b_ollama" \
    "gemma2_9b_ollama_gate_full1588_nothink" "$LOCAL_UNAVAILABLE_REASON"
  record_skipped "08" "llama3_1_8b_ollama" \
    "llama3_1_8b_ollama_gate_full1588_native_schema" "$LOCAL_UNAVAILABLE_REASON"
  record_skipped "09" "mistral_7b_ollama" \
    "mistral_7b_ollama_gate_full1588_nothink" "$LOCAL_UNAVAILABLE_REASON"
fi

# -----------------------------------------------------------------------------
# 10. Slow DeepSeek API model
# Explicit model override prevents accidental use of DeepSeek Flash.
# -----------------------------------------------------------------------------

run_gate \
  "10" \
  "deepseek" \
  "deepseek_v4_pro_gate_full1588" \
  "$API_CONCURRENCY" \
  --model "deepseek-v4-pro"

# -----------------------------------------------------------------------------
# 11. Slow local DeepSeek model
# -----------------------------------------------------------------------------

if [[ "$LOCAL_AVAILABLE" -eq 1 ]]; then
  run_gate \
    "11" \
    "deepseek_r1_8b_ollama" \
    "deepseek_r1_8b_ollama_gate_full1588_native_nothink_plain" \
    "$LOCAL_CONCURRENCY"
else
  record_skipped "11" "deepseek_r1_8b_ollama" \
    "deepseek_r1_8b_ollama_gate_full1588_native_nothink_plain" \
    "$LOCAL_UNAVAILABLE_REASON"
fi

# -----------------------------------------------------------------------------
# Final report
# -----------------------------------------------------------------------------

echo
echo "================================================================================"
echo "All 11 requested model runs have finished or been attempted."
echo "Completed:   $(date)"
echo "Status file: $STATUS_FILE"
echo "================================================================================"
echo

if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$STATUS_FILE"
else
  cat "$STATUS_FILE"
fi

echo
echo "Generated run directories:"
find runs -maxdepth 1 -type d -name "*gate_full1588*" -print 2>/dev/null | sort || true

echo
echo "Logs are stored in:"
echo "$LOG_DIR"
