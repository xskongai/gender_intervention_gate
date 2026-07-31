#!/usr/bin/env bash
set -euo pipefail

# Run the frozen Judge v04 Balanced on every available 219-item Rewriter input.
# Execute from the gender_intervention_gate project root.

CONFIG="configs/judge/rewrite_judge_v04_balanced_gpt4o.yaml"
RUN_INDEX="runs/judge_v04_balanced_run_index.tsv"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: $CONFIG not found. Run this script from the project root." >&2
  exit 1
fi

mkdir -p runs
printf "version\tinput\trun_dir\n" > "$RUN_INDEX"

find_input() {
  local version="$1"
  local filename="rewrite_judge_input_${version}_dev219_filled.csv"
  local candidates=(
    "$filename"
    "data/review/$filename"
    "data/$filename"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

ran=0
for version in v01 v02 v03 v04; do
  if input_path="$(find_input "$version")"; then
    name="rewriter_${version}_dev219_judge_v04_balanced"
    log_file="runs/${name}.log"

    echo "============================================================"
    echo "Running $version with frozen Judge v04 Balanced"
    echo "Input: $input_path"

    python scripts/run_rewrite_judge.py \
      --config "$CONFIG" \
      --input "$input_path" \
      --name "$name" | tee "$log_file"

    run_dir="$(grep '^Run directory:' "$log_file" | tail -1 | sed 's/^Run directory: //')"
    if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
      echo "ERROR: Could not identify run directory for $version." >&2
      exit 1
    fi

    printf "%s\t%s\t%s\n" "$version" "$input_path" "$run_dir" >> "$RUN_INDEX"
    ran=$((ran + 1))
  else
    echo "SKIP $version: rewrite_judge_input_${version}_dev219_filled.csv not found."
  fi
done

if [[ "$ran" -eq 0 ]]; then
  echo "ERROR: No filled 219-item Judge inputs were found." >&2
  exit 1
fi

echo "============================================================"
echo "Completed $ran run(s)."
echo "Run index: $RUN_INDEX"
echo "Next command:"
echo "  python scripts/summarize_frozen_judge_v04.py --index $RUN_INDEX"
