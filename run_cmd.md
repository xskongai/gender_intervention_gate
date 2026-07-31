

python scripts/run_rewriter_experiment.py \
  --config configs/rewriter/rewriter_v02_gpt4o.yaml \
  --name rewriter_v02_pilot33


V02_RUN=$(ls -td runs/*_rewriter_rewriter_v02_pilot33 | head -1)

python scripts/compare_rewriter_runs.py \
  "runs/20260731T015757Z_direct_direct_rewrite_pilot_v23" \
  "$V02_RUN" \
  --output-dir runs/rewriter_v01_vs_v02_pilot33