# Independent Rewriter Experiment

## Scope

The Rewriter is evaluated independently from the Intervention Gate.

- Input: POSITIVE instances only.
- Output: one gender-inclusive rewritten sentence.
- The runner does not load, read, or accept Gate predictions.
- Negative preservation and over-editing belong to the Gate/direct-system experiments, not this experiment.

## Frozen baseline

`Rewrite v01` uses the original direct prompt:

- Prompt: `prompts/rewrite_original_direct.txt`
- Config: `configs/rewriter/rewriter_v01_gpt4o.yaml`

The previously collected pilot outputs remain the baseline evidence. Running the independent v01 config is optional when the outputs are already available.

## Candidate

`Rewrite v02` targets the failure modes found in the 33-item Positive pilot:

1. norm generalization;
2. factual-scope expansion or contraction;
3. idiom and metaphor misinterpretation;
4. speech-act and register drift;
5. unsupported moralizing or invented conditions;
6. mechanical gender-term deletion.

Prompt examples come only from the group-isolated `exemplar_pool`, not from Dev or Test.

- Prompt: `prompts/rewriter_v02_semantic_preserving.txt`
- Config: `configs/rewriter/rewriter_v02_gpt4o.yaml`

## Positive-only splits

- Pilot: `data/splits/group_aware_v2.3/dev_pilot_positive_33.jsonl`
- Full Dev: `data/splits/group_aware_v2.3/dev_positive_219.jsonl`

No positive-only Test file is created yet. Test remains untouched until the Rewriter and manual evaluation protocol are frozen.

## Run v02 on the 33-item pilot

```bash
python scripts/run_rewriter_experiment.py \
  --config configs/rewriter/rewriter_v02_gpt4o.yaml \
  --name rewriter_v02_pilot33
```

The run produces:

- `predictions.jsonl`
- `predictions.csv`
- `metrics.json`
- `unchanged_outputs.csv`
- `errors.csv`
- `manual_review.csv`
- `summary.md`
- `manifest.json`

## Manual evaluation

Automatic change rate is not rewrite success. Complete `manual_review.csv` using:

- `bias_removed`: YES / PARTIAL / NO
- `semantic_preserved`: YES / PARTIAL / NO
- `unsupported_insertion`: YES / NO
- `meaning_distortion`: YES / NO
- `fluency`: YES / PARTIAL / NO
- `verdict`: PASS / PARTIAL / FAIL

A strict PASS requires:

1. the gender-related problem is removed;
2. the non-problematic meaning and speech act are preserved;
3. no unsupported fact, reason, or moral conclusion is introduced;
4. the output is natural Chinese.

## Compare v01 and v02

Use the existing 60-item Direct v01 run as the baseline; the comparison script automatically selects its 33 Positive items:

```bash
python scripts/compare_rewriter_runs.py \
  runs/20260731T015757Z_direct_direct_rewrite_pilot_v23 \
  runs/<V02_RUN> \
  --output-dir runs/rewriter_v01_vs_v02_pilot33
```

Use `paired_manual_review.csv` for the authoritative comparison. Exact reference match is diagnostic only.

## Decision rule before Full Dev

Proceed to the 219-item Dev Positive split only if v02 improves strict PASS rate or reduces the named failure modes without materially increasing meaning distortion or unsupported insertion.

## Candidate v03

`Rewrite v03` adds a minimal-sufficient constraint: completely remove the core biased proposition, then minimize all other textual changes.

- Prompt: `prompts/rewriter_v03_minimal_sufficient.txt`
- Config: `configs/rewriter/rewriter_v03_gpt4o.yaml`
- Detailed instructions: `UPDATE_REWRITER_V03.md`

```bash
python scripts/run_rewriter_experiment.py \
  --config configs/rewriter/rewriter_v03_gpt4o.yaml \
  --name rewriter_v03_dev219
```

Evaluate v03 with the frozen Judge v02 and the same frozen rewrite-type map used for v01 and v02.
