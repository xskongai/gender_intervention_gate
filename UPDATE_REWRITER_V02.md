# Update: Independent Rewriter v02

## Frozen components

- Intervention Gate remains frozen at `v03_contrastive`.
- Direct Rewrite baseline remains frozen at `rewrite_v01` with the original prompt.
- Existing Pilot 60 baseline outputs and manual review are not overwritten.

## New independent component

`Rewriter v02` is a POSITIVE-only generation experiment. Its runner:

- accepts only a POSITIVE-only JSONL split;
- rejects any Negative item;
- has no `--gate-run` argument;
- does not load Gate predictions;
- reports generation diagnostics and exports a human review sheet.

## Data isolation

- Pilot Positive: 33 items extracted from Dev Pilot 60.
- Dev Positive: 219 items extracted from Dev 400.
- Prompt demonstrations: four items from the group-isolated exemplar pool.
- Held-out Test: untouched.

## Targeted error types

- norm generalization;
- factual-scope drift;
- idiom misinterpretation;
- speech-act drift;
- unsupported insertion;
- mechanical gender-term deletion.
