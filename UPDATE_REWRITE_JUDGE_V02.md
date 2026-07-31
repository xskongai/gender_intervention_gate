# Rewrite Judge v02 Update

## Changed

- Local Repair type-specific metric changed from `no_added_facts` to `fidelity`.
- Fidelity checks both unsupported additions and preservation of non-biased content/communicative function.
- Debiasing rules now distinguish surface gender-word deletion from correction of the underlying biased proposition.
- Added explicit handling for harmful-norm generalization, idioms/proverbs/metaphors, and direct proposition reversal.
- Report output now labels the Local Repair type-specific metric dynamically.

## Unchanged

- Rewrite types: `LOCAL_REPAIR`, `PROPOSITION_RECONSTRUCTION`
- 1–3 scoring scale
- Weights: Debiasing 50%, Naturalness 25%, type-specific metric 25%
- PASS/PARTIAL/FAIL rules
- Gate and Rewriter code

## Compatibility

- v01 prompt/config are retained.
- v01 `no_added_facts` outputs remain parseable and scoreable.
