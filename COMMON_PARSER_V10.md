# Common Gate Output Parser v10

This update keeps the frozen Gate prompt unchanged and improves only the shared
output-normalization layer used by every model.

Accepted forms include:

- exact `POSITIVE` / `NEGATIVE`;
- exact `EDIT` / `KEEP` aliases;
- JSON such as `{"label":"POSITIVE"}`;
- explicit final markers such as `Final answer: NEGATIVE` or `最终判断：KEEP`;
- a final standalone label after explanatory text;
- labels outside a closed `<think>...</think>` block.

Ambiguous output such as `POSITIVE or NEGATIVE` remains a format error.
Empty API/model responses cannot be repaired by a parser and remain failures.

## Offline reparse

```bash
python scripts/reparse_gate_run.py --run-dir runs/<RUN_DIRECTORY>
```

The original run is not modified. New reports are written under:

```text
runs/<RUN_DIRECTORY>/reparsed_common_parser_v10/
```
