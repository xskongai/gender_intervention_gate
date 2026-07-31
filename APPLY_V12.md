# Apply v12 patch

From the existing project root:

```bash
unzip -o ~/Downloads/gender_gate_deepseek_plain_patch_v12.zip -d .
pip install -e '.[dev]'
python -m pytest -q
python -u scripts/run_local_gate.py --models deepseek --stage smoke20
```

Expected test result: `83 passed`.

DeepSeek's new run name ends with `native_nothink_plain`. Llama remains
`native_schema`.
