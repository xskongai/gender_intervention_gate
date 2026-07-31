# 确定性规则快速验证

## 1. 不调用API：验证规则自身

```bash
python validate_deterministic_gate_v01.py
```

预期结果：

- Dataset rows: 1588
- Rule-routed: 202（12.72%）
- Direct KEEP / EDIT: 99 / 103
- Gold conflicts: 0
- Disputed matches: 0
- 最后显示 `PASS`

这一步验证的是：在最新 v2.3 Gold 数据上，确定性规则没有覆盖任何错误或争议样本。

## 2. 不重跑LLM：复用已有Gate预测

找到现有 LLM-only Gate 的 CSV 或 JSONL：

```bash
python validate_deterministic_gate_v01.py \
  --llm-results /你的路径/gate_predictions.csv
```

脚本会自动比较：

- Decision Accuracy
- Positive Recall
- Negative Preservation
- Over-edit Rate
- LLM-only 与 Rule-first Hybrid 的差值

并自动断言前三个指标不能下降。

支持常见字段：

- ID：`id`、`编号`、`item_id`、`sample_id`
- 预测：`decision`、`predicted_decision`、`model_decision`、`prediction`

输出目录：

```text
validation_output/
├── rule_validation_summary.json
├── rule_matched_rows.csv
├── hybrid_comparison.json
└── hybrid_overrides.csv
```

## 3. 最终判定

快速实验通过需要同时满足：

1. Rule conflicts = 0
2. Disputed matches = 0
3. Hybrid Decision Accuracy 不下降
4. Hybrid Positive Recall 不下降
5. Hybrid Negative Preservation 不下降
6. 理论 LLM Call Rate 从100%降至87.28%

这里只验证 Gate 决策，不需要重跑 Rewriter 或 Judge。

## 本次 run_result.zip 中确认的文件

```text
runs/20260731T012001Z_contrastive_dev_v23/predictions.jsonl
```

`20260731T015826Z_gated_gated_rewrite_pilot_v23` 的配置表明，它复用了上述 Gate 运行。
