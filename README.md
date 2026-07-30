# Gender Intervention Gate

一个只解决首要问题的实验项目：

> 将需要性别包容干预的 `POSITIVE` 与必须保持原样的 `NEGATIVE` 区分开，并让两类 Recall 都达到 90% 以上。

当前版本刻意不包含改写器、复杂 span schema、A/B/C 多层规则和 97 类预测。项目优先支持快速换模型、换 prompt、复现实验和定位错误。

## 数据

项目只使用两个工作簿中 `处置 == 主集` 的数据：

- Positive: 734
- Negative: 798
- Total: 1,532

模型输入始终只有 `text`。`id`、标签、类别、语体、难度等字段只用于评估，禁止进入 prompt。

原工作簿保存在：

```text
data/raw/source_workbooks/
```

已经提取的主集 CSV 和统一 JSONL 保存在：

```text
data/raw/positive_main.csv
data/raw/negative_main.csv
data/processed/main.jsonl
```

## 核心验收标准

```text
Positive Recall >= 0.90
Negative Recall >= 0.90
```

开发阶段建议目标是两类都达到 0.94，以给最终盲测留出波动空间。

## 安装

```bash
cd gender_intervention_gate
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

在 `.env` 中填入 API Key 和实际模型名称。

## 运行顺序

```bash
pytest
python scripts/validate_data.py
```

Pilot 60：

```bash
python scripts/run_experiment.py \
  --config configs/experiments/baseline_zero_shot.yaml
```

对比边界 prompt：

```bash
python scripts/run_experiment.py \
  --config configs/experiments/boundary_zero_shot.yaml
```

对比式 few-shot：

```bash
python scripts/run_experiment.py \
  --config configs/experiments/contrastive_fewshot.yaml
```

完整 dev：

```bash
python scripts/run_experiment.py \
  --config configs/experiments/contrastive_fewshot.yaml \
  --split data/splits/iid_v1/dev.jsonl \
  --name contrastive_dev_v1
```

比较实验：

```bash
python scripts/compare_runs.py runs/<run1> runs/<run2> runs/<run3>
```

分析错误：

```bash
python scripts/analyze_errors.py runs/<run_directory>
```

## 结果目录

每次运行会生成：

```text
runs/<timestamp>_<name>/
├── config.yaml
├── manifest.json
├── prompt.txt
├── examples.jsonl
├── predictions.jsonl
├── metrics.json
├── summary.md
├── by_l1.csv
├── by_l2.csv
├── positive_misses.csv
├── negative_false_alarms.csv
└── format_errors.csv
```

## 数据划分

### IID v1

- `exemplar_pool.jsonl`: 80 条候选示例，不会自动加入 prompt
- `dev.jsonl`: 400 条，用于调 prompt 和模型
- `dev_pilot_60.jsonl`: dev 的固定 60 条子集
- `test.jsonl`: 1,052 条，冻结测试集

### Subclass holdout v1

部分 L2 子类整体只进入测试集，用于检查模型是否只是记住模板和关键词。

## 设计原则

1. 第一阶段只做二分类。
2. 任何格式错误都按错误计入对应类别 Recall。
3. 不通过大量输出 `UNSURE` 绕开二分类。
4. 每次 prompt 修改必须比较“修复了什么”和“新增了什么错误”。
5. test 不用于 prompt、模型或示例选择。
