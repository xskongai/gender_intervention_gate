# Data contract

`data/processed/main.jsonl` 是模型实验的唯一主数据入口。

每条数据包含：

```json
{
  "id": "POS-0001",
  "text": "...",
  "label": "POSITIVE",
  "meta": {
    "l1": "...",
    "l2": "...",
    "register_group": "...",
    "register": "...",
    "noise": "...",
    "difficulty": "...",
    "controversial": "...",
    "source": "...",
    "original_split": "..."
  }
}
```

模型只能读取 `text`。其他字段只能用于划分、评估和错误分析。
