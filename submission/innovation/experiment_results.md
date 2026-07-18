# Task 4 数据飞轮实验结果

## 复现命令

```bash
make feedback
```

输入为 `submission/router/hard_negative.jsonl`，产物如下：

| 文件 | 作用 |
|---|---|
| `feedback.jsonl` | 已审核反馈记录 |
| `training_data.jsonl` | 已批准反馈物化的增量训练样本 |
| `react_trace.jsonl` | Observation / Reflection / Action 审计轨迹 |
| `model_registry.jsonl` | 基线与候选模型注册记录 |
| `experiment_results.json` | 完整机器可读指标 |

## 实验配置

| 配置项 | 值 |
|---|---:|
| hard-negative 环境样本数 | 12 |
| seed | 7 |
| CMA-ES generations | 90 |
| 最小反馈触发阈值 | 2 |
| replay 权重 | 1 |
| 基线 checkpoint | `submission/router/router_weights.npy` |

## 反馈闭环结果

| 指标 | 结果 |
|---|---:|
| 环境平均奖励 | 0.7708 |
| `keep_policy` | 10 |
| `approve_training_signal` | 2 |
| 自动批准反馈 | 2 |
| 导出增量训练样本 | 2 |

两条训练信号均来自可信环境金标：

| 样本 | 期望 Worker | 裸模型 | 最终路由 | 奖励 |
|---|---|---|---|---:|
| `hn-04` | Worker-B | Worker-D | Worker-D | -1.0 |
| `hn-05` | Worker-A | Worker-B | Worker-A | 0.25 |

## 训练策略对比

| 策略 | 固定测试 raw macro-F1 | 固定测试 gated macro-F1 | 挑战集 raw accuracy | 挑战集 gated accuracy | 结论 |
|---|---:|---:|---:|---:|---|
| 基线 | 0.8587 | 1.0000 | 0.8333 | 0.9167 | 发布前模型 |
| feedback only | 0.1176 | 1.0000 | 未作为候选发布 | 未作为候选发布 | 明显灾难性遗忘 |
| replay + feedback | 1.0000 | 1.0000 | 0.9167 | 1.0000 | 通过门禁 |

## 发布门禁

| 门禁 | 结果 |
|---|---|
| 固定测试 raw macro-F1 不下降超过 0.01 | 通过 |
| 固定测试 gated macro-F1 不下降超过 0.01 | 通过 |
| 固定测试各 Worker recall 不下降超过 0.05 | 通过 |
| 挑战集 gated accuracy 不下降超过 0.01 | 通过 |
| 挑战集 raw 或 gated accuracy 至少一项提升 | 通过 |

12 项门禁全部通过。候选版本 `router-react-v1` 的父版本为 `router-baseline-v1`。

## 可复现性边界

- `feedback.jsonl`、`model_registry.jsonl` 和 `experiment_results.json` 不写入本机 checkout 绝对路径。
- `router_incremental.npy` 是可重建运行产物，不纳入 git。
- 挑战集包含本轮纠错样本，提升只代表已知错误修复，不等同于未知分布泛化。
- 固定测试集未写回训练，用于回归门禁。
