# Task 4 数据飞轮设计

## 闭环

用户纠错先进入 `FeedbackStore`，校验 Worker 枚举、纠错原因和数据来源，按图片/查询/目标 Worker 去重。测试集反馈直接拒绝，避免污染固定盲测集。演示脚本使用 hard-negative 作为独立纠错挑战集，不把固定 test 写回训练。

累计反馈由 `IncrementalTrainer` 转为带来源字段的训练样本，采用旧训练数据 replay + 新反馈样本混合训练。候选模型在固定测试集上同时对 raw logits 和 gated policy 回归，只有满足两者的 macro-F1 均不下降超过 1 个百分点且任一类别 recall 不下降超过 5 个百分点时，才由 `ModelRegistry` 激活。

## 运行

```bash
python -m submission.innovation.run_feedback_experiment
```

所有模型版本保留父版本、数据来源、指标、门禁结果和时间；失败候选只登记不激活。
