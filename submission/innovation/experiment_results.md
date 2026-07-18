# Task 4 实验结果

运行命令：

```bash
python -m submission.innovation.run_feedback_experiment
```

实验脚本默认从固定测试集选取 5 条反馈样本，但这些样本只作为演示输入；训练器不会把测试集写回主训练 JSONL。候选模型使用旧训练数据 replay + 新反馈样本，随后在固定测试集上计算 gated accuracy、macro-F1 和各类别 recall。

门禁规则：raw logits 与 gated policy 两套结果的 macro-F1 均不得下降超过 1 个百分点，且任一类别 recall 均不得下降超过 5 个百分点。通过后才激活新版本，否则只登记候选模型并保留基线版本。

实际数字以同目录生成的 `experiment_results.json` 为准，避免把一次小样本演示结果误认为稳定泛化结论。
