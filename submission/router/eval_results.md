# Task 2 评估结果

## 1. 数据与口径

| 项目 | 值 |
|---|---:|
| 总样本 | 91 |
| 训练集 / 验证集 / 测试集 | 60 / 16 / 15 |
| Worker 类别 | A / B / C / D |
| 路由头参数量 | 708 |
| 优化器 | sep-CMA-ES，seed=7，90 generations，population=18 |
| 特征后端 | gamma-offline（图像统计 + 哈希文本 + 任务信号；模态权重 0.10/0.70/0.20） |
| 测试日期 | 2026-07-18 |

测试集使用 4/4/4/3 条分别对应 A/B/C/D，验证集使用每类 4 条独立生成夹具，均未参与训练；训练、验证、测试的图片内容、查询和 template_group 均做泄漏检查。75 条 train/test 图片来自 Wikimedia Commons 的真实照片，16 条 validation 图片为可复现的独立控制夹具；来源、作者、文件页和许可保存在 `real_fixture_sources.jsonl`。另有 12 条完全独立的 hard-negative 边界集，不参与训练。准确率分别报告裸 logits（raw）与执行在线 gate 后（gated），不把门控结果混入 raw 指标。

`python -m submission.router.evaluate` 使用显式 validation split，固定 test 只在模型冻结后评估；默认运行 seed `7,17,27`，输出 macro-F1、逐类指标、延迟和特征消融。机器可读结果见 `eval_results.json`。

## 2. Worker 选择准确率与混淆矩阵

```text
                 predicted
actual        A    B    C    D
Worker-A      4    0    0    0
Worker-B      0    4    0    0
Worker-C      0    0    4    0
Worker-D      0    0    0    3
```

Worker 选择准确率：**raw 13/15 = 86.7%，gated 15/15 = 100.0%**。

`eval_results.json`（seed=7/17/27、generations=90）的纯 logits test accuracy 为 86.7% / 100.0% / 100.0%，macro-F1 为 0.8587 / 1.0000 / 1.0000；rule-only 与 gated accuracy 均为 100%。gate change rate 为 13.33% / 0% / 0%，净收益为 +2 / 0 / 0，gated 平均成本为 1.2 个 Worker 单位。100% 是“路由头 + 查询意图门控”的系统结果，不是纯 sep-CMA-ES 路由头指标。

Clean test（seed=7）的 `gate_changes=2`，2 条 raw 错误均由明确查询意图优先规则修正，`gate_corrections=2`、`gate_regressions=0`、`gate_neutral_changes=0`。这里的 change rate 只表示门控改变了路由，不把所有改变都称为提升；训练 objective、评估和部署使用同一个 `_gate()`。

指标只覆盖公开照片分布，不等价于 Worker 任务正确率或跨门店泛化能力。生产验收需要按门店、摄像头、光照和商品类别分组留出，并报告 macro-F1、D 路由率、任务成功率、延迟和成本。

## 3. 延迟

在 CPU 上连续路由 15 条测试样本，计时范围为图像读取、特征提取、路由推断和 prompt rewrite，不包含 Worker 推理：

| 指标 | 结果 |
|---|---:|
| P50 | 8.902 ms |
| P95 | 14.453 ms |
| 最大 | 16.240 ms |
| 约束 | < 1,000 ms |

首次图像解码受文件缓存影响，最大值高于稳定态；即便以最大值计仍显著低于题目约束。

## 4. Hard-negative 典型错误与边界分析

Hard-negative 使用独立的自然 Worker gold：hn-02/08/12 为 B，hn-05/10 为 A，hn-09 为 C；hn-04 按 Worker 口径标为 B，并在 `expected_action` 中标记为“模糊请求应升级 D”。Hard-negative 复用训练照片，仅隔离 query 侧冲突，不计入 clean test 的 image_overlap。

严格 Worker 选择结果：**raw 10/12 = 83.3%，gated 11/12 = 91.7%**，`gate_changes=1`。这不是把 gate 输出当 gold，而是将最终 Worker 与独立 `label` 比较。raw 混淆矩阵如下：

```text
                 predicted
actual        A    B    C    D
Worker-A      1    1    0    0
Worker-B      0    5    0    1
Worker-C      0    0    4    0
Worker-D      0    0    0    0
```

门控后矩阵为：

```text
                 predicted
actual        A    B    C    D
Worker-A      2    0    0    0
Worker-B      0    5    0    1
Worker-C      0    0    4    0
Worker-D      0    0    0    0
```

gate 的策略指标单独计算，不等同 Worker accuracy：12 条 hard-negative 中，6 条 `follow_query`（明确意图，应服从 B/A/C）、2 条 `explicit_text_only`、1 条 `escalate_ambiguous`（hn-04）、3 条 `none`。明确意图样本的误升级率为 **0/11 = 0%**；唯一升级的模糊样本 hn-04 命中 D。clean test 中 3 条显式 D 请求的 D 决策召回为 **3/3 = 100%**；它们 raw 本来就是 D，因此不计作 upgrade。两项 gate-policy 统计均使用独立的 `expected_action`，不使用模型输出生成标签。

具体行为：hn-02/08/12 的“通用视觉”被路由到 B；hn-05/10 的商品数量请求被路由到 A；hn-09 的管理摘要请求被路由到 C；hn-06/11 的文字-only 请求被显式转 C；只有 hn-04 的模糊开放式请求升级 D。这样“冲突 + 意图明确”服从查询，“冲突 + 意图模糊”才升级 D。

逐条对照如下。`policy_hit` 只表示实现符合预先写入的 `expected_action`，不表示模型泛化能力：

| 样本 | label | expected_action | gated_pred | Worker 命中 | policy 命中 |
|---|---|---|---|---|---|
| hn-01 | B | none | B | 是 | 是 |
| hn-02 | B | follow_query | B | 是 | 是 |
| hn-03 | C | none | C | 是 | 是 |
| hn-04 | B | escalate_ambiguous | D | 否 | 是 |
| hn-05 | A | follow_query | A | 是 | 是 |
| hn-06 | C | explicit_text_only | C | 是 | 是 |
| hn-07 | B | none | B | 是 | 是 |
| hn-08 | B | follow_query | B | 是 | 是 |
| hn-09 | C | follow_query | C | 是 | 是 |
| hn-10 | A | follow_query | A | 是 | 是 |
| hn-11 | C | explicit_text_only | C | 是 | 是 |
| hn-12 | B | follow_query | B | 是 | 是 |

因此 `policy_accuracy=1.0` 验证的是 gate 实现与预先定义的 policy spec 一致，不是泛化指标；泛化与路由质量由严格 Worker accuracy（91.7%）、明确意图误升级率（0%）和显式 D 召回率（100%）体现。

压力查询 `识别价格标签并生成报告` 会升级到 D，这是有意的保守策略：A 提取视觉证据，C 生成报告，避免只调用单一 Worker 造成职责缺失。

已知边界：

* “帮我看看这家店怎么样”受货架视觉先验影响而偏向 A；高置信度使低 margin 门控失效，需要意图不明时转 D、澄清或校准。
* raw 错误主要来自照片先验与查询意图不一致；0.10/0.70/0.20 模态归一化和意图优先策略不能替代真实门店留出集。
* 默认离线代理不是 CLIP 的语义质量；切换真实编码器后必须重新拟合路由头，不能直接复用 `router_weights.npy`。

## 5. 特征消融

消融固定 seed=7、90 generations，均使用相同的 train/validation/test 切分。下表的 raw 指标只衡量学习路由头，gated 指标包含确定性业务门控。

| 特征 | raw accuracy | raw macro-F1 | gated accuracy | P95 延迟 |
|---|---:|---:|---:|---:|
| 仅文本哈希 | 80.0% | 0.7952 | 100.0% | 14.997 ms |
| 仅图像统计 | 26.7% | 0.1053 | 100.0% | 14.111 ms |
| 图像 + 文本 + 任务信号 | 86.7% | 0.8587 | 100.0% | 15.041 ms |
| 仅任务信号 | 93.3% | 0.9365 | 100.0% | 16.049 ms |

仅图像统计无法表达用户意图；多模态方案相比仅文本提升 6.7 个准确率百分点。仅任务信号在小测试集上最高，说明标签与显式巡检类型高度相关，尚不能证明图像语义带来稳定增益。真实验收需要“相同巡检类型、不同视觉域、不同正确 Worker”的跨门店对照组。

## 6. 方案与门控结论

实验后端为 γ-轻量混合变体，不是 CLIP α。切换 α 后需重新训练并单独报告；生产评估需按门店和摄像头分组，报告 hard-negative、D 召回率、平均成本和 P95 延迟。
