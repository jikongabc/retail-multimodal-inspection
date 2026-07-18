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
| 测试日期 | 2026-07-17（修订版） |

测试集使用 4/4/4/3 条分别对应 A/B/C/D，验证集使用每类 4 条独立生成夹具，均未参与训练；训练、验证、测试的图片内容、查询和 template_group 均做泄漏检查。75 条 train/test 图片来自 Wikimedia Commons 的真实照片，16 条 validation 图片为可复现的独立控制夹具；来源、作者、文件页和许可保存在 `real_fixture_sources.jsonl`。另有 12 条完全独立的 hard-negative 边界集，不参与训练。准确率分别报告裸 logits（raw）与执行在线 gate 后（gated），不把门控结果混入 raw 指标。

新增 `python -m submission.router.evaluate` 作为可重复评估入口：使用 JSONL 中显式声明的 validation split，固定 test 只在最后评估；默认运行 seed `7,17,27`，并输出 macro-F1、每类 precision/recall/F1、p50/p95 和特征消融。完整机器可读结果见 `eval_results.json`。

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

当前 `eval_results.json`（seed=7/17/27、generations=30）的纯 logits test accuracy 为 80.0% / 80.0% / 86.7%，rule-only 与 gated accuracy 均为 100%；gate change rate 为 20% / 20% / 13.33%，对应净收益为 +3 / +3 / +2（无 regressions），gated 平均成本为 1.2 个 Worker 单位。因此 100% 不能解释为纯 sep-CMA-ES 路由头能力，而是“路由头 + 查询意图门控”的系统结果。

Clean test（seed=7）的 `gate_changes=3`，3 条 raw 错误均由明确查询意图优先规则修正，`gate_corrections=3`、`gate_regressions=0`、`gate_neutral_changes=0`。这里的 change rate 只表示门控改变了路由，不把所有改变都称为提升；训练 objective、评估和部署使用同一个 `_gate()`。

这个数字只说明在公开照片分布上路由头能复现路由标签，不等价于真实 Worker 的任务正确率，也不证明跨门店泛化。真实验收仍应增加按门店、摄像头、光照和商品类别分组的留出集，并报告 macro-F1、D 路由率、任务成功率、延迟和成本。

## 3. 延迟

在 CPU 上连续路由 15 条测试样本，计时范围为图像读取、特征提取、路由推断和 prompt rewrite，不包含 Worker 推理：

| 指标 | 结果 |
|---|---:|
| 平均 | 10.015 ms |
| P95 | 14.495 ms |
| 最大 | 15.146 ms |
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

* 仅写“帮我看看这家店怎么样”时，路由偏向 A，因为“店/货架”视觉先验强但意图不明确；当前置信度仍高，低 margin 门控无法兜底。改进方向是增加“意图不明 → D/澄清”规则或校准模型。
* 真实照片的视觉外观差异较大，raw 的错误主要来自照片先验与查询意图不一致；当前已加入 0.10/0.70/0.20 的模态归一化，并让明确查询意图优先于弱视觉提示。公开照片上的 gated 结果改善了 clean 与 hard-negative，但仍不能替代真实门店留出集。
* 默认离线代理不是 CLIP 的语义质量；切换真实编码器后必须重新拟合路由头，不能直接复用 `router_weights.npy`。

## 5. 方案与门控结论

实验后端为 γ-轻量混合变体，不是 CLIP α。α 为可替换生产后端，切换后需重新训练并单独报告。后续评估应增加按门店和摄像头分组的留出集，报告 hard-negative、D 召回率、平均成本和 P95 延迟。
