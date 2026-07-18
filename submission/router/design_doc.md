# Task 2 多模态路由器设计说明

## 1. 设计结论

OpenFugu 的“轻量路由头 + 外部 Worker”扩展为图文路由：冻结图像/文本特征提取器，只训练四分类线性头；置信度、复杂度和不确定性可触发 Worker-D 的 A+B 并行聚合。

复杂度升级和结构化输出参考 `step-router-v1` 的公开产品原则，不使用其内部权重或算法。

## 2. Worker 策略

| Worker | 适用请求 | 执行策略 |
|---|---|---|
| A Ostrakon-VL-8B | 商品盘点、缺货、货架陈列、零售合规 | 单 Worker；强调位置、数量、证据和不确定项 |
| B 通用 VLM | 开放域图像理解、非零售物体或场景 | 单 Worker；先描述事实再回答 |
| C 纯文本 LLM | 报告生成、数据汇总、逻辑分析 | 只消费已提取的视觉结果，避免重复看图 |
| D A+B 聚合 | 综合巡检、高风险、高置信度要求、冲突/不确定 | A 与 B 并行，比较证据后再交给聚合器 |

D 不是一个更大的模型，而是一个执行策略。这样既保留专长模型的性价比，又在关键结论上引入异构证据；当 D 的资源不可用时，`worker_pool` 可以替换为题目允许的 Mock Worker。

## 3. 特征方案评估与选择

### 3.1 候选方案与默认选择

* α：CLIP 图像 embedding + sentence embedding 拼接。跨模态语义完整，但需要额外权重和显存；通过 `MM_ROUTER_USE_CLIP=1` 启用。
* β：先用 Ostrakon 生成图片描述，再做文本 embedding。语义解释性最好，但会把一次 VLM 推理放进路由关键路径，违反“路由额外延迟 <1 秒”的风险约束，并可能把 Ostrakon 的识别偏差传给路由器。
* γ：图像统计量 + 关键词 one-hot。延迟低、无需模型，但对开放域语义泛化弱。

默认后端为 **γ-轻量混合变体**：64 维图像颜色/纹理/边缘统计、96 维确定性哈希文本特征和 16 维关键词/复杂度信号，模态权重为 0.10/0.70/0.20。该后端不是 CLIP；切换 α 后必须重新训练 `router_weights.npy`。报告指标仅对应 γ 后端。

显式的 16 维信号包括盘点、合规、OCR、环境、报告、推理、开放域、复杂度、高风险、多图、不确定性等。

### 3.2 融合与复杂度门控

设图像向量为 (v)，文本向量为 (t)，任务提示为 (s)，输入为

\[
x = \mathrm{Norm}([0.10\,\mathrm{Norm}(v);0.70\,\mathrm{Norm}(t);0.20\,\mathrm{Norm}(s)]),\quad z=W x,\quad p=\mathrm{softmax}(z).
\]

路由头只输出四个 Worker logits，不生成答案。在线决策为：先取 `argmax(p)`；若请求包含“综合/交叉验证/高风险/不能确定/多图”等门控信号，或最大概率低于 0.56、前两类概率 margin 小于 0.08，则升级到 D。该门控模仿 Step Router 的“复杂任务升级、高频任务走快路径”，并防止一个过度自信的专长模型独占风险结论。

`image_scene_hint()` 根据颜色和背景统计给出 shelf/open/report/safety 提示，再与查询意图比较。明确查询意图优先于弱视觉提示：货架图的通用描述转 B，商品数量请求转 A，已有文字结果转 C；意图模糊或要求交叉验证时升级 D。该提示不是 VLM，阈值只适用于照片夹具；真实门店需要校准的轻量 scene head。

## 4. sep-CMA-ES 训练

输入特征冻结，路由头参数量为 `4 × (176+1) = 708`。优化器遵循 OpenFugu 的 `ask → evaluate → rank → tell`：围绕均值采样候选，按适应度排序，使用前半数精英更新均值和对角协方差。对角协方差把状态从 O(n²) 降为 O(n)，适合黑盒 Worker 奖励；具备逐样本标签时，交叉熵梯度下降通常更省样本。

目标函数为：

\[
J(\theta)=\mathrm{gated\_accuracy}(\theta)-0.03\times\mathrm{cross\_entropy}(\theta)-0.005\times(1-\mathrm{raw\_accuracy}(\theta)).
\]

其中 `gated_accuracy` 是门控后的在线决策准确率，交叉熵只用于区分同准确率候选；训练目标在每个候选参数上执行与 `route()` 相同的 gate，而不是只优化裸 logits 的 argmax。`-0.005×(1-raw_accuracy)` 防止路由器把所有样本升级到 D 以换取 gated accuracy=1 的退化解。因此训练、离线评估和部署行为口径一致，评估同时保留门控前 raw accuracy。若接入真实 Worker，可将 gated accuracy 换成“任务成功率 − λ·延迟 − μ·成本”的终局奖励，路由器无需反向传播穿过外部 API。

## 5. 数据构造

`training_data.jsonl` 共 91 条：60 条训练、16 条验证、15 条测试。训练集每类 15 条；验证集每类 4 条；测试集 A/B/C 各 4 条、D 3 条。三组查询模板、图片路径和图片内容互不重叠。train/test 使用 Wikimedia Commons 许可照片，validation 使用独立控制夹具；场景覆盖货架、开放域、文档和安全通道。

1. 单一意图：盘点、开放域理解、报告汇总；
2. 复合意图：盘点 + 合规、OCR + 报告、多图汇总；
3. 风险与不确定性：高风险、模糊、要求交叉验证。

`hard_negative.jsonl` 含 12 条边界样本，不参与训练，覆盖“开放域图片 + 商品词”“货架图但只要通用描述”“混合图但只要报告”。它复用训练照片以隔离 query 冲突，不计入 clean split 的 image overlap。`label` 是独立 Worker gold；`expected_action` 是 gate-policy 标注，只用于误升级率和 D 召回。来源、作者、文件页和许可记录在 `real_fixture_sources.jsonl`。

## 6. 运行方式

```bash
python submission/router/build_fixtures.py
python submission/router/mm_router.py --mode train
python submission/router/mm_router.py --mode evaluate
python submission/router/mm_router.py --mode predict \
  --image submission/router/fixtures/a_15.png \
  --query "请盘点货架上的商品数量，并指出缺货空位。"
```

路由器不加载 Ostrakon 或任何 Worker，不执行生成；本地实验只测特征提取和线性头，因此满足额外延迟约束。真实部署时由上层 Worker Pool 根据 `worker` 和 `strategy` 调用模型。

`--mode train/evaluate` 输出 clean test 的 raw/gated 指标和 hard-negative 指标；`route()` 返回 `raw_worker`、`worker`、`gate_upgraded`、`gate_reasons` 和 `prompt_rewrite`。

## 参考

* Step Router V1：<https://platform.stepfun.com/docs/zh/guides/developer/step-router>
* 阶跃星辰视觉理解模型：<https://platform.stepfun.com/docs/zh/guides/models/vision>
* 阶跃星辰推理模型最佳实践：<https://platform.stepfun.com/docs/zh/guides/developer/reasoning>
* OpenFugu 机制分析：`submission/analysis/openfugu_mechanism.md`
