# 多模态零售巡检系统

一个面向零售巡检垂直场景的多模态编排系统：根据门店图片与巡检指令选择最合适的视觉或文本 Worker，执行单模型或并行协作，合成带证据的结构化报告，并通过环境反馈持续训练路由器。

> 工程主线：以 OpenFugu 的黑盒路由优化范式训练轻量图文路由头，以 Ostrakon-VL-8B 承担零售视觉理解，再通过 Worker Pool、证据合成、Schema 门禁和 ReAct 数据飞轮形成可复现的端到端系统。

> ⚠️ 默认 Demo 使用确定性 Mock Worker，输出会标记 `mock_mode=true`。真实模型结论只能来自配置了 OpenAI 兼容服务的 `real` 模式；公开夹具和小规模评测不能替代真实门店泛化验证。

## 功能特点

- **图文联合路由** — 组合图像统计、哈希文本特征和任务信号，以 708 参数四分类路由头选择 Worker-A/B/C/D。

- **黑盒路由优化** — 使用 sep-CMA-ES 的 `ask → evaluate → rank → tell` 流程训练路由头，不需要对外部 Worker 反向传播。

- **异构 Worker 编排** — Worker-A 处理零售视觉，Worker-B 处理开放域视觉，Worker-C 处理纯文本汇总，Worker-D 并行调用 A+B 并聚合证据。

- **风险与不确定性门控** — 低置信度、低 margin、高风险、复杂请求和多图任务可升级到 Worker-D；明确查询意图优先于弱视觉先验。

- **结构化证据合成** — Pipeline 支持 1–5 张图片，统一 findings、compliance、recommendations、routing log 和模型版本，并在返回前执行 Schema 校验。

- **冲突保真** — 不同 Worker 对同一合规项给出相反证据时，系统输出 `unclear` 并保留来源，不用静默投票掩盖冲突。

- **真实模型服务** — 内置 Ostrakon-VL-8B 的 OpenAI 兼容服务，可在单卡 GPU 上完成 Task 2 路由到 Task 3 巡检报告的真实端到端验收。

- **反馈驱动的数据飞轮** — 环境金标触发 Observe、Reflect、Act、增量样本生成、replay 训练、固定测试门禁、版本发布或回滚。

- **本地开发、云端推理** — 代码保留在本地，通过 SSH 和 rsync 同步到 GPU 主机运行；密钥、虚拟环境、缓存和日志不上传。

- **可复现交付** — Makefile 统一管理数据校验、训练、评估、Demo、反馈实验、云端验收、测试和 SHA-256 清单。

## 快速开始

环境要求：Python 3.10+。本地路由与 Mock Pipeline 只依赖 NumPy 和 Pillow，不需要 GPU。

```bash
git clone https://github.com/jikongabc/retail-multimodal-inspection.git
cd retail-multimodal-inspection
python -m venv .venv
source .venv/bin/activate
make reproduce
```

`make reproduce` 会安装依赖、构建可复现夹具、训练路由头、生成三个 Pipeline Demo 并运行测试。

完整交付验收：

```bash
make verify
```

该命令依次执行数据泄漏检查、快速三随机种子评估、Demo 重建、反馈飞轮实验、单元测试和交付清单生成。

## 使用流程

### 训练与评估路由器

```bash
make validate-data
make train
make eval
```

`make eval` 使用 seed `7,17,27`、90 generations，输出 clean test、hard-negative、门控、成本、延迟和特征消融指标。机器可读结果位于 `submission/router/eval_results.json`。

### 运行巡检 Pipeline

```bash
python submission/pipeline/inspection_pipeline.py \
  --images submission/router/fixtures/a_00.jpg \
  --inspection-type 商品盘点 \
  --store-id store-demo
```

批量生成三个可复现 Demo：

```bash
make demos
```

### 运行反馈飞轮

```bash
make feedback
```

该目标会重建演示状态，适合复现实验。生产运行应直接调用 `submission.innovation.run_feedback_experiment`，使用独立持久化目录且不传 `--reset-demo-state`。

### 运行真实 Ostrakon Worker

安装 GPU 依赖并启动本地 OpenAI 兼容服务：

```bash
python -m pip install -r requirements-gpu.txt
python -m submission.services.ostrakon_server \
  --model-path /path/to/Ostrakon-VL-8B \
  --port 8000
```

在另一个终端执行：

```bash
export OSTRAKON_BASE_URL=http://127.0.0.1:8000/v1
export OSTRAKON_MODEL=Ostrakon/Ostrakon-VL-8B
python submission/pipeline/inspection_pipeline.py \
  --worker-mode real \
  --images submission/router/fixtures/a_00.jpg \
  --inspection-type 商品盘点
```

## 任务交付

| 任务 | 交付内容 | 入口 |
|---|---|---|
| Task 0 | OpenFugu 环境、自测、真实双 Worker API；Ostrakon 五场景推理与性能基线 | [`submission/env_report.md`](submission/env_report.md) |
| Task 1 | OpenFugu 路由机制与 Ostrakon 微调体系分析 | [`submission/analysis/`](submission/analysis/) |
| Task 2 | 图文特征、sep-CMA-ES 路由、门控、数据校验、三种子评估与消融 | [`submission/router/design_doc.md`](submission/router/design_doc.md) |
| Task 3 | Worker Pool、证据合成、Schema 校验、Mock/真实模式和三个 Demo | [`submission/pipeline/README.md`](submission/pipeline/README.md) |
| Task 4 | 环境反馈、ReAct 状态循环、增量训练、模型注册、门禁与回滚 | [`submission/innovation/design_doc.md`](submission/innovation/design_doc.md) |

## 技术栈

- **路由与优化** — Python、NumPy、Pillow、sep-CMA-ES、可选 CLIP / Sentence Transformers 特征后端。
- **真实视觉模型** — PyTorch 2.8、Transformers 4.57、Accelerate、Ostrakon-VL-8B、BF16。
- **服务协议** — Python 标准库 HTTP Server，提供 OpenAI 兼容 `/v1/chat/completions` 和健康检查。
- **编排与存储** — Worker Pool、线程池并发、进程内 KV Store、JSON Schema 风格校验、JSON/JSONL 审计产物。
- **工程工具** — Make、unittest、Ruff、SSH、rsync、SHA-256 manifest。

## 系统架构

| 层级 | 组件 | 职责 |
|---|---|---|
| 输入层 | `InspectionPipeline` | 校验 1–5 张图片、巡检类型、门店和请求编号 |
| 路由层 | `MultimodalFeatureExtractor`、`MultimodalRouter` | 提取图文特征、输出 raw Worker、执行门控并改写 Worker 提示词 |
| 执行层 | `WorkerPool` | 调度 A/B/C；D 使用线程池并行调用 A+B |
| 服务层 | `ostrakon_server` | 加载 Ostrakon-VL-8B，解析 OpenAI 多模态消息并返回结构化结果 |
| 状态层 | `InMemoryKVStore` | 保存请求级 Worker 原始输出、延迟和元数据 |
| 合成层 | `EvidenceSynthesizer` | 合并发现与合规证据、处理冲突、计算评分并生成建议 |
| 校验层 | `schemas.py` | 检查字段、枚举、时间戳、证据和 Mock 标记 |
| 学习层 | `ReActFlywheel`、`IncrementalTrainer` | 消费可信环境反馈、生成样本、训练候选并执行回归门禁 |
| 注册层 | `ModelRegistry` | 记录父子版本、激活状态、发布原因和回滚 |

一次请求的执行顺序：输入校验、逐图路由、Worker 执行、原始结果落 KV、证据合成、Schema 校验、结构化报告返回。只有可信环境金标或独立验证器反馈可以自动进入训练；模型自评只能进入审核队列。

## Worker 策略

| Worker | 能力 | 适用请求 | 执行方式 |
|---|---|---|---|
| Worker-A | Ostrakon-VL-8B 零售视觉 | 盘点、缺货、陈列、OCR、零售合规 | 单 Worker |
| Worker-B | 通用 VLM | 开放域图片、非零售物体与场景理解 | 单 Worker |
| Worker-C | 文本 LLM | 已有视觉结果的报告、汇总和逻辑分析 | 单 Worker，不重复看图 |
| Worker-D | A+B 聚合策略 | 高风险、复杂、多图、冲突和不确定任务 | A/B 并行后合成 |

Worker-D 是执行策略，不是额外模型。真实模式只强制要求 Worker-A 的 Ostrakon endpoint；Worker-B/C 未配置真实 endpoint 时会显式降级为 Mock，并在报告中保留标记。

## 路由器与评测

默认 `gamma-offline` 特征由 64 维图像统计、96 维哈希文本和 16 维任务信号组成，融合权重为 0.10/0.70/0.20。四分类线性头参数量为 `4 × (176 + 1) = 708`。设置 `MM_ROUTER_USE_CLIP=1` 可切换实验特征后端，但必须重新训练权重并单独评估。

| 指标 | 结果 |
|---|---:|
| 数据切分 | 60 train / 16 validation / 15 test |
| Seed 7/17/27 raw test accuracy | 86.7% / 100.0% / 100.0% |
| Seed 7/17/27 gated test accuracy | 100.0% / 100.0% / 100.0% |
| Seed 7 gate net benefit | +2 |
| Hard-negative raw / gated accuracy | 83.3% / 91.7% |
| 明确意图误升级率 | 0% |
| 显式 D 请求召回率 | 100% |
| CPU 路由延迟 P50 / P95 | 8.902 ms / 14.453 ms |

100% gated accuracy 是小规模公开夹具上的“路由头 + 确定性门控”系统结果，不是纯模型泛化结论。完整口径、混淆矩阵、消融与错误分析见 [`submission/router/eval_results.md`](submission/router/eval_results.md)。

## 数据飞轮

反馈闭环使用确定性的 `Observe → Reflect → Act → Observe` 状态循环：

1. 同时记录裸路由和门控后路由，与环境金标比较并计算奖励。
2. 区分正确、门控救回、模型错误、策略错误和低可信反馈。
3. 对反馈执行去重、冲突隔离、来源校验、置信度门槛和人工审核。
4. 将已批准记录物化为增量 JSONL，不修改固定测试集和主训练文件。
5. 对比 `feedback_only` 与 `replay_plus_feedback`，只允许 replay 候选进入发布门禁。
6. 固定测试 macro-F1、逐类 recall 和环境挑战指标全部通过后注册并热加载，否则保留基线或回滚。

当前演示从 12 条 hard-negative 环境样本中产生 2 条可信训练信号；`replay_plus_feedback` 将挑战集 gated accuracy 从 91.67% 提升到 100%，固定测试 gated macro-F1 保持 1.0。完整结果见 [`submission/innovation/experiment_results.md`](submission/innovation/experiment_results.md)。

## 云端真实模型验收

复制配置模板并填写 SSH 主机、端口、私钥和模型目录：

```bash
cp .env.example .env
make gpu-cloud
make init-gpu-cloud
make real-e2e-cloud
```

Makefile 会把本地工作区同步到 GPU 主机，启动 Ostrakon 服务，执行 Task 2 路由、Task 3 Worker 调用、证据合成和门禁校验，再把验收产物拉回本地。

| 项目 | 已验证结果 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 24GB |
| 模型 | `Ostrakon/Ostrakon-VL-8B`，BF16 |
| 模型加载显存 | 16,722 MB |
| 路由结果 | Worker-A，`single_worker` |
| 路由延迟 | 54.514 ms |
| Worker 延迟 | 8,684.114 ms |
| Schema | 通过 |
| Mock 降级 | 否 |

真实验收的输入、输出和逐项门禁见 [`submission/pipeline/demos/real_cloud/README.md`](submission/pipeline/demos/real_cloud/README.md)。

## 项目结构

| 路径 | 内容 |
|---|---|
| `submission/analysis/` | OpenFugu 与 Ostrakon 机制分析 |
| `submission/router/` | 数据、特征、路由器、权重、评估和配置 |
| `submission/pipeline/` | Pipeline、Worker Pool、Schema、合成器和 Demo |
| `submission/services/` | Ostrakon OpenAI 兼容服务 |
| `submission/innovation/` | ReAct 数据飞轮、反馈仓库、增量训练和模型注册 |
| `submission/screenshots/` | Task 0 原始运行证据索引 |
| `scripts/` | Task 0 基线、云端真实 E2E 和 manifest 工具 |
| `tests/` | 跨模块 Smoke 与真实服务协议测试 |
| `docs/` | 作业要求、参考材料和项目计划 |
| `Makefile` | 本地、云端和交付验收入口 |

## 配置

### 真实 Worker

| 环境变量 | 用途 | 默认值 |
|---|---|---|
| `OSTRAKON_BASE_URL` | Worker-A OpenAI 兼容 endpoint | 无；real 模式必填 |
| `OSTRAKON_MODEL` | Worker-A 模型名 | `Ostrakon/Ostrakon-VL-8B` |
| `OSTRAKON_API_KEY` | Worker-A API Key | 空 |
| `WORKER_B_BASE_URL` | Worker-B 通用 VLM endpoint | 未配置时 Mock |
| `WORKER_B_MODEL` | Worker-B 模型名 | `generic-vlm` |
| `WORKER_B_API_KEY` | Worker-B API Key | 空 |
| `WORKER_C_BASE_URL` | Worker-C 文本 LLM endpoint | 未配置时 Mock |
| `WORKER_C_MODEL` | Worker-C 模型名 | `text-llm` |
| `WORKER_C_API_KEY` | Worker-C API Key | 空 |
| `MM_ROUTER_USE_CLIP` | 启用可选 CLIP 特征后端 | `0` |

### 云端执行

`.env.example` 定义 `REMOTE_HOST`、`REMOTE_PORT`、`REMOTE_BASE`、`REMOTE_DIR`、`REMOTE_PYTHON`、`REMOTE_MODEL_PATH`、`REMOTE_SSH_KEY` 和 `OSTRAKON_PORT`。`.env` 只供本机使用，不纳入版本控制。

## 开发与测试

| 命令 | 作用 |
|---|---|
| `make fixtures` | 构建确定性控制夹具 |
| `make fetch-fixtures` | 下载 Wikimedia 公开照片并重建数据引用 |
| `make validate-data` | 检查字段、切分、模板、查询和图片哈希泄漏 |
| `make train` | 训练 Task 2 路由权重 |
| `make eval` | 完整三随机种子评估与消融 |
| `make eval-fast` | 30 generations 快速评估，不覆盖正式结果 |
| `make demos` | 重建三个 Mock Pipeline Demo |
| `make feedback` | 重建 Task 4 数据飞轮实验 |
| `make test` | 运行全部 unittest |
| `make quality` | 运行 Ruff 和空白符检查 |
| `make verify` | 执行完整交付验收 |
| `make manifest` | 生成 `submission/MANIFEST.sha256` |
| `make test-cloud` | 同步代码并在云端运行测试 |
| `make real-e2e-cloud` | 在云端运行真实 Ostrakon E2E |

## 已知限制

- 91 条路由数据和 12 条 hard-negative 只能验证工程闭环，不能代表跨门店、摄像头、光照和商品域的生产泛化。
- 默认图像统计特征不是语义视觉编码器；切换 CLIP 后需要重新训练和重新报告指标。
- Mock Demo 只验证路由、并发、合成、Schema 和错误处理，不是模型能力证据。
- Ostrakon 在密集规则网格计数中仍可能漏计；Task 0 五场景基线有 1 个计数场景存在偏差。
- 当前 KV Store 为进程内实现，生产部署需要替换为持久化并支持并发控制的存储。
- Worker-B/C 的真实服务不是仓库必需依赖，未配置时会降级为 Mock。
- 数据飞轮演示只有 2 条增量样本；生产触发阈值、时间窗口、分层采样、标注一致性和 canary 监控仍需按业务校准。

## 故障排查

| 现象 | 处理 |
|---|---|
| `make` 提示没有目标 | 先进入仓库根目录，再运行 `make help` |
| 数据校验报告 overlap | 检查 `template_group`、查询文本、图片路径和图片哈希，禁止跨 split 复用 |
| real 模式提示缺少 endpoint | 设置 `OSTRAKON_BASE_URL`，并确认服务的 `/health` 和 `/v1/chat/completions` 可访问 |
| Worker 返回非法 JSON | 查看报告中的 `error`、`warnings` 和保留的 `raw_text` |
| 云端 SSH 反复要求密码 | 配置 `ssh-copy-id`，并在 `.env` 设置 `REMOTE_SSH_KEY` |
| GPU 显存不足 | 关闭其他 GPU 进程，确认使用单个 Ostrakon 服务，并将模型目录放在数据盘 |
| 云端同步失败 | 先运行 `make gpu-cloud` 验证 SSH，再检查 `REMOTE_HOST`、`REMOTE_PORT` 和私钥权限 |
| 修改路由特征后结果异常 | 删除旧假设，重新执行 `make train && make eval`，不要复用不兼容权重 |
