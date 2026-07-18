# 零售巡检 Pipeline

Task 3 的多模态执行层：接收 1–5 张门店图片和巡检类型，调用 Task 2 路由器选择 Worker，执行单模型或 A+B 并行策略，再合成通过 Schema 校验的结构化巡检报告。

> ⚠️ 默认 `mock` 模式只验证编排逻辑。报告中的 `mock_mode=true` 和 warning 必须保留，不能作为真实模型效果。

## 功能特点

- **逐图动态路由** — 每张图片独立记录 raw Worker、最终 Worker、门控原因、提示词改写和路由延迟。

- **统一 Worker 协议** — Mock、Ostrakon、通用 VLM 和文本 LLM 适配器返回同一 `WorkerResult` 结构。

- **并行聚合策略** — Worker-D 通过线程池并行调用 A 与 B，保留两个来源的结果、错误和耗时。

- **请求级状态隔离** — 中间结果写入 `inspection:{request_id}:...` 命名空间，避免不同请求混淆。

- **证据优先合成** — findings 按严重度去重，compliance 冲突转为 `unclear`，recommendations 根据风险类别生成。

- **严格输出门禁** — 输出前校验字段、枚举、时间戳、图片引用、模型版本和 Mock 标记。

- **显式失败语义** — 真实 Worker 超时或返回非法 JSON 时保留原文，并写入 `error` 和 `warnings`。

## 快速开始

从仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m submission.pipeline.demos.run_demos
```

生成结果位于 `submission/pipeline/demos/scenario_*/`。

单次请求：

```bash
python submission/pipeline/inspection_pipeline.py \
  --images submission/router/fixtures/a_00.jpg submission/router/fixtures/a_01.jpg \
  --inspection-type 商品盘点 \
  --store-id store-demo
```

支持的巡检类型：`商品盘点`、`货架盘点`、`合规检查`、`综合评估`。

## 执行流程

1. `InspectionPipeline` 校验图片数量、文件存在性、巡检类型和门店编号。
2. 根据巡检类型生成查询；多图请求追加逐图分析要求。
3. `MultimodalRouter` 对每张图片输出 Worker、策略、置信度、门控原因和改写提示词。
4. `WorkerPool` 调用对应适配器；Worker-D 并行执行 A+B。
5. Worker 原始结果、路由日志和延迟写入请求级 KV 命名空间。
6. `EvidenceSynthesizer` 合并 findings、compliance、recommendations 和 overall score。
7. `validate_report` 校验报告，通过后返回 JSON。

## Worker 模式

| Worker | 默认 Mock 能力 | 真实适配器 | 未配置行为 |
|---|---|---|---|
| Worker-A | 货架、缺货、OCR、合规 | Ostrakon OpenAI 兼容服务 | `real` 模式缺少 endpoint 时拒绝启动 |
| Worker-B | 开放域视觉 | 通用 VLM OpenAI 兼容服务 | 自动降级为 Mock |
| Worker-C | 报告与文本分析 | 文本 LLM OpenAI 兼容服务 | 自动降级为 Mock |
| Worker-D | A+B 结果聚合 | 并行调用当前 A/B 适配器 | 保留所有来源及错误 |

`submission/router/configs/workers.real.yaml` 描述服务拓扑；运行时 endpoint 和凭据由 `worker_pool.py` 从环境变量读取。

## 真实模型模式

启动 Ostrakon 服务：

```bash
python -m pip install -r requirements-gpu.txt
python -m submission.services.ostrakon_server \
  --model-path /path/to/Ostrakon-VL-8B \
  --host 127.0.0.1 \
  --port 8000
```

执行真实巡检：

```bash
export OSTRAKON_BASE_URL=http://127.0.0.1:8000/v1
export OSTRAKON_MODEL=Ostrakon/Ostrakon-VL-8B
python submission/pipeline/inspection_pipeline.py \
  --worker-mode real \
  --images submission/router/fixtures/a_00.jpg \
  --inspection-type 商品盘点 \
  --store-id real-store
```

可选 Worker-B/C：

| 环境变量 | 用途 |
|---|---|
| `WORKER_B_BASE_URL`、`WORKER_B_MODEL`、`WORKER_B_API_KEY` | 通用 VLM |
| `WORKER_C_BASE_URL`、`WORKER_C_MODEL`、`WORKER_C_API_KEY` | 文本 LLM |
| `OSTRAKON_API_KEY` | Ostrakon 服务鉴权 |

## 输入约束

| 字段 | 约束 |
|---|---|
| `image_paths` | 1–5 个存在的本地图片路径 |
| `inspection_type` | 商品盘点、货架盘点、合规检查或综合评估 |
| `store_id` | 非空字符串 |
| `request_id` | 可选；缺省时自动生成 |
| `worker_mode` | `mock` 或 `real` |

## 输出契约

| 字段 | 含义 |
|---|---|
| `request_id` | 请求唯一标识 |
| `inspection_time` | 带时区的 ISO 8601 时间 |
| `store_id`、`inspection_type` | 请求上下文 |
| `overall_score` | 根据发现严重度和合规失败计算的 0–100 分 |
| `findings` | 类别、严重度、描述和图片证据引用 |
| `compliance_items` | `pass`、`fail` 或 `unclear` 及其证据 |
| `recommendations` | 与风险和发现匹配的整改建议 |
| `routing_log` | raw/final Worker、策略、门控、模型、路由与执行延迟 |
| `model_versions` | 本次请求实际使用的模型版本 |
| `mock_mode` | 是否包含 Mock Worker 结果 |
| `warnings` | 降级、解析失败和其他非致命异常 |

Worker-A/B 对同一合规项给出 `pass` 与 `fail` 时，合成层必须输出 `unclear`，并在 evidence 中保留冲突文本。

## Demo

| 目录 | 场景 | 验证重点 |
|---|---|---|
| `demos/scenario_1_inventory/` | 货架盘点 | 多图输入、库存发现和补货建议 |
| `demos/scenario_2_compliance/` | 高风险合规 | 安全风险、失败项和整改建议 |
| `demos/scenario_3_comprehensive/` | 综合评估 | 路由升级、A+B 并行和冲突保留 |
| `demos/real_cloud/` | 真实 Ostrakon | 非 Mock Worker、模型版本、结构化证据和延迟门禁 |

每个 Mock Demo 包含输入、输出、路由日志、摘要和测试图片。图片是可控工程夹具，不代表真实门店采集数据。

## 测试

```bash
python -m unittest submission.pipeline.tests.test_pipeline -v
python -m unittest tests.test_real_service -v
python -m unittest discover -v
```

Pipeline 测试覆盖 1/5 张图片边界、所有 Worker 策略、D 并行、证据冲突、Schema、时间戳、KV Store、解析失败和真实服务协议。

## 已知限制

- `InMemoryKVStore` 不跨进程持久化，生产环境需要替换为外部存储。
- Mock Worker 依赖确定性场景 profile，只用于编排回归测试。
- Worker-B/C 的真实 endpoint 由部署方提供，仓库不包含对应模型服务。
- Pipeline 当前按图片逐次路由；大批量门店任务需要增加队列、背压、重试和请求级超时预算。
- 合成器基于结构化规则，不替代人工对高风险现场结论的复核。

## 故障排查

| 现象 | 处理 |
|---|---|
| 图片数量错误 | 保证每次请求输入 1–5 张图片 |
| 找不到路由权重 | 在仓库根目录执行 `make train` |
| real 模式启动失败 | 设置 `OSTRAKON_BASE_URL` 并检查 `/health` |
| 输出含 `mock_mode=true` | 检查 Worker-B/C 是否降级，或确认当前是否故意使用 Mock |
| 输出缺少结构化证据 | 查看 Worker 的 `raw_text`、`error` 和 `warnings` |
| D 结果不完整 | 检查 `metadata.source_results` 中 A/B 的独立错误 |
