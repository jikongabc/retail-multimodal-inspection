# 云端真实模型验收

该目录保存 Task 2 多模态路由、Task 3 Worker 编排和 Ostrakon-VL-8B 真实推理的端到端证据。验收运行于单卡 RTX 3090，未使用 Mock Worker。

## 验收链路

1. 在云端加载 `Ostrakon/Ostrakon-VL-8B` BF16 权重并启动本机 OpenAI 兼容服务。
2. Pipeline 读取公开许可的零售图片和商品盘点指令。
3. Task 2 路由器选择 Worker-A 与 `single_worker` 策略。
4. Worker-A 调用真实 Ostrakon 服务并返回结构化发现与合规证据。
5. 合成器生成巡检报告并执行 Schema、路径和 Mock 门禁。

## 运行环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 24GB |
| 模型 | `Ostrakon/Ostrakon-VL-8B` |
| 精度 | BF16 |
| 模型加载显存 | 16,722 MB |
| 服务地址 | `127.0.0.1:8000` |
| 图片 | `submission/router/fixtures/a_00.jpg` |
| 巡检类型 | 商品盘点 |

## 验收结果

| 项目 | 结果 |
|---|---|
| Worker | Worker-A |
| 策略 | `single_worker` |
| 路由延迟 | 54.514 ms |
| Worker 延迟 | 8,684.114 ms |
| `mock_mode` | `false` |
| 模型版本 | `Ostrakon/Ostrakon-VL-8B` |
| findings | 1 条 |
| compliance items | 2 条 |
| overall score | 75 |
| warnings | 0 条 |
| Schema 校验 | 通过 |
| 总门禁 | 9/9 通过 |

## 证据文件

| 文件 | 内容 |
|---|---|
| `input.json` | 可移植输入、巡检类型、模型和本机 endpoint |
| `output.json` | 完整结构化巡检报告、路由日志和模型版本 |
| `verification.json` | GPU 健康状态及 9 项机器门禁结果 |

`ostrakon_server.log` 是运行日志，不纳入版本控制，也不是验收必需文件。

## 验收门禁

`verification.json` 必须同时满足：

- Schema 输出存在；
- `mock_mode=false`；
- 路由日志存在；
- 最终选择 Worker-A；
- Worker 执行无错误；
- 模型版本不是 Mock 或 unknown；
- 路由延迟低于 1 秒；
- findings 或 compliance 包含结构化证据；
- 图片引用不包含云端 checkout 绝对路径。

## 复现

在仓库根目录复制并填写云端配置：

```bash
cp .env.example .env
make gpu-cloud
make init-gpu-cloud
make real-e2e-cloud
```

`make real-e2e-cloud` 会同步本地代码、启动真实服务、执行验收、关闭服务，并把本目录中的 JSON 产物拉回本地。

## 结果边界

- 该结果证明真实模型服务、路由、Worker 适配器、证据合成和 Schema 门禁可以在单卡 GPU 上贯通。
- 单张公开夹具不能证明跨门店泛化、业务准确率或生产稳定性。
- Worker 延迟包含模型生成，不应与只包含特征提取的路由延迟直接比较。
- 生产验收需要增加真实门店留出集、并发压测、超时重试、成本和按场景分组指标。
