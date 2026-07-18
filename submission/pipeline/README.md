# 零售巡检 Pipeline

Python 3.10+；依赖 `numpy`、`Pillow`。

## 运行

```bash
python -m submission.pipeline.demos.run_demos
```

路由器加载 `submission/router/router_weights.npy`，逐图执行图文路由。Worker Pool 支持 A/B/C/D，D 并行调用 A、B。中间结果写入进程内 KV Store，合成器输出结构化 JSON。结果保存在 `submission/pipeline/demos/scenario_*/output.json`。

单次请求：

```bash
python submission/pipeline/inspection_pipeline.py \
  --images submission/router/fixtures/a_00.jpg submission/router/fixtures/a_01.jpg \
  --inspection-type 商品盘点 --store-id store-demo
```

测试：

```bash
python -m unittest discover -v
```

## Worker 模式

设置 `OSTRAKON_BASE_URL` 后可使用 `--worker-mode real`。接口需要提供 OpenAI 兼容的 `/chat/completions` 路径。`OSTRAKON_MODEL` 和 `OSTRAKON_API_KEY` 为可选环境变量。Worker-B/C 可分别使用 `WORKER_B_BASE_URL`、`WORKER_C_BASE_URL`、对应的 `*_MODEL` 和 `*_API_KEY` 接入真实服务；未配置时自动使用 Mock 适配器。

```bash
export OSTRAKON_BASE_URL=http://127.0.0.1:8000/v1
export OSTRAKON_MODEL=Ostrakon/Ostrakon-VL-8B
python submission/pipeline/inspection_pipeline.py --worker-mode real \
  --images submission/router/fixtures/a_00.png --inspection-type 合规检查
```

真实服务与云端验收：

```bash
python -m submission.services.ostrakon_server \
  --model-path /path/to/Ostrakon-VL-8B --port 8000

make init-gpu-cloud
make real-e2e-cloud
```

`real-e2e-cloud` 仅监听 `127.0.0.1`，结束后关闭服务并拉回 `demos/real_cloud/output.json` 与 `verification.json`。

## 输出

- `routing_log` 同时记录 raw Worker、最终 Worker、升级原因、路由延迟和执行延迟。
- Worker 原始输出和元数据写入 `inspection:{request_id}:...` 命名空间。
- A/B 证据冲突时合成层输出 `unclear`，并在 evidence 中保留冲突文本。
- Mock 报告包含 `mock_mode=true` 和 warning，不应被当作真实模型评测结果。
- `output.json` 返回前执行 Schema 校验。
- 真实 Worker 返回非法 JSON 时保留原文，并在 `error`/`warnings` 中标记解析失败。

| Demo | 内容 |
|---|---|
| `demos/scenario_1_inventory/` | 货架盘点 |
| `demos/scenario_2_compliance/` | 高风险合规 |
| `demos/scenario_3_comprehensive/` | 综合评估、错误路由和冲突证据 |

Demo 图片为可控测试夹具，不代表真实门店采集数据。
