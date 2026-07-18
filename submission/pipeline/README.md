# 零售巡检 Pipeline

Python 3.10+；依赖 `numpy`、`Pillow`。

## 运行

在仓库根目录执行：

```bash
python -m submission.pipeline.demos.run_demos
```

路由器加载 `submission/router/router_weights.npy`，逐图执行图文路由。Worker Pool 支持 A/B/C/D，D 并行调用 A、B。中间结果写入进程内 KV Store，合成器输出结构化 JSON。结果保存在 `submission/pipeline/demos/scenario_*/output.json`。

也可以直接运行单次请求：

```bash
python submission/pipeline/inspection_pipeline.py \
  --images submission/router/fixtures/a_00.png submission/router/fixtures/a_01.png \
  --inspection-type 商品盘点 --store-id store-demo
```

运行仓库级 smoke test：

```bash
python -m unittest discover -v
```

## Worker 模式

设置 `OSTRAKON_BASE_URL` 后可使用 `--worker-mode real`。接口需要提供 OpenAI 兼容的 `/chat/completions` 路径。`OSTRAKON_MODEL` 和 `OSTRAKON_API_KEY` 为可选环境变量。真实模式将 Worker-A 接入 Ostrakon，Worker-B、Worker-C 保持 Mock 适配器。

```bash
export OSTRAKON_BASE_URL=http://127.0.0.1:8000/v1
export OSTRAKON_MODEL=Ostrakon/Ostrakon-VL-8B
python submission/pipeline/inspection_pipeline.py --worker-mode real \
  --images submission/router/fixtures/a_00.png --inspection-type 合规检查
```

## 输出

- `routing_log` 同时记录 raw Worker、最终 Worker、升级原因、路由延迟和执行延迟。
- Worker 原始输出和元数据写入 `inspection:{request_id}:...` 命名空间。
- A/B 证据冲突时合成层输出 `unclear`，并在 evidence 中保留冲突文本。
- Mock 报告包含 `mock_mode=true` 和 warning，不应被当作真实模型评测结果。
- `output.json` 返回前执行 Schema 校验。
- 真实 Worker 返回非法 JSON 时保留原文，并在 `error`/`warnings` 中标记解析失败。

三组 Demo 的输入清单、路由日志、完整 JSON 和摘要分别位于：

- `demos/scenario_1_inventory/`
- `demos/scenario_2_compliance/`
- `demos/scenario_3_comprehensive/`

每组目录还包含 `screenshot.png`，展示路由日志、合成结果和完整输出 JSON。

Demo 图片为可控测试夹具，不代表真实门店采集数据。
