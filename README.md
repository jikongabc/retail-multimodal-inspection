# 多模态零售巡检系统

Mock-first 的图文路由、Worker 执行、证据合成和反馈增量训练示例。

## 五步复现

推荐直接使用 Makefile：

```bash
make reproduce
```

等价的手动命令：

```bash
python -m pip install -r requirements.txt
python -m submission.router.build_fixtures
python -m submission.router.mm_router --mode train
python -m submission.pipeline.demos.run_demos
python -m unittest discover -v
```

提交前可生成交付文件 hash 清单：

```bash
make manifest
```

提交前完整验证：

```bash
make verify
```

如需重新下载 Wikimedia 真实夹具，再执行：

```bash
make fetch-fixtures
```

完整 Task 2 评估（validation、3 seed、macro-F1、消融和延迟）：

```bash
make eval
```

Task 4 环境反馈 + ReAct 数据飞轮：

```bash
make feedback
```

该命令运行“环境行动 → 反馈观察 → 结构化反思 → 自动样本生成 → sep-CMA-ES 自训练 → 固定测试门禁 → 发布/回滚”的完整闭环，并生成 `submission/innovation/training_data.jsonl`、`react_trace.jsonl` 和量化实验报告。演示会重建 Task4 本地状态；生产使用时应去掉 `--reset-demo-state`。

基础 Demo 使用确定性 Mock Worker，报告会标记 `mock_mode=true`。设置 `OSTRAKON_BASE_URL` 后，Worker-A 可切换到 OpenAI 兼容的 Ostrakon 服务；也可分别设置 `WORKER_B_BASE_URL`、`WORKER_C_BASE_URL` 接入通用 VLM 和文本 LLM，否则 B/C 自动降级为 Mock。

`submission/router/configs/workers.real.yaml` 仅描述运行拓扑，不负责环境变量插值；实际 endpoint 由 `worker_pool.py` 在运行时读取。
