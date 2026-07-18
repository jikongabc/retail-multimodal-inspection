# 多模态零售巡检系统

Mock-first 的图文路由、Worker 执行、证据合成和反馈增量训练示例。

## 六步复现

```bash
python -m pip install -r requirements.txt
# 可选：已有真实 JPG 时可跳过；无网络时会使用合成夹具
python -m submission.router.fetch_real_fixtures
python -m submission.router.build_fixtures
python -m submission.router.mm_router --mode train
python -m submission.pipeline.demos.run_demos
python -m unittest discover -v
```

完整 Task 2 评估（validation、3 seed、macro-F1、消融和延迟）：

```bash
python -m submission.router.evaluate
```

Task 4 数据飞轮：

```bash
python -m submission.innovation.run_feedback_experiment
```

基础 Demo 使用确定性 Mock Worker，报告会标记 `mock_mode=true`。设置 `OSTRAKON_BASE_URL` 后，Worker-A 可切换到 OpenAI 兼容的 Ostrakon 服务；Worker-B/C 按资源降级策略使用 Mock。
