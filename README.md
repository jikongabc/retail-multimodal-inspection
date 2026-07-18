# 多模态零售巡检系统

支持图文路由、Worker 编排、证据合成、真实模型调用和反馈增量训练。

## 双项目集成架构

```text
门店图片 + 巡检指令
  → 多模态特征
  → OpenFugu 范式的 sep-CMA-ES 四分类路由头
  → Ostrakon-VL-8B / 通用 VLM / 文本 LLM / A+B 并行策略
  → 证据合成与 Schema 校验
  → 环境反馈、增量训练与回归门禁
```

Task 0 独立验证官方 OpenFugu checkpoint 和服务；Task 2 复用其 `ask → evaluate → rank → tell` 黑盒优化范式，按零售图文特征和 A/B/C/D 四种策略重新定义 708 参数路由头。OpenFugu 原始 19,456 参数的 7 Worker + 3 Role 头与本任务输出空间不兼容，因此不直接加载其路由权重。Task 3 通过仓库内置 OpenAI 兼容服务调用真实 Ostrakon-VL-8B，云端证据位于 `submission/pipeline/demos/real_cloud/`。

## 复现

```bash
make reproduce
```

五步手动命令：

```bash
python -m pip install -r requirements.txt
python -m submission.router.build_fixtures
python -m submission.router.mm_router --mode train
python -m submission.pipeline.demos.run_demos
python -m unittest discover -v
```

完整验证：

```bash
make verify
```

Wikimedia 夹具：

```bash
make fetch-fixtures
```

Task 2 三种子评估、消融和延迟：

```bash
make eval
```

Task 4 数据飞轮：

```bash
make feedback
```

流程为“环境行动 → 反馈观察 → 结构化反思 → 样本生成 → sep-CMA-ES 自训练 → 固定测试门禁 → 发布或回滚”。`make feedback` 重建演示状态；生产运行不得使用 `--reset-demo-state`。

基础 Demo 使用确定性 Mock Worker，报告会标记 `mock_mode=true`。设置 `OSTRAKON_BASE_URL` 后，Worker-A 可切换到 OpenAI 兼容的 Ostrakon 服务；也可分别设置 `WORKER_B_BASE_URL`、`WORKER_C_BASE_URL` 接入通用 VLM 和文本 LLM，否则 B/C 自动降级为 Mock。

`submission/router/configs/workers.real.yaml` 仅描述拓扑；`worker_pool.py` 从环境变量读取 endpoint。

## 云端真实模型验收

将 `.env.example` 复制为 `.env` 并填写 SSH、私钥和模型目录。Makefile 通过 SSH/rsync 同步代码，本地 `.env`、虚拟环境和缓存不上传。

```bash
make init-gpu-cloud
make real-e2e-cloud
```

验收链路为“Task 2 路由 → Ostrakon Worker → 证据合成 → Schema 校验”。产物位于 `submission/pipeline/demos/real_cloud/`；门禁要求 `mock_mode=false`、真实模型版本存在、路由延迟低于 1 秒、结构化证据非空且 Worker 无错误。
