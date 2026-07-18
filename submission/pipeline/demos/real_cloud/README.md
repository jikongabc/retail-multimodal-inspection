# 云端真实模型验收

| 项目 | 结果 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 24GB |
| Worker-A | `Ostrakon/Ostrakon-VL-8B`，BF16 |
| 模型加载显存 | 16722.0 MB |
| 路由结果 | Worker-A，single_worker |
| 路由延迟 | 54.514 ms |
| Worker 延迟 | 8684.114 ms |
| Pipeline 输出 | Schema 校验通过 |
| Mock 降级 | 否 |

`input.json` 保存可移植输入，`output.json` 保存完整巡检报告，`verification.json` 保存真实模型、路由延迟、执行状态、结构化证据和路径可移植性门禁结果。

测试图片 `submission/router/fixtures/a_00.jpg` 来自项目公开许可夹具，仅证明真实工程链路可运行，不代表门店生产数据上的泛化效果。
