# Task 0 运行证据

该目录保存 OpenFugu 与 Ostrakon-VL-8B 基线验证的原始截图。完整环境、参数、延迟口径和结果分析见 [`../env_report.md`](../env_report.md)。

## 证据索引

| 文件 | 验证对象 | 证明内容 |
|---|---|---|
| `openfugu_selftest.png` | OpenFugu checkpoint | 37 条自测；模型选择 36/37、角色选择 37/37 |
| `openfugu_server_real.png` | OpenFugu 服务 | 真实 API 服务启动、模型和协作信息 |
| `openfugu_3queries_real.png` | OpenFugu 双 Worker | 代码、数学、零售巡检三类真实请求均返回非空结果 |
| `ostrakon_5scenes.png` | Ostrakon-VL-8B | 货架盘点、合规、OCR、环境、缺货五场景输出、延迟和显存 |

## 结果摘要

| 系统 | 结果 |
|---|---|
| OpenFugu 自测 | Agent 97%，Role 100%，通过 checkpoint 一致性验证 |
| OpenFugu 真实 API | 三类请求平均端到端延迟 19,283.370 ms |
| Ostrakon 五场景 | 4/5 场景结论符合夹具标注 |
| Ostrakon 推理 | 五场景平均 `model.generate` 延迟 1,231.1 ms |
| Ostrakon 峰值显存 | 16,821.4 MB |

## 证据边界

- OpenFugu 延迟包含本地路由、远程 Worker 网络请求和多轮协作。
- Ostrakon 延迟只统计同步后的 `model.generate`，两组数字不能直接比较。
- Ostrakon 输入是 512×512 可控夹具；密集缺货网格存在漏计，不代表真实门店准确率。
- 截图不包含 API Key、SSH 密码或其他凭据。
