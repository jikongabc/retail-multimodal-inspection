# Task 0 环境与基线验证报告

> 测试日期：2026-07-15（Asia/Shanghai）  
> 执行环境：AutoDL 内蒙 B 区，RTX 3090 单卡实例

## 1. 硬件与运行环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| GPU 数量 | 1 |
| 显存 | 24,576 MiB |
| NVIDIA 驱动 | 580.105.08 |
| `nvidia-smi` CUDA 上限 | 13.0 |
| CPU | Intel Xeon Gold 6330，分配 14 核 |
| 内存 | 90 GB |
| 系统盘 | 30 GB |
| 数据盘 | 50 GB（模型、缓存与日志均放在 `/root/autodl-tmp`） |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA Runtime | 12.8 |
| Transformers | 4.57.6 |
| Accelerate | 1.14.0 |
| LiteLLM | 1.92.0 |
| Pillow | 11.3.0 |

说明：`nvidia-smi` 的 CUDA 13.0 表示驱动可支持的最高 CUDA 版本；当前 PyTorch wheel 实际使用 CUDA 12.8。二者不冲突。

## 2. OpenFugu 验证

### 2.1 版本与权重

| 项目 | 值 |
|---|---|
| OpenFugu commit | `7ad7ccf977c1b5f38bbd07ba33d86fe655c17be8` |
| 路由基座 | Qwen3-0.6B |
| 路由向量 | `artifacts/model_iter_60.npy`，shape `(19456,)` |
| 路由向量 SHA256 | `307bce5df3317e461d9a56d099f2e9a249fee7e56555d9951968ca8e2b680314` |
| 实际 Worker | `deepseek/deepseek-chat`、`zai/glm-4.5-flash` |
| 服务端口 | 6008 |
| 最大协作轮次 | 2（响应中的 `fugu_turns` 含调度计数，可能显示为 2 或 3） |

Worker 使用 DeepSeek Chat、GLM-4.5-Flash，服务端口为 6008。路由器、双槽位调度和 OpenAI 兼容接口保持不变。密钥通过环境变量传入，未写入代码、日志或仓库。

### 2.2 自测结果

```text
self-test on 37 cases:
  agent 36/37 = 97%   (baseline 51%)
  role  37/37 = 100%  (baseline 49%)
  PASS — implementation faithful to checkpoint
```

OpenFugu checkpoint 与实现匹配，自测通过。证据：`screenshots/openfugu_selftest.png`。

### 2.3 真实 API 端到端延迟

测试方法：依次向 `/v1/chat/completions` 发送代码、数学和巡检请求；使用 curl `time_total` 测量客户端端到端时间。时间包含本地路由、远程 Worker 网络请求和多轮协作，不等同于纯 GPU `model.generate` 时间。

| 查询类型 | Worker 模式 | 端到端延迟（ms） |
|---|---|---:|
| 代码生成 | DeepSeek + GLM 真实双 Worker | 21,428.019 |
| 数学推理 | DeepSeek + GLM 真实双 Worker | 10,561.326 |
| 零售巡检分析 | DeepSeek + GLM 真实双 Worker | 25,860.765 |
| **算术平均** |  | **19,283.370** |

三类请求均返回 `model=fugu`、非空真实答案和协作轮次。补充证据为 `screenshots/openfugu_server_real.png` 与 `screenshots/openfugu_3queries_real.png`。

## 3. Ostrakon-VL-8B 五场景验证

### 3.1 推理配置与测量口径

| 项目 | 值 |
|---|---|
| 模型 | `Ostrakon/Ostrakon-VL-8B` |
| 本地体积 | 约 17 GB，4 个 safetensors 分片 |
| 精度 | `torch.bfloat16` |
| 设备分配 | `device_map="auto"`，实际全部位于 GPU 0 |
| 模型加载显存 | 16,722.0 MB |
| 输入尺寸 | 512 × 512 |
| 最大生成长度 | 160 tokens |
| 重复次数 | 每场景 3 次 |
| 延迟口径 | 仅计时 `model.generate`，CUDA 前后同步 |
| 显存口径 | `torch.cuda.max_memory_allocated()` 峰值 |

五张输入图由 `scripts/generate_task0_images.py` 生成，属于带有确定标注的可控测试夹具，不代表真实门店照片。真实门店泛化能力需使用独立数据集评估。

### 3.2 推理结果与性能

| 场景 | 三次延迟（ms） | 平均延迟（ms） | 峰值显存（MB） | 输出摘要 | 核验 |
|---|---|---:|---:|---|---|
| 货架盘点 | 1930.0 / 629.7 / 626.1 | 1,061.9 | 16,820.8 | 每层 7 个，共 21 个 | 正确 |
| 合规检查 | 1573.7 / 1591.5 / 1591.3 | 1,585.5 | 16,821.4 | 高风险；纸箱遮挡消防出口，要求立即移除 | 正确 |
| 文字提取 | 1441.0 / 1460.2 / 1471.0 | 1,457.4 | 16,818.9 | 正确提取 4 组商品、价格和促销文字 | 正确 |
| 环境评估 | 1050.1 / 1021.9 / 1054.7 | 1,042.2 | 16,818.9 | 识别湿滑地面和纸箱堵塞通道 | 正确 |
| 缺货识别 | 1001.8 / 1002.8 / 1021.1 | 1,008.6 | 16,821.4 | 输出有货 3、空位 15、总位 18、缺货率 83.3% | 计数偏差 |
| **五场景平均/最大值** |  | **1,231.1** | **16,821.4** |  | 4/5 场景结论符合夹具标注 |

缺货测试图的真实标注为：有货位 4、空货位 17、总货位 21、缺货率约 80.95%。模型漏计 3 个货位并漏计 1 个有货位，说明其对密集规则网格的精确计数仍有限制。该错误予以保留，不对结果进行人工修饰。

货架盘点首轮 1930.0 ms 明显高于后两轮，反映首次生成的缓存预热开销；报告保留首轮并纳入平均值，避免选择性剔除数据。

完整证据见：

- `screenshots/ostrakon_5scenes.png`
- 云端 `/root/autodl-tmp/ostrakon_task0/ostrakon_5scenes_results.json`
- 云端 `/root/autodl-tmp/ostrakon_task0/ostrakon_benchmark.log`

## 4. 两模型延迟汇总

| 模型/系统 | 测量范围 | 平均延迟（ms） | 峰值显存（MB） | 说明 |
|---|---|---:|---:|---|
| OpenFugu + DeepSeek/GLM | 三类真实 API 端到端请求 | 19,283.370 | 未单独隔离测量 | 包含网络、远程 API 和多轮协作 |
| Ostrakon-VL-8B | 五场景、本地 BF16、每场景 3 次 | 1,231.1 | 16,821.4 | 仅计 `model.generate` |

两项延迟的测量边界不同，不能据此直接判断模型本体快慢；该表用于报告实际 Task 0 工作流体验。

## 5. Task 0 交付核对

- [x] `screenshots/openfugu_selftest.png`：终端中包含 97%、100% 和 PASS。
- [x] `screenshots/ostrakon_5scenes.png`：包含五场景输出、平均延迟和显存。
- [x] `env_report.md`：即本文件。
- [x] 补充证据：`screenshots/openfugu_server_real.png`。
- [x] 补充证据：`screenshots/openfugu_3queries_real.png`。

截图文件位于 `submission/screenshots/`。
