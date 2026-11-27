# TabICLv2 Pilot 环境检查

## 初始检查

- Python：3.11.9
- GPU：NVIDIA GeForce RTX 2070 with Max-Q Design
- NVIDIA 驱动：可用
- 项目主环境：未安装 `torch` 和 `tabicl`
- 已知候选模型便携环境：未安装 `torch` 和 `tabicl`
- 未发现可直接复用的 Conda 环境或 pip 缓存 wheel
- 没有修改项目主环境，采用独立环境：`runs/tabiclv2_pilot/环境/.venv`

## 安装记录

- `torch==2.6.0+cu124`：来自 PyTorch 官方 CUDA 12.4 wheel 源。
- `tabicl==2.2.0`：已安装。
- 同时安装了 pilot 所需的 `numpy`、`pandas`、`scikit-learn`、`xgboost` 等基础包。
- 首次下载约 2.53 GB 的 CUDA wheel 曾因外层短时限中止，但不是兼容性失败；随后在较长运行窗口内安装成功。

## 当前隔离环境验证

```text
Python: 3.11.9
torch: 2.6.0+cu124
torch.version.cuda: 12.4
torch.cuda.is_available(): True
GPU: NVIDIA GeForce RTX 2070 with Max-Q Design
tabicl: 2.2.0
from tabicl import TabICLRegressor: OK
```

## TabICLv2 合成 smoke test

- 调用：`TabICLRegressor(device="cuda", use_amp="auto", offload_mode="auto", random_state=42)`。
- 200 行、10 个数值特征；训练 160 行，预测 40 行。
- checkpoint：官方默认 `tabicl-regressor-v2-20260212.ckpt`，自动下载成功。
- 预测形状：`(40,)`。
- NaN：无。
- CUDA：实际可用，未发生 CPU fallback。
- 运行时间：约 36.6 秒（含 checkpoint 首次加载）。

## 结论

环境没有阻塞，可以进入 Chl-a surface 的正式 pilot。后续仍使用本隔离环境，不改变正式主链环境和依赖。

官方接口说明：[TabICL API](https://tabicl.readthedocs.io/en/latest/api.html)。
