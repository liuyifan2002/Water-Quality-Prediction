# Linux安装方案（尚未跨平台实测）

Python3.11.9及可用NVIDIA CUDA GPU，源实验在Windows完成；Linux运行仅可在同版本依赖和预测审查后评价，不预先保证数值等价。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r 06_environment/requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python 05_scripts/run_preflight.py --dependencies
```

torch的Linux NVIDIA运行库由cu124发行包按平台解析；源Windows锁表未虚构Linux专属包版本。不可安装不同torch/tabicl版本后绕过预检。所有入口使用pathlib包根路径，不依赖盘符、PowerShell、msvcrt或旧隔离patch。
