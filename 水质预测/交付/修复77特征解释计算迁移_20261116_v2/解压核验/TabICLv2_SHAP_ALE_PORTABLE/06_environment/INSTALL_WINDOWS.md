# Windows安装（源环境为Python venv，不是Conda）

使用Python 3.11.9 x64，NVIDIA CUDA GPU。源机器8GB RTX2070 Max-Q，torch内置cu124运行时；目的端驱动需支持该运行时。没有在本轮运行nvidia-smi或模型加载。

在解压包根目录PowerShell执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r 06_environment/requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe 05_scripts/run_preflight.py --dependencies
```

`py -3.11`实际选择版本必须是3.11.9，预检会核对。后续命令也使用该.venv的python，不要误用系统环境。checkpoint已打包，禁止自动下载新权重。未打包wheelhouse：CUDA轮子较大且Linux依赖与Windows不同，采用联网同版本安装。锁文件是源环境已安装依赖闭包，不表示已在一台干净机器上验证网络安装；若同版本不可获得，记录错误停止，不擅自升级。

不需要Earth Engine、CMEMS、模型平台账号、令牌或本机临时目录。源Conda环境不存在，因此不虚构environment.yml。
