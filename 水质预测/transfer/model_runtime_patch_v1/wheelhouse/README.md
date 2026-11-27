# wheelhouse 说明

本目录为空。已检查原始隔离环境使用的 pip cache 和项目内 wheel 候选，没有发现完整、可信且匹配 Windows x64 / Python 3.11 / `torch==2.6.0+cu124` 的 PyTorch wheel；因此记录为 `TORCH_WHEEL_DOWNLOAD_REQUIRED`。

不要把原始 `.venv` 的 `site-packages` 当作 wheel 迁移。新电脑应获取精确版本的官方 PyTorch CUDA 12.4 wheel，再按 `historical_exact_versions.csv` 安装其余包，并在运行任何实验前执行独立的环境检查。
