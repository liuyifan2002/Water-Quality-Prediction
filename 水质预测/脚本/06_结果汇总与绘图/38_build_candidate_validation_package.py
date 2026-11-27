"""构建候选表格模型稳健性验证的异机便携运行包。

验证范围限定为 Chl-a 三水层、透明度三水层和 TN 底层。包内复用已
核验过的 108 列建模数据、TabPFN 本地权重及 GPU 环境安装脚本；它只
输出到扩展实验目录，不会覆盖正式 full_station 结果。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE = PROJECT_ROOT / "结果/扩展实验/表格预训练算法对比_20260908/portable_package"
SOURCE_RUNNER = PROJECT_ROOT / "脚本/04_模型训练/31_run_tabular_pretrained_ab.py"
SOURCE_VALIDATOR = PROJECT_ROOT / "脚本/04_模型训练/32_validate_candidate_tabular_models.py"
SOURCE_VALIDATION_CONFIG = PROJECT_ROOT / "配置/候选模型稳健性验证配置.json"

EXPERIMENT_ROOT = PROJECT_ROOT / "结果/扩展实验/候选模型稳健性验证_20260914"
PACKAGE_DIR = EXPERIMENT_ROOT / "portable_package"
ZIP_PATH = EXPERIMENT_ROOT / "候选模型稳健性验证_便携运行包_20260914.zip"
ARCHIVE_DIR = EXPERIMENT_ROOT / "历史运行包"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8-sig")


def archive_existing(path: Path, label: str) -> None:
    if not path.exists():
        return
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = ARCHIVE_DIR / f"{label}_{stamp}{path.suffix}"
    shutil.move(str(path), str(target))


def check_sources() -> None:
    required = [
        SOURCE_PACKAGE / "data/model_table.csv",
        SOURCE_PACKAGE / "config.json",
        SOURCE_PACKAGE / "models/tabpfn-v2-regressor.ckpt",
        SOURCE_PACKAGE / "requirements.txt",
        SOURCE_PACKAGE / "setup_nvidia.bat",
        SOURCE_PACKAGE / "setup_cpu.bat",
        SOURCE_RUNNER,
        SOURCE_VALIDATOR,
        SOURCE_VALIDATION_CONFIG,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少打包来源文件：\n" + "\n".join(missing))


def make_bat_files() -> dict[str, str]:
    common = r"""
@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 请先运行 setup_nvidia.bat。TabM 和 TabPFN 的完整验证需要 NVIDIA GPU。
  pause
  exit /b 1
)
"""
    return {
        "01_通路检查.bat": common
        + r"""
.venv\Scripts\python.exe run_candidate_validation.py --modes random_seeds --random-seeds 42 --tasks chla_surface secchi_middle tn_bottom --output-dir outputs_smoke
pause
""",
        "02_三随机种子验证.bat": common
        + r"""
.venv\Scripts\python.exe run_candidate_validation.py --modes random_seeds --output-dir outputs_random_seeds
pause
""",
        "03_年份留出验证.bat": common
        + r"""
.venv\Scripts\python.exe run_candidate_validation.py --modes time_holdout --output-dir outputs_time_holdout
pause
""",
        "04_五折留站点验证.bat": common
        + r"""
.venv\Scripts\python.exe run_candidate_validation.py --modes station_5fold --output-dir outputs_station_5fold
pause
""",
        "05_打包回传结果.bat": r"""
@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$items = @('outputs_random_seeds','outputs_time_holdout','outputs_station_5fold') | Where-Object { Test-Path $_ }; if ($items.Count -eq 0) { throw '未找到正式验证输出目录。请先完成至少一项正式验证。' }; if (Test-Path '候选模型稳健性验证结果.zip') { $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'; Move-Item -LiteralPath '候选模型稳健性验证结果.zip' -Destination ('历史回传包_' + $stamp + '.zip') }; Compress-Archive -Path $items -DestinationPath '候选模型稳健性验证结果.zip' -Force"
if errorlevel 1 (
  echo 打包失败，请保留屏幕上的错误信息。
  pause
  exit /b 1
)
echo 已生成 候选模型稳健性验证结果.zip
pause
""",
    }


def build_readme() -> str:
    return """
# 候选模型稳健性验证便携运行包

本包仅验证 7 个已在随机 80%/20% 留出中出现正信号的任务：

- Chl-a：surface、middle、bottom
- 透明度：surface、middle、bottom
- TN：bottom

每个任务都在同一数据、同一特征集、同一划分下比较 XGBoost、TabM、TabPFN。
同期 TN、TP、盐度、DO、pH、浊度等水质变量始终不作为输入。

## 验证口径

1. `02_三随机种子验证.bat`：种子 7、42、123 的随机 80%/20% 划分。保留本研究随机预测设定中允许使用的目标历史特征。
2. `03_年份留出验证.bat`：2014-2022 年训练，2023-2024 年测试。排除目标自身历史特征，避免未来信息进入输入。
3. `04_五折留站点验证.bat`：按站点五折留出。排除目标自身历史特征，避免未见站点的真实历史水质进入输入。

年份留出和留站点结果用于判断候选模型能否在更严格的泛化情形下保持优势；它们不替换本研究当前的正式随机留出评价口径。

## 运行顺序

1. 运行 `setup_nvidia.bat`，完成后必须看到 `CUDA available: True`。
2. 运行 `01_通路检查.bat`。该步骤只检查 3 个代表性任务，确认环境、数据和模型权重可用。
3. 依次运行 `02_三随机种子验证.bat`、`03_年份留出验证.bat`、`04_五折留站点验证.bat`。
4. 三项正式验证都结束后，运行 `05_打包回传结果.bat`。
5. 回传生成的 `候选模型稳健性验证结果.zip`。

中途关闭后，可在对应命令末尾手动加上 `--resume` 后重跑；请不要删除原输出目录。

## 输出文件

每个 `outputs_*` 目录包含：

- `model_metrics.csv`：每个任务、模型、划分的 R2、RMSE、MAE、Bias、NSE。
- `candidate_validation_summary.csv`：按验证方式合并预测后的汇总结果及相对 XGBoost 变化。
- `split_metric_summary.csv`：多随机种子或五折留站点的 R2 均值、标准差、最小值、最大值及相对 XGBoost 均值变化。
- `model_predictions.csv`：预测值与残差，供后续绘图和高值误差分析。
- `feature_manifest.csv`：每个任务实际使用的字段。
- `split_manifest.csv`：训练/测试样本键，供复核划分。
- `结果摘要.md`：可直接阅读的汇总表。

## 数据和权重来源

- `data/model_table.csv`：与前一轮完整 12 任务算法对比相同的 108 列建模表。
- `models/tabpfn-v2-regressor.ckpt`：已随包提供的本地 TabPFN 权重，不需要连接 Hugging Face。
- `config.json`：固定模型参数和 7 项任务对应的既有特征组。
"""


def create_package() -> None:
    check_sources()
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    archive_existing(PACKAGE_DIR, "portable_package")
    archive_existing(ZIP_PATH, "候选模型稳健性验证_便携运行包")

    (PACKAGE_DIR / "data").mkdir(parents=True)
    (PACKAGE_DIR / "models").mkdir()
    (PACKAGE_DIR / "outputs").mkdir()

    copy_pairs = {
        SOURCE_PACKAGE / "data/model_table.csv": PACKAGE_DIR / "data/model_table.csv",
        SOURCE_PACKAGE / "config.json": PACKAGE_DIR / "config.json",
        SOURCE_PACKAGE / "models/tabpfn-v2-regressor.ckpt": PACKAGE_DIR / "models/tabpfn-v2-regressor.ckpt",
        SOURCE_PACKAGE / "requirements.txt": PACKAGE_DIR / "requirements.txt",
        SOURCE_PACKAGE / "setup_nvidia.bat": PACKAGE_DIR / "setup_nvidia.bat",
        SOURCE_PACKAGE / "setup_cpu.bat": PACKAGE_DIR / "setup_cpu.bat",
        SOURCE_RUNNER: PACKAGE_DIR / "run_experiment.py",
        SOURCE_VALIDATOR: PACKAGE_DIR / "run_candidate_validation.py",
        SOURCE_VALIDATION_CONFIG: PACKAGE_DIR / "validation_config.json",
    }
    for source, target in copy_pairs.items():
        shutil.copy2(source, target)

    for name, content in make_bat_files().items():
        write_text(PACKAGE_DIR / name, content)
    write_text(PACKAGE_DIR / "README_运行说明.md", build_readme())

    provenance = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "实验范围": "Chl-a 三水层、透明度三水层、TN 底层的多随机种子、年份留出和五折留站点验证",
        "正式主链影响": "无；本包只用于扩展实验",
        "来源": {
            "数据": str(SOURCE_PACKAGE / "data/model_table.csv"),
            "模型配置": str(SOURCE_PACKAGE / "config.json"),
            "TabPFN权重": str(SOURCE_PACKAGE / "models/tabpfn-v2-regressor.ckpt"),
            "基础运行器": str(SOURCE_RUNNER),
            "验证运行器": str(SOURCE_VALIDATOR),
            "验证配置": str(SOURCE_VALIDATION_CONFIG),
        },
        "文件校验": {
            "model_table.csv": sha256(PACKAGE_DIR / "data/model_table.csv"),
            "config.json": sha256(PACKAGE_DIR / "config.json"),
            "tabpfn-v2-regressor.ckpt": sha256(PACKAGE_DIR / "models/tabpfn-v2-regressor.ckpt"),
            "run_candidate_validation.py": sha256(PACKAGE_DIR / "run_candidate_validation.py"),
            "validation_config.json": sha256(PACKAGE_DIR / "validation_config.json"),
        },
    }
    (PACKAGE_DIR / "数据来源与校验.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    root_name = "候选模型稳健性验证_便携运行包"
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in PACKAGE_DIR.rglob("*"):
            if path.is_file():
                archive.write(path, Path(root_name) / path.relative_to(PACKAGE_DIR))
    print(f"已生成便携运行包：{ZIP_PATH}")


if __name__ == "__main__":
    create_package()
