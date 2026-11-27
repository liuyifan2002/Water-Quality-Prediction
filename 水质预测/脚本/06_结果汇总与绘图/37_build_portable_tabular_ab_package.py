from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA = PROJECT_ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv"
SOURCE_REFERENCE = PROJECT_ROOT / "结果/正式结果/full_station/full_station_best_by_target_depth.csv"
SOURCE_RUNNER = PROJECT_ROOT / "脚本/04_模型训练/31_run_tabular_pretrained_ab.py"
SOURCE_TABPFN_MODEL = Path(r"D:/Users/刘逸凡/Downloads/tabpfn-v2-regressor.ckpt")
EXPERIMENT_ROOT = PROJECT_ROOT / "结果/扩展实验/表格预训练算法对比_20260908"
PACKAGE_DIR = EXPERIMENT_ROOT / "portable_package"
ZIP_PATH = EXPERIMENT_ROOT / "表格预训练算法对比_便携运行包_20260908.zip"

TARGET_COLUMNS = ["chla_ug_l", "secchi_depth_m", "tn_mg_l", "tp_mg_l"]
META_COLUMNS = [
    "obs_id",
    "sample_id",
    "station",
    "station_name_en",
    "water_zone_en",
    "lon",
    "lat",
    "date",
    "year",
    "month",
    "quarter",
    "day_of_year",
    "year_offset",
    "month_sin",
    "month_cos",
    "is_wet_season",
    "depth",
]
BASE_FEATURES = [
    "days_since_prev_sample",
    "t2m",
    "u10",
    "v10",
    "tcc",
    "wind_speed",
    "wind_speed_sq",
    "wind_dir_sin",
    "wind_dir_cos",
    "rain_3d",
    "rain_14d",
    "rain_month",
    "tidal_range",
    "ntl_avg_rad",
    "viirs_count",
    "t2m_x_rain14d",
    "wind_speed_x_rain14d",
    "rain14d_x_tidal_range",
    "t2m_x_tidal_range",
    "ssrd_sum_3d",
    "ssrd_mean_3d",
    "ssrd_sum_7d",
    "ssrd_mean_7d",
    "ssrd_sum_14d",
    "ssrd_mean_14d",
    "sst_mean_3d",
    "sst_max_3d",
    "sst_anom_3d",
    "sst_mean_7d",
    "sst_max_7d",
    "sst_anom_7d",
    "sst_mean_14d",
    "sst_max_14d",
    "sst_anom_14d",
    "sst_mean_7d_x_ssrd_mean_7d",
    "sst_anom_7d_x_ssrd_mean_7d",
    "rain14d_x_ssrd_mean_7d",
    "landsat_available",
    "hq_satellite_available_count",
]
LAG_BASES = ["t2m", "tcc", "wind_speed", "rain_3d", "rain_14d", "rain_month", "tidal_range", "ntl_avg_rad"]
SUFFIXES = ["_lag1", "_lag2", "_roll3", "_delta1"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8-sig")


def prepare_portable_data() -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(SOURCE_DATA, nrows=0).columns.tolist()
    history_columns = [f"{target}{suffix}" for target in TARGET_COLUMNS for suffix in SUFFIXES]
    process_lags = [f"{base}{suffix}" for base in LAG_BASES for suffix in SUFFIXES]
    requested = list(dict.fromkeys(META_COLUMNS + TARGET_COLUMNS + BASE_FEATURES + history_columns + process_lags))
    selected = [column for column in requested if column in header]
    missing = [column for column in requested if column not in header]
    data = pd.read_csv(SOURCE_DATA, usecols=selected, low_memory=False)
    return data, missing


def build_config(reference: pd.DataFrame) -> dict[str, object]:
    task_feature_sets = {
        f"{row.target}_{row.depth}": row.feature_set
        for row in reference.itertuples()
    }
    return {
        "experiment_name": "表格预训练算法同数据同划分对比",
        "dataset": "data/model_table.csv",
        "reference_metrics_file": "reference_metrics.csv",
        "split": {"mode": "random", "test_size": 0.2, "random_state": 42},
        "target_transform": "log1p_when_nonnegative",
        "same_day_water_quality_covariates": "excluded",
        "task_feature_sets": task_feature_sets,
        "model_parameters": {
            "xgboost": {
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_jobs": 4,
            },
            "catboost": {
                "iterations": 500,
                "depth": 6,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
                "thread_count": 4,
            },
            "tabpfn": {"n_estimators": 8, "model_path": "models/tabpfn-v2-regressor.ckpt"},
            "tabm": {
                "arch_type": "tabm-mini",
                "k": 16,
                "batch_size": 256,
                "learning_rate": 0.002,
                "weight_decay": 0.00001,
                "max_epochs": 200,
                "patience": 25,
                "validation_fraction": 0.2,
            },
        },
    }


def create_package_files(data: pd.DataFrame, reference: pd.DataFrame, missing: list[str]) -> None:
    if not SOURCE_TABPFN_MODEL.exists():
        raise FileNotFoundError(f"找不到待打包的 TabPFN 权重：{SOURCE_TABPFN_MODEL}")
    if PACKAGE_DIR.exists():
        archive_dir = EXPERIMENT_ROOT / "历史运行包"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(str(PACKAGE_DIR), str(archive_dir / f"portable_package_{stamp}"))
    (PACKAGE_DIR / "data").mkdir(parents=True)
    (PACKAGE_DIR / "models").mkdir()
    (PACKAGE_DIR / "outputs").mkdir()
    shutil.copy2(SOURCE_RUNNER, PACKAGE_DIR / "run_experiment.py")
    shutil.copy2(SOURCE_TABPFN_MODEL, PACKAGE_DIR / "models/tabpfn-v2-regressor.ckpt")
    data.to_csv(PACKAGE_DIR / "data/model_table.csv", index=False, encoding="utf-8-sig")
    reference[
        ["target", "depth", "feature_set", "train_rows", "test_rows", "r2", "rmse", "mae"]
    ].to_csv(PACKAGE_DIR / "reference_metrics.csv", index=False, encoding="utf-8-sig")
    (PACKAGE_DIR / "config.json").write_text(
        json.dumps(build_config(reference), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    requirements = """
numpy>=1.26,<3
pandas>=2.2,<3
scikit-learn>=1.5,<2
matplotlib>=3.8,<4
xgboost>=2.1,<4
catboost>=1.2,<2
tabpfn>=2.2,<4
tabm>=0.0.3,<0.1
"""
    write_text(PACKAGE_DIR / "requirements.txt", requirements)

    setup_nvidia = r"""
@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_CMD="
py -3.11 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD (
  python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
  echo 未找到可用的 Python 3.11 或 3.12。请先安装 64 位 Python。
  pause
  exit /b 1
)
%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
  echo 虚拟环境创建失败，安装已停止。
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo CUDA PyTorch is unavailable. Do not run TabPFN or TabM until this is fixed.
  pause
  exit /b 1
)
echo 环境安装结束。TabPFN 第一次运行时还会下载本地模型权重。
pause
"""
    setup_cpu = r"""
@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_CMD="
py -3.11 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD (
  python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
  echo 未找到可用的 Python 3.11 或 3.12。请先安装 64 位 Python。
  pause
  exit /b 1
)
%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
  echo 虚拟环境创建失败，安装已停止。
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo CPU 环境安装结束。CPU 模式不建议运行 TabPFN 和 TabM 全量实验。
pause
"""
    run_quick = r"""
@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 请先运行 setup_nvidia.bat 或 setup_cpu.bat。
  pause
  exit /b 1
)
.venv\Scripts\python.exe run_experiment.py --quick --resume
pause
"""
    run_all = r"""
@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 请先运行 setup_nvidia.bat。
  pause
  exit /b 1
)
.venv\Scripts\python.exe run_experiment.py --resume
pause
"""
    run_tree_only = r"""
@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 请先运行安装脚本。
  pause
  exit /b 1
)
.venv\Scripts\python.exe run_experiment.py --models xgboost catboost --resume
pause
"""
    pack_results = r"""
@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path 'results_to_return.zip') { $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'; Move-Item 'results_to_return.zip' ('results_to_return_previous_' + $stamp + '.zip') }"
powershell -NoProfile -Command "Compress-Archive -Path outputs\* -DestinationPath results_to_return.zip -Force"
echo 已生成 results_to_return.zip，请把这个文件复制回原电脑。
pause
"""
    write_text(PACKAGE_DIR / "setup_nvidia.bat", setup_nvidia)
    write_text(PACKAGE_DIR / "setup_cpu.bat", setup_cpu)
    write_text(PACKAGE_DIR / "run_quick.bat", run_quick)
    write_text(PACKAGE_DIR / "run_all.bat", run_all)
    write_text(PACKAGE_DIR / "run_tree_only.bat", run_tree_only)
    write_text(PACKAGE_DIR / "pack_results.bat", pack_results)
    write_text(PACKAGE_DIR / "outputs/这里保存运行结果.txt", "该目录由程序自动写入，请勿放入其他文件。")

    readme = f"""
# 表格预训练算法对比便携运行包

## 一、实验目的

在完全相同的数据、特征和固定随机 80%/20% 划分下，对比 XGBoost、CatBoost、TabPFN 和 TabM。实验不使用同日其他水质指标，不覆盖原项目正式结果。

## 二、数据说明

- 数据行数：{len(data):,}
- 数据列数：{len(data.columns)}
- 时间范围：2014—2024 年
- 研究对象：香港近岸 94 个站点
- 目标：Chl-a、透明度、TN、TP
- 水层：surface、middle、bottom
- 已删除同期水质协变量，只保留四个目标值、目标自身历史特征、环境特征、时间空间特征和正式模型实际使用的遥感可用性字段。
- 透明度在三个水层任务中沿用同一站点日期观测值；论文解释时不能把三层记录当作三次独立透明度测量。
- 正式主表中未找到的候选字段数量：{len(missing)}。这些字段不会在本实验中临时补造。

## 三、推荐电脑

- Windows 10/11，64 位 Python 3.11 或 3.12。
- NVIDIA 显卡，建议显存不少于 8 GB。
- 至少 16 GB 内存，磁盘剩余空间建议 15 GB 以上。
- 首次安装依赖时需要联网；TabPFN 回归权重已随本包提供，运行时不再访问 Hugging Face。
- `setup_nvidia.bat` 固定安装 CUDA 12.4 版 PyTorch；安装结束必须显示 `CUDA available: True` 才可运行 TabPFN 和 TabM。

## 四、运行步骤

1. 解压整个 ZIP，不要只打开压缩包直接运行。
2. 双击 `setup_nvidia.bat`。没有 NVIDIA 显卡时可运行 `setup_cpu.bat`。
3. 先双击 `run_quick.bat`，确认四个代表任务能完成。
4. 快速测试无严重错误后，双击 `run_all.bat` 运行全部任务。
5. 如果 TabPFN 或 TabM 安装失败，可先运行 `run_tree_only.bat`，检查数据和固定划分是否可复现。
6. 全部结束后双击 `pack_results.bat`，得到 `results_to_return.zip`。
7. 将 `results_to_return.zip` 复制回原电脑交给 Codex 分析。

脚本支持断点继续。重新双击 `run_all.bat` 时，已经成功的任务会跳过，失败任务会再次尝试。

## 五、必须带回的结果

`results_to_return.zip` 中应至少包含：

- `model_metrics.csv`：全部模型指标；
- `model_predictions.csv`：逐样本实测值和预测值；
- `split_manifest.csv`：训练集和测试集编号；
- `feature_manifest.csv`：每个任务的完整特征；
- `environment.json` 和 `pip_freeze.txt`：电脑与软件环境；
- `run_status.json`：完成和失败数量；
- `model_errors.log`：如有失败，保存完整报错；
- `结果摘要.md` 和两张对比图。

## 六、重要限制

- 不要修改 `data/model_table.csv`、`config.json` 或 `reference_metrics.csv`。
- 不要把盐度、DO、pH、浊度等同期水质重新加入输入。
- 不要根据测试集结果修改参数后重复报告同一测试集成绩。
- TabPFN 必须使用本地运行方式，本包不会把数据上传到第三方接口。
- `models/tabpfn-v2-regressor.ckpt` 是 TabPFN 的固定本地回归权重，不要移动、改名或删除。
- 本轮只判断新算法是否值得继续。通过后才进行多水层共享模型和年份/留站点验证。

## 七、远程协作

当前 Codex 不能自动接管另一台未连接的电脑。最简单的方式是在另一台电脑安装 Codex，打开本目录并让它运行 `run_all.bat`；或者运行完成后把 `results_to_return.zip` 带回当前项目。不要通过聊天发送账号、密码或远程控制验证码。
"""
    write_text(PACKAGE_DIR / "README_运行说明.md", readme)

    provenance = {
        "source_dataset": str(SOURCE_DATA),
        "source_dataset_sha256": sha256(SOURCE_DATA),
        "portable_dataset_rows": len(data),
        "portable_dataset_columns": len(data.columns),
        "portable_dataset_sha256": sha256(PACKAGE_DIR / "data/model_table.csv"),
        "source_reference": str(SOURCE_REFERENCE),
        "source_reference_sha256": sha256(SOURCE_REFERENCE),
        "tabpfn_model_source": str(SOURCE_TABPFN_MODEL),
        "tabpfn_model_sha256": sha256(SOURCE_TABPFN_MODEL),
        "tabpfn_model_bytes": SOURCE_TABPFN_MODEL.stat().st_size,
        "excluded_same_day_covariates": True,
        "missing_requested_fields": missing,
    }
    (PACKAGE_DIR / "data_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_manifest_and_zip() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(PACKAGE_DIR.rglob("*")):
        if path.is_file() and path.name != "manifest_sha256.csv":
            rows.append(
                {
                    "relative_path": path.relative_to(PACKAGE_DIR).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    with (PACKAGE_DIR / "manifest_sha256.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        archive_dir = EXPERIMENT_ROOT / "历史运行包"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(str(ZIP_PATH), str(archive_dir / f"{ZIP_PATH.stem}_{stamp}.zip"))
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=Path("表格预训练算法对比_便携运行包") / path.relative_to(PACKAGE_DIR))


def main() -> None:
    for path in [SOURCE_DATA, SOURCE_REFERENCE, SOURCE_RUNNER]:
        if not path.exists():
            raise FileNotFoundError(path)
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    data, missing = prepare_portable_data()
    reference = pd.read_csv(SOURCE_REFERENCE)
    create_package_files(data, reference, missing)
    write_manifest_and_zip()
    print(f"便携目录：{PACKAGE_DIR}")
    print(f"ZIP 文件：{ZIP_PATH}")
    print(f"数据：{len(data):,} 行 × {len(data.columns)} 列")


if __name__ == "__main__":
    main()
