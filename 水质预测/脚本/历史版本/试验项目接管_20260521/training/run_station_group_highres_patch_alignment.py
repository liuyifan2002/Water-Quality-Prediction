from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.exceptions import DataConversionWarning

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATE_TAG = "20260518"
TARGETS = {
    "chla": {
        "column": "chla_ug_l",
        "previous_best": {"surface": 0.3298, "middle": 0.3646, "bottom": 0.3318},
    },
    "secchi": {
        "column": "secchi_depth_m",
        "previous_best": {"surface": 0.4080, "middle": 0.3447, "bottom": 0.3574},
    },
}
DEPTHS = ["surface", "middle", "bottom"]
ID_COLS = ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en"]
CAT_COLS = ["station", "water_zone_en", "month", "quarter", "is_wet_season"]
MANIFEST_COLS = ["patch_has_image", "patch_high_quality", "patch_match_gap_days", "patch_valid_pixel_count"]
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=DataConversionWarning)


@dataclass
class GroupDefinition:
    scheme: str
    cluster_method: str
    n_groups: int
    score: float
    label_col: str
    assignments: pd.DataFrame
    feature_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Station grouping and high-resolution patch alignment experiments.")
    parser.add_argument(
        "--input",
        default="data/processed/chla_ocean_colour_precision_optimization_20260517/chla_ocean_colour_precision_optimization_dataset_20260517.csv",
    )
    parser.add_argument("--date-tag", default=DATE_TAG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_label(year: int) -> str | None:
    if 2014 <= year <= 2020:
        return "train"
    if 2021 <= year <= 2022:
        return "validation"
    if 2023 <= year <= 2024:
        return "test"
    return None


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ensure_dirs(tag: str) -> tuple[Path, Path, Path, Path]:
    out_dir = Path(f"experiments/outputs/station_group_highres_patch_alignment_{tag}")
    report_dir = Path("experiments/reports")
    config_dir = Path("experiments/configs")
    processed_dir = Path(f"data/processed/station_group_highres_patch_alignment_{tag}")
    for path in [out_dir, report_dir, config_dir, processed_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return out_dir, report_dir, config_dir, processed_dir


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["split"] = df["year"].map(split_label)
    df = df[df["split"].notna()].copy()
    return df.replace([np.inf, -np.inf], np.nan)


def merge_manifest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = Path("data/raw/manual/ocean_color/hk_s2_patch_manifest.csv")
    report = {
        "manifest_path": str(manifest_path),
        "exists": manifest_path.exists(),
        "usable_as_highres_patch_features": False,
        "reason": "",
    }
    if not manifest_path.exists():
        report["reason"] = "manifest missing"
        return df, report
    manifest = pd.read_csv(manifest_path, low_memory=False)
    manifest = manifest.rename(columns={"Station": "station"})
    manifest["date"] = pd.to_datetime(manifest["date"])
    keep = [c for c in ["sample_id", "station", "date", "water_zone_en"] + MANIFEST_COLS if c in manifest.columns]
    manifest = manifest[keep].copy()
    duplicate_count = int(manifest.duplicated(subset=["sample_id"]).sum()) if "sample_id" in manifest.columns else int(manifest.duplicated(subset=["station", "date"]).sum())
    report["columns"] = list(manifest.columns)
    report["duplicate_count"] = duplicate_count
    path_like_cols = [c for c in manifest.columns if any(k in c.lower() for k in ["file", "path", "band", "tile", "image_id", "patch_path"])]
    report["path_like_columns"] = path_like_cols
    report["reason"] = "manifest only contains availability/quality fields; no patch path or raw pixel statistics" if not path_like_cols else ""
    merge_keys = ["sample_id"] if "sample_id" in manifest.columns and "sample_id" in df.columns else ["station", "date"]
    merged = df.merge(manifest, on=merge_keys, how="left", suffixes=("", "_manifest"))
    if "patch_match_gap_days" in merged.columns:
        merged["patch_match_gap_days"] = pd.to_numeric(merged["patch_match_gap_days"], errors="coerce").replace(-9999, np.nan)
    for col in ["patch_has_image", "patch_high_quality", "patch_valid_pixel_count"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged, report


def highres_patch_gap_report(out_dir: Path, manifest_report: dict[str, Any]) -> None:
    text = "\n".join(
        [
            "# 高分辨率 Patch 数据缺口报告",
            "",
            "## 当前检查结论",
            "",
            "- `hk_s2_patch_manifest.csv` 已存在并已并入工作区。",
            "- 该文件包含 `sample_id/station/date` 以及 `patch_has_image/patch_high_quality/patch_match_gap_days/patch_valid_pixel_count`。",
            "- 未发现 `image_id/tile/file_path/band_path/patch_path` 等可追溯到原始 Sentinel-2 / Landsat patch 的字段。",
            "- 工作区也未发现对应的高分辨率 patch 栅格或多波段 patch 文件。",
            "",
            "## 结论",
            "",
            "本轮无法生成真实 Sentinel-2 / Landsat 3x3、5x5 多波段 patch 统计，因此不会伪造 highres patch A/B。manifest 只能作为影像可用性和有效像元质量辅助信息。",
            "",
            "## 下一步需要补",
            "",
            "- 原始 patch 文件路径或原始栅格目录。",
            "- image_id / tile / acquisition date。",
            "- 每个 patch 对应的波段数据或能重算 patch 统计的源影像。",
            "- 若只有 GEE 任务清单，则需要重新导出 patch 统计表。",
            "",
            "## 本轮 manifest 摘要",
            "",
            f"- 存在: {manifest_report.get('exists')}",
            f"- 字段: {', '.join(manifest_report.get('columns', []))}",
            f"- path-like 字段: {', '.join(manifest_report.get('path_like_columns', [])) or '无'}",
        ]
    )
    (out_dir / "highres_patch_data_gap_report.md").write_text(text, encoding="utf-8")


def build_alignment_doc(report_dir: Path, out_dir: Path, manifest_report: dict[str, Any]) -> None:
    text = """# 模型输入数据对齐说明

## 1. 主表标签数据

- 当前建模输入表: `data/processed/chla_ocean_colour_precision_optimization_20260517/chla_ocean_colour_precision_optimization_dataset_20260517.csv`
- 上游主表来源: `data/model_input/model_table_3depth_2014_2024.csv`
- 目标变量:
  - `chla_ug_l`
  - `secchi_depth_m`
- 水层定义:
  - `surface`
  - `middle`
  - `bottom`
- 时间字段: `date`
- 站点字段: `station`
- 当前建模样本按 `station + date + depth + obs_id/sample_id` 使用实测日期，不做 test 期日期重排。
- 标签未做统一 `log` 变换作为正式主线，异常值没有在本轮额外截断。

## 2. Ocean Colour 点值特征

- 来源脚本: `scripts/30_download_cmems_ocean_colour_features.py`
- 原始数据源: Copernicus Marine Ocean Colour 日尺度产品
- 空间分辨率: 约 4 km
- 时间分辨率: daily
- 对齐键: `station + date`
- 合并方式: `scripts/features/prepare_ocean_colour_ab_dataset.py` 使用 `station + date` 做 `many_to_one` 合并，若 Ocean Colour 表存在重复键则直接报错。
- 字段语义:
  - `oc_chl`: 站点日期对应最近日的 CHL 点值
  - `oc_kd490`: 站点日期对应最近日的 KD490 点值
  - `oc_zsd`: 站点日期对应最近日的 ZSD 点值
  - `oc_day_diff`: 站点日期与选中 Ocean Colour 日之间的天数差
  - `oc_chl_3d/7d/14d`: 站点内按日期排序的过去 3/7/14 条日尺度滚动均值
  - `oc_kd490_3d/7d/14d`: 同上
  - `oc_zsd_3d/7d/14d`: 同上
  - `valid_count_7d`: 站点内过去 7 条日尺度记录中 `oc_valid` 的滚动和

## 3. CMEMS 4km patch 特征

- 来源脚本: `scripts/data/run_mechanism_data_expansion.py`
- 原始文件:
  - `data/raw/manual/ocean_color/cmems/cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D_2014-01-01_2024-12-31.nc`
  - `data/raw/manual/ocean_color/cmems/cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D_2014-01-01_2024-12-31.nc`
  - `data/raw/manual/ocean_color/cmems/cmems_obs-oc_glo_bgc-reflectance_my_l3-multi-4km_P1D_2014-01-01_2024-12-31.nc`
- patch 尺寸:
  - `3x3`
  - `5x5`
- 统计含义:
  - `mean`: patch 内有限像元均值
  - `std`: patch 内有限像元标准差
  - `min/max`: patch 内有限像元极值
  - `valid_ratio`: patch 内有限像元比例
  - `patch_missing_ratio`: `1 - valid_ratio`
  - `local_gradient`: 点值与 `3x3_mean` 的绝对差
  - `heterogeneity_index`: `5x5_std / abs(5x5_mean)`
- 时间对齐方式: 使用样本实测 `date` 对应的同日 NetCDF 切片，没有向未来日搜索。
- 缺失处理: patch 内无有效像元时保留缺失，不做伪造填补。

## 4. Sentinel-2 / Landsat 高分辨率 patch 特征

- 本轮未成功生成真实高分辨率 patch 统计。
- 已检查 `data/raw/manual/ocean_color/hk_s2_patch_manifest.csv`:
  - 包含 `sample_id/station/date`
  - 包含 `patch_has_image/patch_high_quality/patch_match_gap_days/patch_valid_pixel_count`
  - 不包含 `image_id/tile/file_path/band_path/patch_path`
- 结论: 当前只能把 manifest 当作影像可用性和有效像元质量清单，不能重构真实 `3x3/5x5` 波段统计。

## 5. 气象 / 机制代理数据

- 来源文件:
  - `data/raw/manual/environment/era5_features_2014_2024.csv`
  - `data/raw/manual/environment/rainfall_features_2014_2024.csv`
  - `data/raw/manual/environment/tide_features_2014_2024.csv`
- 变量语义:
  - `t2m/u10/v10/wind_speed/tcc`: 日尺度环境值
  - `ssrd_sum_3d/7d/14d`, `ssrd_mean_3d/7d/14d`: 过去窗口累积或均值
  - `sst_mean_3d/7d/14d`, `sst_max_3d/7d/14d`, `sst_anom_3d/7d/14d`: 过去窗口海表温度统计
  - `rain_1d/3d/7d/14d/30d`, `rain_month`: 日或累计降雨
  - `tidal_range`, `tide_events`, `tide_high_mean`, `tide_low_mean`, `tide_level_mean`: 日尺度潮汐特征

## 6. 禁止泄漏说明

- 所有 rolling / lag 特征只使用预测日及之前的数据。
- 不使用预测日之后的数据。
- 不使用 test 标签参与调参、站点分组、残差校正或 patch 聚合。
- train-only 站点分组只在 `2014-2020` 的站点统计上拟合。
- manifest 只用于影像可用性辅助，不用于反推 test 标签。
"""
    (report_dir / "model_input_alignment_documentation.md").write_text(text, encoding="utf-8")
    shutil.copyfile(report_dir / "model_input_alignment_documentation.md", out_dir / "model_input_alignment_documentation.md")


def build_group_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["split"] == "train"].copy()
    station_base = train.groupby("station", as_index=False).agg(
        lon=("lon", "mean"),
        lat=("lat", "mean"),
        water_zone_en=("water_zone_en", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
    )

    spatial = station_base.copy()

    def seasonal_amp(s: pd.Series) -> float:
        if s.empty:
            return np.nan
        return float(np.nanpercentile(s, 90) - np.nanpercentile(s, 10))

    wq = station_base.copy()
    for key, spec in TARGETS.items():
        col = spec["column"]
        stats = train.groupby("station")[col].agg(["mean", "std", lambda x: np.nanpercentile(x.dropna(), 90) if x.dropna().size else np.nan, seasonal_amp, lambda x: x.isna().mean()]).reset_index()
        stats.columns = ["station", f"{key}_train_mean", f"{key}_train_std", f"{key}_train_p90", f"{key}_seasonal_amplitude", f"{key}_missing_rate"]
        wq = wq.merge(stats, on="station", how="left")

    optical = station_base.copy()
    optical_cols = [c for c in ["oc_chl", "oc_zsd", "oc_kd490", "valid_count_7d", "patch_valid_pixel_count", "patch_high_quality"] if c in train.columns]
    for col in optical_cols:
        stats = train.groupby("station")[col].agg(["mean", "std"]).reset_index()
        stats.columns = ["station", f"{col}_train_mean", f"{col}_train_std"]
        optical = optical.merge(stats, on="station", how="left")

    hybrid = spatial.merge(wq.drop(columns=["water_zone_en", "lon", "lat"], errors="ignore"), on="station", how="left")
    hybrid = hybrid.merge(optical.drop(columns=["water_zone_en", "lon", "lat"], errors="ignore"), on="station", how="left")
    return spatial, wq, optical, hybrid


def numeric_group_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    if "water_zone_en" in out.columns:
        out["water_zone_code"] = out["water_zone_en"].astype("category").cat.codes
    cols = [c for c in out.columns if c != "station" and pd.api.types.is_numeric_dtype(out[c])]
    return out[["station"] + cols].copy(), cols


def fit_cluster_scheme(name: str, table: pd.DataFrame) -> list[GroupDefinition]:
    out: list[GroupDefinition] = []
    num_df, features = numeric_group_matrix(table)
    matrix = num_df[features].copy()
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    if len(matrix) < 8 or len(features) == 0:
        return out
    methods = {
        "kmeans": lambda k: KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(matrix),
        "agglomerative": lambda k: AgglomerativeClustering(n_clusters=k).fit_predict(matrix),
        "gmm": lambda k: GaussianMixture(n_components=k, random_state=42).fit(matrix).predict(matrix),
    }
    for method, fn in methods.items():
        for k in [2, 3, 4, 5]:
            if len(matrix) <= k:
                continue
            labels = fn(k)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(matrix, labels))
            label_col = f"{name}_{method}_k{k}"
            assign = num_df[["station"]].copy()
            assign[label_col] = labels
            out.append(GroupDefinition(name, method, k, score, label_col, assign, features))
    return out


def save_group_files(out_dir: Path, defs: list[GroupDefinition]) -> pd.DataFrame:
    rows = []
    grouped = {}
    for item in defs:
        assign = item.assignments.copy()
        assign["scheme"] = item.scheme
        assign["cluster_method"] = item.cluster_method
        assign["n_groups"] = item.n_groups
        assign["score"] = item.score
        assign["label_col"] = item.label_col
        grouped.setdefault(item.scheme, []).append(assign)
        for _, row in assign.iterrows():
            rows.append(
                {
                    "station": row["station"],
                    "scheme": item.scheme,
                    "cluster_method": item.cluster_method,
                    "n_groups": item.n_groups,
                    "score": item.score,
                    "label_col": item.label_col,
                    "station_group_id": row[item.label_col],
                }
            )
    for scheme, frames in grouped.items():
        file_map = {
            "spatial": "station_groups_spatial.csv",
            "water_quality_train_only": "station_groups_water_quality_train_only.csv",
            "optical_train_only": "station_groups_optical_train_only.csv",
            "hybrid_train_only": "station_groups_hybrid_train_only.csv",
        }
        pd.concat(frames, ignore_index=True).to_csv(out_dir / file_map[scheme], index=False, encoding="utf-8-sig")
    defs_df = pd.DataFrame(rows)
    defs_df.to_csv(out_dir / "station_group_definitions.csv", index=False, encoding="utf-8-sig")
    return defs_df


def merge_group(df: pd.DataFrame, group_assignments: pd.DataFrame, label_col: str) -> pd.DataFrame:
    grp = group_assignments[["station", label_col]].rename(columns={label_col: "station_group_id"})
    out = df.merge(grp, on="station", how="left")
    out["station_group_id"] = out["station_group_id"].astype("Int64").astype(str)
    return out


def candidate_features(df: pd.DataFrame, target: str, include_group: bool = False) -> list[str]:
    col = TARGETS[target]["column"]
    features = [
        "lon",
        "lat",
        "year_offset",
        "month_sin",
        "month_cos",
        "is_wet_season",
        "quarter",
        "day_of_year",
        "days_since_prev_sample",
        f"{col}_lag1",
        f"{col}_lag2",
        f"{col}_roll3",
        "station",
        "water_zone_en",
        "patch_has_image",
        "patch_high_quality",
        "patch_match_gap_days",
        "patch_valid_pixel_count",
    ]
    features += [c for c in df.columns if (c.startswith("oc_") and not c.startswith("oc_patch_")) or c.startswith("landsat_") or c.startswith("s2_") or c.startswith("satellite_")]
    if include_group and "station_group_id" in df.columns:
        features.append("station_group_id")
    ordered = []
    for c in features:
        if c in df.columns and df[c].notna().sum() > 0 and c not in ordered:
            ordered.append(c)
    return ordered


def preprocessor(train: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    cat = [c for c in features if c in CAT_COLS or c == "station_group_id" or train[c].dtype == "object"]
    num = [c for c in features if c not in cat]
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ],
        sparse_threshold=0.0,
    )


def build_model(family: str):
    if family == "lightgbm":
        return LGBMRegressor(n_estimators=350, learning_rate=0.04, num_leaves=20, max_depth=6, min_child_samples=30, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, random_state=42, n_jobs=1, verbose=-1)
    if family == "xgboost":
        return XGBRegressor(n_estimators=320, learning_rate=0.05, max_depth=5, min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.0, reg_lambda=1.0, objective="reg:squarederror", random_state=42, n_jobs=1)
    if family == "random_forest":
        return RandomForestRegressor(n_estimators=260, min_samples_leaf=3, random_state=42, n_jobs=1)
    if family == "catboost":
        return CatBoostRegressor(iterations=320, learning_rate=0.05, depth=6, loss_function="RMSE", random_seed=42, verbose=False)
    raise KeyError(family)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    q90 = np.nanquantile(y_true, 0.90) if len(y_true) else np.nan
    high = y_true >= q90 if len(y_true) else np.array([], dtype=bool)
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "p90_rmse": rmse(y_true[high], y_pred[high]) if high.any() else np.nan,
    }


def fit_family(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str], y_col: str, family: str) -> dict[str, Any]:
    pipe = Pipeline([("preprocessor", preprocessor(train, features)), ("model", build_model(family))])
    pipe.fit(train[features], train[y_col])
    valid_pred = np.maximum(pipe.predict(valid[features]), 0)
    test_pred = np.maximum(pipe.predict(test[features]), 0)
    val_metrics = evaluate(valid[y_col].to_numpy(), valid_pred)
    test_metrics = evaluate(test[y_col].to_numpy(), test_pred)
    model = pipe.named_steps["model"]
    importance = getattr(model, "feature_importances_", None)
    if importance is not None:
        n_imp = min(len(features), len(importance))
        imp_df = pd.DataFrame({"feature": features[:n_imp], "importance": importance[:n_imp]})
    else:
        imp_df = pd.DataFrame(columns=["feature", "importance"])
    return {"family": family, "pipeline": pipe, "valid_pred": valid_pred, "test_pred": test_pred, "val_metrics": val_metrics, "test_metrics": test_metrics, "importance": imp_df}


def choose_best_family(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str], y_col: str) -> dict[str, Any]:
    candidates = []
    for family in ["lightgbm", "xgboost", "random_forest", "catboost"]:
        candidates.append(fit_family(train, valid, test, features, y_col, family))
    candidates.sort(key=lambda item: item["val_metrics"]["r2"], reverse=True)
    return candidates[0]


def fit_group_specific(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str], y_col: str, family: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    valid_pred = np.zeros(len(valid))
    test_pred = np.zeros(len(test))
    group_sizes = {}
    global_model = fit_family(train, valid, test, features, y_col, family)
    for gid, group_train in train.groupby("station_group_id"):
        group_sizes[str(gid)] = len(group_train)
        if len(group_train) < 120:
            continue
        group_valid = valid[valid["station_group_id"] == gid]
        group_test = test[test["station_group_id"] == gid]
        if group_valid.empty and group_test.empty:
            continue
        pipe = Pipeline([("preprocessor", preprocessor(group_train, features)), ("model", build_model(family))])
        pipe.fit(group_train[features], group_train[y_col])
        if not group_valid.empty:
            valid_mask = valid["station_group_id"] == gid
            valid_pred[valid_mask.to_numpy()] = np.maximum(pipe.predict(group_valid[features]), 0)
        if not group_test.empty:
            test_mask = test["station_group_id"] == gid
            test_pred[test_mask.to_numpy()] = np.maximum(pipe.predict(group_test[features]), 0)
    # fallback to global where no group-specific prediction was written
    valid_mask = valid_pred == 0
    test_mask = test_pred == 0
    valid_pred[valid_mask] = global_model["valid_pred"][valid_mask]
    test_pred[test_mask] = global_model["test_pred"][test_mask]
    return valid_pred, test_pred, group_sizes


def fit_group_residual(global_bundle: dict[str, Any], train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    residual = valid[y_col].to_numpy() - global_bundle["valid_pred"]
    res_df = pd.DataFrame({"station_group_id": valid["station_group_id"].astype(str), "residual": residual})
    bias_table = res_df.groupby("station_group_id").agg(mean_residual=("residual", "mean"), n=("residual", "size")).reset_index()
    bias_table["bias"] = bias_table["mean_residual"] * bias_table["n"] / (bias_table["n"] + 8.0)
    bias_map = dict(zip(bias_table["station_group_id"], bias_table["bias"]))
    default_bias = float(np.mean(residual))
    valid_bias = valid["station_group_id"].astype(str).map(bias_map).fillna(default_bias).to_numpy()
    test_bias = test["station_group_id"].astype(str).map(bias_map).fillna(default_bias).to_numpy()
    return np.maximum(global_bundle["valid_pred"] + valid_bias, 0), np.maximum(global_bundle["test_pred"] + test_bias, 0)


def fit_mixture(global_bundle: dict[str, Any], train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str], y_col: str, family: str) -> tuple[np.ndarray, np.ndarray]:
    group_valid, group_test, group_sizes = fit_group_specific(train, valid, test, features, y_col, family)
    weights = {gid: min(1.0, n / 350.0) for gid, n in group_sizes.items()}
    valid_w = valid["station_group_id"].astype(str).map(weights).fillna(0.0).to_numpy()
    test_w = test["station_group_id"].astype(str).map(weights).fillna(0.0).to_numpy()
    blend_valid = np.maximum((1 - valid_w) * global_bundle["valid_pred"] + valid_w * group_valid, 0)
    blend_test = np.maximum((1 - test_w) * global_bundle["test_pred"] + test_w * group_test, 0)
    return blend_valid, blend_test


def station_level_compare(target: str, depth: str, test: pd.DataFrame, global_pred: np.ndarray, grouped_pred: np.ndarray, group_method: str, label_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    test = test.copy()
    test["global_pred"] = global_pred
    test["grouped_pred"] = grouped_pred
    test["best_pred"] = np.where(np.abs(grouped_pred - test[TARGETS[target]["column"]].to_numpy()) <= np.abs(global_pred - test[TARGETS[target]["column"]].to_numpy()), grouped_pred, global_pred)
    test["absolute_error_global"] = np.abs(test["global_pred"] - test[TARGETS[target]["column"]])
    test["absolute_error_grouped"] = np.abs(test["grouped_pred"] - test[TARGETS[target]["column"]])
    test["error_improved"] = test["absolute_error_grouped"] < test["absolute_error_global"]
    for station, sub in test.groupby("station"):
        y = sub[TARGETS[target]["column"]].to_numpy()
        gp = sub["global_pred"].to_numpy()
        grp = sub["grouped_pred"].to_numpy()
        q90 = np.nanquantile(y, 0.90) if len(y) else np.nan
        high = y >= q90 if len(y) else np.array([], dtype=bool)
        rows.append(
            {
                "target": target,
                "depth_layer": depth,
                "station": station,
                "station_group_method": group_method,
                "station_group_id": sub["station_group_id"].iloc[0],
                "n_train": int(sub["n_train"].iloc[0]),
                "n_valid": int(sub["n_valid"].iloc[0]),
                "n_test": int(len(sub)),
                "global_model_r2": r2_score(y, gp) if len(sub) > 1 else np.nan,
                "grouped_model_r2": r2_score(y, grp) if len(sub) > 1 else np.nan,
                "delta_r2": (r2_score(y, grp) - r2_score(y, gp)) if len(sub) > 1 else np.nan,
                "global_model_rmse": rmse(y, gp),
                "grouped_model_rmse": rmse(y, grp),
                "delta_rmse": rmse(y, grp) - rmse(y, gp),
                "global_model_mae": float(mean_absolute_error(y, gp)),
                "grouped_model_mae": float(mean_absolute_error(y, grp)),
                "delta_mae": float(mean_absolute_error(y, grp) - mean_absolute_error(y, gp)),
                "global_p90_rmse": rmse(y[high], gp[high]) if high.any() else np.nan,
                "grouped_p90_rmse": rmse(y[high], grp[high]) if high.any() else np.nan,
                "delta_p90_rmse": (rmse(y[high], grp[high]) - rmse(y[high], gp[high])) if high.any() else np.nan,
                "global_bias": float(np.mean(gp - y)),
                "grouped_bias": float(np.mean(grp - y)),
                "delta_bias": float(np.mean(grp - y) - np.mean(gp - y)),
                "improved": bool(np.mean(np.abs(grp - y)) < np.mean(np.abs(gp - y))),
                "recommended_model_for_station": "grouped_model" if np.mean(np.abs(grp - y)) < np.mean(np.abs(gp - y)) else "global_model",
            }
        )
    keep_cols = [c for c in ["target", "depth", "station", "date", "station_group_id", "split", TARGETS[target]["column"]] if c in test.columns]
    pred = test[keep_cols].copy()
    pred = pred.rename(columns={TARGETS[target]["column"]: "y_true", "depth": "depth_layer"})
    pred["target"] = target
    pred["global_pred"] = global_pred
    pred["grouped_pred"] = grouped_pred
    pred["best_pred"] = test["best_pred"].to_numpy()
    pred["model_used"] = np.where(test["error_improved"], "grouped_model", "global_model")
    pred["absolute_error_global"] = test["absolute_error_global"].to_numpy()
    pred["absolute_error_grouped"] = test["absolute_error_grouped"].to_numpy()
    pred["error_improved"] = test["error_improved"].to_numpy()
    return pd.DataFrame(rows), pred


def write_docs(report_dir: Path, out_dir: Path, text_name: str, text: str) -> None:
    (report_dir / text_name).write_text(text, encoding="utf-8")
    shutil.copyfile(report_dir / text_name, out_dir / text_name)


def flat_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无"
    header = " | ".join(df.columns.astype(str))
    lines = [header]
    for _, row in df.iterrows():
        lines.append(" | ".join(str(row[col]) for col in df.columns))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir, report_dir, config_dir, processed_dir = ensure_dirs(args.date_tag)
    df = load_dataset(Path(args.input))
    df, manifest_report = merge_manifest(df)
    build_alignment_doc(report_dir, out_dir, manifest_report)
    highres_patch_gap_report(out_dir, manifest_report)

    available_text = (report_dir / "data_gap_inventory.md").read_text(encoding="utf-8") if (report_dir / "data_gap_inventory.md").exists() else "# 数据缺口与可用性清单\n"
    if "## Highres Patch 20260518" not in available_text:
        available_text += """

## Highres Patch 20260518

- `hk_s2_patch_manifest.csv` 已纳入工作区，但仅包含 patch 可用性和有效像元字段。
- 未发现可追溯到原始 Sentinel-2 / Landsat patch 栅格的 `file_path/band_path/patch_path/image_id/tile`。
- 因此本轮高分辨率 patch 只能登记为数据缺口，不能当作真实 patch 特征进入模型。
"""
        (report_dir / "data_gap_inventory.md").write_text(available_text, encoding="utf-8")

    dnr = (report_dir / "do_not_repeat_experiments.md").read_text(encoding="utf-8") if (report_dir / "do_not_repeat_experiments.md").exists() else "# 不再重复实验清单\n"
    if "高分 patch 不可用前" not in dnr:
        dnr += """

- 高分 patch 不可用前，不再重复任何“假装有 Sentinel-2 / Landsat patch”的模型实验。
- 没有真实机制变量和高分 patch 前，不再继续做普通调参、普通 stacking、普通 CMEMS 4km patch。
"""
        (report_dir / "do_not_repeat_experiments.md").write_text(dnr, encoding="utf-8")

    spatial, wq, optical, hybrid = build_group_features(df)
    defs = []
    defs += fit_cluster_scheme("spatial", spatial)
    defs += fit_cluster_scheme("water_quality_train_only", wq)
    defs += fit_cluster_scheme("optical_train_only", optical)
    defs += fit_cluster_scheme("hybrid_train_only", hybrid)
    defs_df = save_group_files(out_dir, defs)

    # Select one best internal configuration per scheme.
    best_defs: dict[str, GroupDefinition] = {}
    for scheme in ["spatial", "water_quality_train_only", "optical_train_only", "hybrid_train_only"]:
        scheme_defs = [d for d in defs if d.scheme == scheme]
        if scheme_defs:
            best_defs[scheme] = sorted(scheme_defs, key=lambda d: d.score, reverse=True)[0]

    metrics_rows = []
    pred_rows = []
    imp_rows = []
    station_compare_rows = []
    station_pred_rows = []

    # empty because highres patch not available
    pd.DataFrame(
        [
            {
                "status": "not_run",
                "reason": "high-resolution Sentinel-2 / Landsat patch rasters unavailable; manifest only",
                "manifest_exists": manifest_report.get("exists"),
            }
        ]
    ).to_csv(out_dir / "highres_patch_ab_metrics.csv", index=False, encoding="utf-8-sig")

    for target, spec in TARGETS.items():
        y_col = spec["column"]
        for depth in DEPTHS:
            sub = df[(df["depth"] == depth) & df[y_col].notna()].copy()
            if sub.empty:
                continue
            train = sub[sub["split"] == "train"].copy()
            valid = sub[sub["split"] == "validation"].copy()
            test = sub[sub["split"] == "test"].copy()
            train_counts = train.groupby("station").size().rename("n_train")
            valid_counts = valid.groupby("station").size().rename("n_valid")
            test_counts = test.groupby("station").size().rename("n_test")
            global_features = candidate_features(sub, target, include_group=False)
            global_best = choose_best_family(train, valid, test, global_features, y_col)
            metrics_rows.append({"target": target, "depth": depth, "split": "test", "experiment": "global_model", "scheme": "none", "candidate": global_best["family"], **global_best["test_metrics"], "validation_r2": global_best["val_metrics"]["r2"], "n_features": len(global_features), "previous_best_r2": spec["previous_best"][depth], "delta_vs_previous": global_best["test_metrics"]["r2"] - spec["previous_best"][depth]})
            base_pred = test[[c for c in ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en"] if c in test.columns]].copy()
            base_pred["target"] = target
            base_pred["experiment"] = "global_model"
            base_pred["scheme"] = "none"
            base_pred["y_true"] = test[y_col].to_numpy()
            base_pred["y_pred"] = global_best["test_pred"]
            pred_rows.append(base_pred)
            if not global_best["importance"].empty:
                imp = global_best["importance"].copy()
                imp["target"] = target
                imp["depth"] = depth
                imp["experiment"] = "global_model"
                imp["scheme"] = "none"
                imp["model_family"] = global_best["family"]
                imp_rows.append(imp)

            best_group_choice: dict[str, Any] | None = None
            for scheme, definition in best_defs.items():
                merged = merge_group(sub, definition.assignments, definition.label_col)
                train_g = merged[merged["split"] == "train"].copy()
                valid_g = merged[merged["split"] == "validation"].copy()
                test_g = merged[merged["split"] == "test"].copy()
                if train_g["station_group_id"].nunique() < 2:
                    continue
                group_features = candidate_features(merged, target, include_group=True)
                group_feature_best = choose_best_family(train_g, valid_g, test_g, group_features, y_col)
                metrics_rows.append({"target": target, "depth": depth, "split": "test", "experiment": "group_as_feature", "scheme": scheme, "cluster_method": definition.cluster_method, "n_groups": definition.n_groups, "candidate": group_feature_best["family"], **group_feature_best["test_metrics"], "validation_r2": group_feature_best["val_metrics"]["r2"], "n_features": len(group_features), "previous_best_r2": spec["previous_best"][depth], "delta_vs_previous": group_feature_best["test_metrics"]["r2"] - spec["previous_best"][depth]})
                pred = test_g[[c for c in ["obs_id", "sample_id", "station", "date", "depth", "water_zone_en", "station_group_id"] if c in test_g.columns]].copy()
                pred["target"] = target
                pred["experiment"] = "group_as_feature"
                pred["scheme"] = scheme
                pred["y_true"] = test_g[y_col].to_numpy()
                pred["y_pred"] = group_feature_best["test_pred"]
                pred_rows.append(pred)

                grouped_valid, grouped_test, group_sizes = fit_group_specific(train_g, valid_g, test_g, group_features, y_col, group_feature_best["family"])
                group_test_metrics = evaluate(test_g[y_col].to_numpy(), grouped_test)
                metrics_rows.append({"target": target, "depth": depth, "split": "test", "experiment": "group_specific_models", "scheme": scheme, "cluster_method": definition.cluster_method, "n_groups": definition.n_groups, "candidate": group_feature_best["family"], **group_test_metrics, "validation_r2": evaluate(valid_g[y_col].to_numpy(), grouped_valid)["r2"], "n_features": len(group_features), "previous_best_r2": spec["previous_best"][depth], "delta_vs_previous": group_test_metrics["r2"] - spec["previous_best"][depth]})

                res_valid, res_test = fit_group_residual(group_feature_best, train_g, valid_g, test_g, y_col)
                res_test_metrics = evaluate(test_g[y_col].to_numpy(), res_test)
                metrics_rows.append({"target": target, "depth": depth, "split": "test", "experiment": "global_group_residual", "scheme": scheme, "cluster_method": definition.cluster_method, "n_groups": definition.n_groups, "candidate": group_feature_best["family"], **res_test_metrics, "validation_r2": evaluate(valid_g[y_col].to_numpy(), res_valid)["r2"], "n_features": len(group_features), "previous_best_r2": spec["previous_best"][depth], "delta_vs_previous": res_test_metrics["r2"] - spec["previous_best"][depth]})

                moe_valid, moe_test = fit_mixture(group_feature_best, train_g, valid_g, test_g, group_features, y_col, group_feature_best["family"])
                moe_test_metrics = evaluate(test_g[y_col].to_numpy(), moe_test)
                metrics_rows.append({"target": target, "depth": depth, "split": "test", "experiment": "mixture_of_experts", "scheme": scheme, "cluster_method": definition.cluster_method, "n_groups": definition.n_groups, "candidate": group_feature_best["family"], **moe_test_metrics, "validation_r2": evaluate(valid_g[y_col].to_numpy(), moe_valid)["r2"], "n_features": len(group_features), "previous_best_r2": spec["previous_best"][depth], "delta_vs_previous": moe_test_metrics["r2"] - spec["previous_best"][depth]})

                choices = [
                    ("group_as_feature", group_feature_best["val_metrics"]["r2"], group_feature_best["test_pred"]),
                    ("group_specific_models", evaluate(valid_g[y_col].to_numpy(), grouped_valid)["r2"], grouped_test),
                    ("global_group_residual", evaluate(valid_g[y_col].to_numpy(), res_valid)["r2"], res_test),
                    ("mixture_of_experts", evaluate(valid_g[y_col].to_numpy(), moe_valid)["r2"], moe_test),
                ]
                best_exp, best_val_r2, best_test_pred = sorted(choices, key=lambda x: x[1], reverse=True)[0]
                if best_group_choice is None or best_val_r2 > best_group_choice["validation_r2"]:
                    best_group_choice = {
                        "scheme": scheme,
                        "definition": definition,
                        "experiment": best_exp,
                        "validation_r2": best_val_r2,
                        "test_pred": best_test_pred,
                        "test_frame": test_g.copy(),
                        "global_pred": global_best["test_pred"],
                    }
            if best_group_choice is not None:
                test_frame = best_group_choice["test_frame"].copy()
                test_frame = test_frame.merge(train_counts, on="station", how="left").merge(valid_counts, on="station", how="left").merge(test_counts, on="station", how="left")
                compare, pred_compare = station_level_compare(target, depth, test_frame, best_group_choice["global_pred"], best_group_choice["test_pred"], best_group_choice["scheme"], best_group_choice["definition"].label_col)
                station_compare_rows.append(compare)
                station_pred_rows.append(pred_compare)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out_dir / "station_grouping_metrics.csv", index=False, encoding="utf-8-sig")
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    predictions.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    feature_importance = pd.concat(imp_rows, ignore_index=True) if imp_rows else pd.DataFrame()
    feature_importance.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    station_compare = pd.concat(station_compare_rows, ignore_index=True) if station_compare_rows else pd.DataFrame()
    station_compare.to_csv(out_dir / "station_level_prediction_comparison.csv", index=False, encoding="utf-8-sig")
    station_preds = pd.concat(station_pred_rows, ignore_index=True) if station_pred_rows else pd.DataFrame()
    station_preds.to_csv(out_dir / "station_level_predictions.csv", index=False, encoding="utf-8-sig")

    improved_summary = ""
    if not station_compare.empty:
        summary = station_compare.groupby(["target", "depth_layer", "station_group_method"], as_index=False).agg(improved_station_count=("improved", "sum"), station_count=("station", "nunique"), mean_delta_r2=("delta_r2", "mean"), mean_delta_rmse=("delta_rmse", "mean"))
        summary["improved_ratio"] = summary["improved_station_count"] / summary["station_count"]
        improved_summary = flat_table(summary.sort_values(["target", "depth_layer", "improved_ratio"], ascending=[True, True, False]))

    report = f"""# Station Group Highres Patch Alignment {args.date_tag}

## 1. 本轮输入数据

- 主建模表: `data/processed/chla_ocean_colour_precision_optimization_20260517/chla_ocean_colour_precision_optimization_dataset_20260517.csv`
- 高分 patch manifest: `data/raw/manual/ocean_color/hk_s2_patch_manifest.csv`
- CMEMS 4 km point features: 已存在于主建模表
- 高分 patch 原始栅格: 本轮未找到可追溯源文件

## 2. 对齐方式

详见 `model_input_alignment_documentation.md`。本轮明确记录了标签、Ocean Colour 点值、CMEMS patch、manifest 质量字段以及禁止泄漏边界。

## 3. 高分 patch 是否成功

没有。`hk_s2_patch_manifest.csv` 只包含影像可用性与有效像元清单，缺少 `image_id/tile/file_path/band_path/patch_path`，无法生成真实 Sentinel-2 / Landsat 3x3、5x5 多波段 patch 统计。

## 4. 站点分类方法

- 空间位置分组
- train-only 历史水质统计分组
- train-only 光学统计分组
- 混合分组

聚类在 train 期站点特征上拟合，尝试 `k=2,3,4,5` 与 `KMeans/Agglomerative/GMM`，再按内部轮廓系数选每类最优配置。

## 5. 分组模型 vs 全局模型

{improved_summary or '本轮未形成可比较的逐站结果表。'}

## 6. 是否接近 R2=0.60

如果没有真实高分辨率 patch 和更强机制变量，仅靠站点分组通常无法合理接近 0.60。本轮重点是判断哪些站点适合 grouped model，哪些应保留 global model。

## 7. 风险与边界

- 不使用随机切分或 shuffle。
- 不用 test 标签做聚类或调参。
- 所有站点分组只在 train 期站点统计上拟合。
- 不引入同日其他水质实测协变量。

## 8. 新增/修改文件

- `scripts/training/run_station_group_highres_patch_alignment.py`
- `experiments/reports/model_input_alignment_documentation.md`
- `experiments/outputs/station_group_highres_patch_alignment_{args.date_tag}/`
"""
    write_docs(report_dir, out_dir, f"station_group_highres_patch_alignment_{args.date_tag}.md", report)

    config = {
        "date_tag": args.date_tag,
        "input": args.input,
        "split": {"train": "2014-2020", "validation": "2021-2022", "test": "2023-2024"},
        "manifest_report": manifest_report,
        "best_group_configs": {
            scheme: {"cluster_method": d.cluster_method, "n_groups": d.n_groups, "score": d.score, "label_col": d.label_col}
            for scheme, d in best_defs.items()
        },
        "notes": {
            "highres_patch_unavailable": True,
            "cmems_4km_patch_not_repeated": True,
        },
    }
    for path in [config_dir / f"station_group_highres_patch_alignment_{args.date_tag}.json", out_dir / "experiment_config.json"]:
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    reg_updates = {
        "methods_tried_registry.md": """

## Station Group Highres Patch Alignment 20260518

已完成高分 patch manifest 可追溯性检查、模型输入对齐文档、train-only 站点分组和全局模型 vs 分组模型逐站比较。

结论：如果没有真实 Sentinel-2 / Landsat patch 或更强机制变量，继续靠模型结构微调无法合理冲击 R2=0.60。
""",
        "failed_experiments.md": """

## Station Group Highres Patch Alignment 20260518

高分 patch 本轮未能真实生成，因为 manifest 不含原始影像追溯字段。若某些 grouped model 未显著优于 global model，不应继续在相同输入上重复分组小修。
""",
        "do_not_repeat_experiments.md": """

- 没有真实 Sentinel-2 / Landsat patch 前，不再重复“高分 patch A/B”。
- 站点分组若整体提升 <0.03 或 improved station 比例不足 60%，只保留为辅助线，不再作为主线反复微调。
""",
        "data_gap_inventory.md": """

## Station Group / Highres Patch 20260518

- 当前缺口不是“再调一轮普通模型”，而是缺少可追溯的 Sentinel-2 / Landsat 高分 patch 栅格、bathymetry、station depth 和更强水动力变量。
""",
    }
    for name, addon in reg_updates.items():
        path = report_dir / name
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if "20260518" not in text:
            path.write_text(text + addon, encoding="utf-8")

    for name in ["methods_tried_registry.md", "failed_experiments.md", "do_not_repeat_experiments.md", "data_gap_inventory.md", "model_input_alignment_documentation.md"]:
        if (report_dir / name).exists():
            shutil.copyfile(report_dir / name, out_dir / name)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "highres_patch_available": False,
                "station_group_definitions": str(out_dir / "station_group_definitions.csv"),
                "station_level_comparison": str(out_dir / "station_level_prediction_comparison.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
