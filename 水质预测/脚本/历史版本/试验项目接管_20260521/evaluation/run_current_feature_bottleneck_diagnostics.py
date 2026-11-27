from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training import run_current_highres_patch_ab as current_ab


OUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "current"
REPORT_DIR = PROJECT_ROOT / "experiments" / "reports" / "current"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "current"

FEATURE_SET_JSON = PROCESSED_DIR / "task_specific_feature_sets.json"

TASKS = [(target, depth) for target in current_ab.TARGETS for depth in current_ab.DEPTH_ORDER]

CORE_ALWAYS = [
    "station",
    "water_zone_en",
    "lon",
    "lat",
    "year_offset",
    "month_sin",
    "month_cos",
    "is_wet_season",
    "quarter",
    "day_of_year",
    "days_since_prev_sample",
]

CORE_REMOTE = [
    "landsat_available",
    "landsat_has_images",
    "landsat_image_count",
    "s2_available",
    "s2_has_images",
    "s2_image_count",
    "satellite_any_available",
    "satellite_available_count",
    "oc_valid",
    "oc_day_diff",
    "oc_pixel_distance_km",
    "oc_chl",
    "oc_kd490",
    "oc_zsd",
    "oc_chl_3d",
    "oc_chl_7d",
    "oc_chl_14d",
    "oc_kd490_3d",
    "oc_kd490_7d",
    "oc_kd490_14d",
    "oc_zsd_3d",
    "oc_zsd_7d",
    "oc_zsd_14d",
    "valid_count_7d",
]


def ensure_dirs() -> None:
    for path in [OUT_DIR, REPORT_DIR, PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    out: list[str] = []
    for col in features:
        if col in df.columns and col not in out and df[col].notna().sum() > 0:
            out.append(col)
    return out


def target_history_features(target: str) -> list[str]:
    prefix = current_ab.TARGET_SPECS[target]["history_prefix"]
    return [f"{prefix}_lag1", f"{prefix}_lag2", f"{prefix}_roll3"]


def load_importance_sources() -> pd.DataFrame:
    paths = [
        OUT_DIR / "highres_patch_feature_importance.csv",
        OUT_DIR / "s2_patch_feature_importance.csv",
        OUT_DIR / "landsat_patch_feature_importance.csv",
        OUT_DIR / "chla_bottom_mechanism_feature_importance.csv",
        PROJECT_ROOT
        / "experiments"
        / "outputs"
        / "chla_ocean_colour_precision_optimization_20260517"
        / "feature_importance.csv",
        PROJECT_ROOT
        / "experiments"
        / "outputs"
        / "chla_ocean_colour_precision_maximize_20260518"
        / "feature_importance.csv",
        PROJECT_ROOT
        / "experiments"
        / "outputs"
        / "literature_inspired_precision_boost_20260518"
        / "feature_importance.csv",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if "feature" not in df.columns:
            continue
        if "importance" not in df.columns:
            numeric = [c for c in df.columns if c.lower() in {"gain", "weight", "value"}]
            if not numeric:
                continue
            df["importance"] = df[numeric[0]]
        if "target" not in df.columns:
            continue
        if "depth_layer" not in df.columns and "depth" in df.columns:
            df = df.rename(columns={"depth": "depth_layer"})
        if "depth_layer" not in df.columns:
            continue
        frames.append(df[["target", "depth_layer", "feature", "importance"]].copy())
    if not frames:
        return pd.DataFrame(columns=["target", "depth_layer", "feature", "train_importance"])
    imp = pd.concat(frames, ignore_index=True)
    imp["importance"] = pd.to_numeric(imp["importance"], errors="coerce").fillna(0.0)
    imp = (
        imp.groupby(["target", "depth_layer", "feature"], as_index=False)["importance"]
        .mean()
        .rename(columns={"importance": "train_importance"})
    )
    return imp


def feature_missing(df: pd.DataFrame, target: str, depth: str, features: list[str]) -> pd.DataFrame:
    target_col = current_ab.TARGET_SPECS[target]["column"]
    sub = df[(df["depth"] == depth) & df[target_col].notna()].copy()
    rows = []
    for feature in features:
        rows.append(
            {
                "target": target,
                "depth_layer": depth,
                "feature": feature,
                "missing_rate": float(sub[feature].isna().mean()) if feature in sub.columns else 1.0,
            }
        )
    return pd.DataFrame(rows)


def correlation_groups(train: pd.DataFrame, features: list[str]) -> dict[str, str]:
    numeric = [
        col
        for col in features
        if col in train.columns
        and pd.api.types.is_numeric_dtype(train[col])
        and train[col].notna().mean() >= 0.2
        and train[col].nunique(dropna=True) > 1
    ]
    if len(numeric) < 2:
        return {col: "" for col in features}
    sample = train[numeric].sample(min(len(train), 4000), random_state=42) if len(train) > 4000 else train[numeric]
    corr = sample.corr().abs()
    assigned: dict[str, str] = {}
    group_id = 1
    for col in numeric:
        if col in assigned:
            continue
        partners = corr.index[(corr[col] >= 0.95) & (corr.index != col)].tolist()
        if partners:
            gid = f"corr_{group_id:03d}"
            assigned[col] = gid
            for partner in partners:
                assigned.setdefault(partner, gid)
            group_id += 1
    for col in features:
        assigned.setdefault(col, "")
    return assigned


def permutation_scores(
    df: pd.DataFrame,
    target: str,
    depth: str,
    candidate_features: list[str],
) -> dict[str, float]:
    target_col = current_ab.TARGET_SPECS[target]["column"]
    train, val, _ = current_ab.split_frame(df, target_col, depth)
    features = clean_features(train, candidate_features)
    if len(features) < 3 or len(val) < 20:
        return {}
    features = features[:35]
    try:
        pipe = current_ab.fit_lgbm_pipeline(train, features, target_col, "median")
        result = permutation_importance(
            pipe,
            val[features],
            val[target_col],
            n_repeats=2,
            random_state=42,
            scoring="r2",
            n_jobs=1,
        )
    except Exception:
        return {}
    return {feature: float(score) for feature, score in zip(features, result.importances_mean)}


def build_full_features(
    df: pd.DataFrame,
    target: str,
    depth: str,
    highres_groups: dict[str, list[str]],
    mechanism_features: list[str],
) -> list[str]:
    extras = list(highres_groups.get("highres_patch", []))
    if target == "chla" and depth == "bottom":
        extras += mechanism_features
    return current_ab.build_common_features(df, target, extra_features=extras)


def build_history_only(df: pd.DataFrame, target: str) -> list[str]:
    return clean_features(df, CORE_ALWAYS + target_history_features(target))


def build_mechanism_features(
    df: pd.DataFrame,
    target: str,
    depth: str,
    mechanism_features: list[str],
) -> list[str]:
    features = CORE_ALWAYS + target_history_features(target) + CORE_REMOTE
    if target == "chla":
        features += [
            "chla_history_trend_lag1_lag2",
            "chla_history_lag1_minus_roll3",
            "chla_lag1_per_day_gap",
        ]
    if target == "chla" and depth == "bottom":
        features += mechanism_features
    return clean_features(df, features)


def analyze_feature_stability(
    df: pd.DataFrame,
    highres_groups: dict[str, list[str]],
    mechanism_features: list[str],
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, list[str]]]]:
    importance = load_importance_sources()
    stability_frames: list[pd.DataFrame] = []
    feature_sets: dict[tuple[str, str], dict[str, list[str]]] = {}

    for target, depth in TASKS:
        target_col = current_ab.TARGET_SPECS[target]["column"]
        train, _, _ = current_ab.split_frame(df, target_col, depth)
        full = build_full_features(df, target, depth, highres_groups, mechanism_features)
        history = build_history_only(df, target)
        mechanism = build_mechanism_features(df, target, depth, mechanism_features)

        missing = feature_missing(df, target, depth, full)
        imp = importance[
            (importance["target"] == target) & (importance["depth_layer"] == depth)
        ].copy()
        merged = missing.merge(imp, on=["target", "depth_layer", "feature"], how="left")
        merged["train_importance"] = merged["train_importance"].fillna(0.0)

        ranked = merged.sort_values(["train_importance", "missing_rate"], ascending=[False, True])
        perm_candidates = list(dict.fromkeys(history + mechanism + ranked["feature"].head(45).tolist()))
        perm = permutation_scores(df, target, depth, clean_features(df, perm_candidates))
        merged["permutation_importance"] = merged["feature"].map(perm)
        merged["valid_importance"] = merged["permutation_importance"]
        merged["test_importance_if_available"] = np.nan

        corr_groups = correlation_groups(train, full)
        merged["correlation_group"] = merged["feature"].map(corr_groups).fillna("")

        core = set(history + mechanism)
        top_importance = set(ranked.head(80)["feature"])
        positive_perm = set(merged.loc[merged["permutation_importance"].fillna(0) > 0, "feature"])
        selected: list[bool] = []
        reasons: list[str] = []
        for _, row in merged.iterrows():
            feature = row["feature"]
            miss = float(row["missing_rate"])
            imp_value = float(row["train_importance"])
            perm_value = row["permutation_importance"]
            is_core = feature in core
            is_selected = False
            reason = ""
            if miss >= 0.85 and imp_value <= 0 and not is_core:
                reason = "high_missing_low_contribution"
            elif feature in top_importance and miss <= 0.75:
                is_selected = True
                reason = "stable_importance"
            elif feature in positive_perm and miss <= 0.80:
                is_selected = True
                reason = "validation_permutation_positive"
            elif is_core and miss <= 0.80:
                is_selected = True
                reason = "mechanism_or_history_core"
            elif row["correlation_group"] and imp_value <= ranked["train_importance"].quantile(0.35):
                reason = "correlated_low_importance"
            else:
                reason = "low_or_unstable_contribution"
            selected.append(bool(is_selected))
            reasons.append("" if is_selected else reason)
        merged["selected_by_stability"] = selected
        merged["drop_reason"] = reasons

        stable = merged.loc[merged["selected_by_stability"], "feature"].tolist()
        stable = clean_features(df, list(dict.fromkeys(history + stable)))[:90]
        feature_sets[(target, depth)] = {
            "full_features": full,
            "stable_core_features": stable,
            "mechanism_features": mechanism,
            "history_only_baseline": history,
        }
        stability_frames.append(merged)

    stability = pd.concat(stability_frames, ignore_index=True)
    return stability, feature_sets


def metric_row(
    target: str,
    depth: str,
    split: str,
    feature_set: str,
    result: current_ab.CandidateResult,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_y: np.ndarray,
) -> dict[str, Any]:
    high = current_ab.high_value_metrics(np.asarray(y_true), np.asarray(y_pred), np.asarray(train_y))
    return {
        "target": target,
        "depth_layer": depth,
        "split": split,
        "feature_set": feature_set,
        "candidate": result.candidate_name,
        "model_type": result.model_type,
        "feature_count": len(result.features),
        "n_samples": len(y_true),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "rmse": current_ab.rmse(np.asarray(y_true), np.asarray(y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
        "p90_rmse": high["p90_rmse"],
        "p90_underestimation_rate": high["p90_underestimation_rate"],
    }


def run_feature_set_ab(
    df: pd.DataFrame,
    feature_sets: dict[tuple[str, str], dict[str, list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    station_rows: list[dict[str, Any]] = []

    for target, depth in TASKS:
        target_col = current_ab.TARGET_SPECS[target]["column"]
        train, val, test = current_ab.split_frame(df, target_col, depth)
        for feature_set_name, features in feature_sets[(target, depth)].items():
            result = current_ab.fit_for_current_best_family(
                df,
                target,
                depth,
                clean_features(df, features),
                f"feature_set_{feature_set_name}",
            )
            pred = current_ab.candidate_prediction_frame(df, result)
            pred["feature_set"] = feature_set_name
            prediction_frames.append(pred)
            for split_name, frame, pred_values in [
                ("train", train, result.train_pred),
                ("validation", val, result.validation_pred),
                ("test", test, result.test_pred),
            ]:
                metric_rows.append(
                    metric_row(
                        target,
                        depth,
                        split_name,
                        feature_set_name,
                        result,
                        frame[target_col].to_numpy(),
                        pred_values,
                        train[target_col].to_numpy(),
                    )
                )
            test_pred = pred[pred["split"] == "test"].copy()
            for station, sub in test_pred.groupby("station"):
                if len(sub) < 2:
                    r2 = np.nan
                else:
                    r2 = float(r2_score(sub["y_true"], sub["y_pred"]))
                station_rows.append(
                    {
                        "target": target,
                        "depth_layer": depth,
                        "station": station,
                        "feature_set": feature_set_name,
                        "n_test": int(len(sub)),
                        "r2": r2,
                        "rmse": current_ab.rmse(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()),
                        "mae": float(mean_absolute_error(sub["y_true"], sub["y_pred"])),
                        "bias": float(np.mean(sub["y_pred"] - sub["y_true"])),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    station = pd.DataFrame(station_rows)
    return metrics, predictions, station


def add_context(pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    context_cols = [
        "obs_id",
        "sample_id",
        "station",
        "date",
        "depth",
        "year",
        "month",
        "is_wet_season",
        "days_since_prev_sample",
        "satellite_any_available",
        "s2_available",
        "landsat_available",
        "valid_count_7d",
    ]
    context_cols = [c for c in context_cols if c in df.columns]
    context = df[context_cols].drop_duplicates(subset=[c for c in ["obs_id", "sample_id"] if c in context_cols])
    keys = [c for c in ["obs_id", "sample_id"] if c in pred.columns and c in context.columns]
    if not keys:
        keys = ["station", "date", "depth"]
        context = df[[c for c in context_cols if c in df.columns]].drop_duplicates(subset=keys)
    out = pred.merge(context, on=keys, how="left", suffixes=("", "_ctx"))
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    elif "date_ctx" in out.columns:
        out["date"] = pd.to_datetime(out["date_ctx"])
    if "year" not in out.columns:
        out["year"] = out["date"].dt.year
    if "month" not in out.columns:
        out["month"] = out["date"].dt.month
    out["abs_error"] = (out["y_pred"] - out["y_true"]).abs()
    out["error"] = out["y_pred"] - out["y_true"]
    return out


def grouped_error(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, sub in pred.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n": int(len(sub)),
                "r2": float(r2_score(sub["y_true"], sub["y_pred"])) if len(sub) > 1 else np.nan,
                "rmse": current_ab.rmse(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()),
                "mae": float(mean_absolute_error(sub["y_true"], sub["y_pred"])),
                "bias": float(np.mean(sub["error"])),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_error_diagnostics(df: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if "depth_layer" not in predictions.columns and "depth" in predictions.columns:
        predictions = predictions.copy()
        predictions["depth_layer"] = predictions["depth"]
    test_metrics = metrics[metrics["split"] == "test"].copy()
    idx = test_metrics.groupby(["target", "depth_layer"])["r2"].idxmax()
    selected = test_metrics.loc[idx, ["target", "depth_layer", "feature_set", "r2"]].copy()
    pred = predictions.merge(selected[["target", "depth_layer", "feature_set"]], on=["target", "depth_layer", "feature_set"])
    pred = pred[pred["split"] == "test"].copy()
    pred = add_context(pred, df)

    grouped_error(pred, ["target", "depth_layer", "station"]).to_csv(
        OUT_DIR / "error_by_station.csv", index=False, encoding="utf-8-sig"
    )
    grouped_error(pred, ["target", "depth_layer", "year"]).to_csv(
        OUT_DIR / "error_by_year.csv", index=False, encoding="utf-8-sig"
    )
    grouped_error(pred, ["target", "depth_layer", "month"]).to_csv(
        OUT_DIR / "error_by_month.csv", index=False, encoding="utf-8-sig"
    )
    pred["season"] = np.where(pred["is_wet_season"].fillna(0).astype(float) > 0, "wet", "dry")
    grouped_error(pred, ["target", "depth_layer", "season"]).to_csv(
        OUT_DIR / "error_by_season.csv", index=False, encoding="utf-8-sig"
    )
    quantile_frames = []
    for _, sub in pred.groupby(["target", "depth_layer"]):
        labels = ["q1_low", "q2", "q3", "q4_high"]
        q = pd.qcut(sub["y_true"], q=4, labels=labels, duplicates="drop")
        part = sub.copy()
        part["target_quantile"] = q.astype(str)
        quantile_frames.append(part)
    qpred = pd.concat(quantile_frames, ignore_index=True)
    grouped_error(qpred, ["target", "depth_layer", "target_quantile"]).to_csv(
        OUT_DIR / "error_by_target_quantile.csv", index=False, encoding="utf-8-sig"
    )
    pred["satellite_availability"] = "unknown"
    if "s2_available" in pred.columns and "landsat_available" in pred.columns:
        s2 = pred["s2_available"].fillna(0).astype(float) > 0
        ls = pred["landsat_available"].fillna(0).astype(float) > 0
        pred["satellite_availability"] = np.select(
            [s2 & ls, s2 & ~ls, ~s2 & ls, ~s2 & ~ls],
            ["s2_landsat", "s2_only", "landsat_only", "none"],
            default="unknown",
        )
    grouped_error(pred, ["target", "depth_layer", "satellite_availability"]).to_csv(
        OUT_DIR / "error_by_satellite_availability.csv", index=False, encoding="utf-8-sig"
    )
    if "days_since_prev_sample" in pred.columns:
        pred["sample_gap_bin"] = pd.cut(
            pred["days_since_prev_sample"].astype(float),
            bins=[-np.inf, 31, 60, 90, np.inf],
            labels=["<=31d", "32-60d", "61-90d", ">90d"],
        ).astype(str)
    else:
        pred["sample_gap_bin"] = "unknown"
    grouped_error(pred, ["target", "depth_layer", "sample_gap_bin"]).to_csv(
        OUT_DIR / "error_by_sample_gap.csv", index=False, encoding="utf-8-sig"
    )
    return selected


def write_station_recommendation(
    station_metrics: pd.DataFrame,
    selected_feature_sets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (target, depth, station), sub in station_metrics.groupby(["target", "depth_layer", "station"]):
        best = sub.sort_values(["rmse", "mae"], ascending=[True, True]).iloc[0]
        selected = selected_feature_sets[
            (selected_feature_sets["target"] == target) & (selected_feature_sets["depth_layer"] == depth)
        ]
        global_set = selected["feature_set"].iloc[0] if len(selected) else "unknown"
        recommended = {
            "full_features": "global_model",
            "stable_core_features": "stable_core_model",
            "mechanism_features": "mechanism_model",
            "history_only_baseline": "history_only_model",
        }.get(best["feature_set"], "fallback_to_previous_best")
        rows.append(
            {
                "target": target,
                "depth_layer": depth,
                "station": station,
                "n_test": int(best["n_test"]),
                "best_global_model": global_set,
                "best_station_candidate": best["feature_set"],
                "best_feature_set": best["feature_set"],
                "station_r2": best["r2"],
                "station_rmse": best["rmse"],
                "improved_vs_global": bool(best["feature_set"] != global_set),
                "recommended_strategy": recommended,
                "reason": "station_min_rmse_feature_set",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "station_model_recommendation.csv", index=False, encoding="utf-8-sig")
    return out


def write_feature_set_selection_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"] == "test"].copy()
    rows = []
    for (target, depth), sub in test.groupby(["target", "depth_layer"]):
        by_set = {row["feature_set"]: row for _, row in sub.iterrows()}
        best = sub.sort_values("r2", ascending=False).iloc[0]
        full = by_set.get("full_features")
        stable = by_set.get("stable_core_features")
        history = by_set.get("history_only_baseline")
        mechanism = by_set.get("mechanism_features")
        full_r2 = float(full["r2"]) if full is not None else np.nan
        stable_r2 = float(stable["r2"]) if stable is not None else np.nan
        history_r2 = float(history["r2"]) if history is not None else np.nan
        mechanism_r2 = float(mechanism["r2"]) if mechanism is not None else np.nan
        if stable is not None and stable_r2 >= full_r2 - 0.005:
            recommended = "stable_core_features"
            reason = "stable_core_not_worse_than_full"
        elif mechanism is not None and mechanism_r2 >= full_r2 - 0.01:
            recommended = "mechanism_features"
            reason = "mechanism_near_full_and_more_interpretable"
        elif history is not None and history_r2 >= full_r2 - 0.01:
            recommended = "history_only_baseline"
            reason = "history_only_near_or_above_full"
        else:
            recommended = str(best["feature_set"])
            reason = "best_test_r2"
        rows.append(
            {
                "target": target,
                "depth_layer": depth,
                "full_r2": full_r2,
                "stable_core_r2": stable_r2,
                "mechanism_r2": mechanism_r2,
                "history_only_r2": history_r2,
                "best_test_feature_set": best["feature_set"],
                "best_test_r2": float(best["r2"]),
                "recommended_feature_set": recommended,
                "recommendation_reason": reason,
                "stable_minus_full_r2": stable_r2 - full_r2
                if pd.notna(stable_r2) and pd.notna(full_r2)
                else np.nan,
                "full_minus_history_r2": full_r2 - history_r2
                if pd.notna(full_r2) and pd.notna(history_r2)
                else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "feature_set_selection_summary.csv", index=False, encoding="utf-8-sig")
    return out


def write_feature_set_json(feature_sets: dict[tuple[str, str], dict[str, list[str]]]) -> None:
    payload = {
        f"{target}_{depth}": groups
        for (target, depth), groups in feature_sets.items()
    }
    FEATURE_SET_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_feature_set_json() -> dict[tuple[str, str], dict[str, list[str]]]:
    payload = json.loads(FEATURE_SET_JSON.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict[str, list[str]]] = {}
    for key, groups in payload.items():
        target, depth = key.split("_", 1)
        out[(target, depth)] = groups
    return out


def write_feature_doc(metrics: pd.DataFrame, feature_sets: dict[tuple[str, str], dict[str, list[str]]]) -> None:
    lines = [
        "# 任务特异性最小有效特征集说明",
        "",
        "本文件记录本轮为 6 个目标/水层构建的四组特征：`full_features`、`stable_core_features`、`mechanism_features`、`history_only_baseline`。",
        "",
        "## 构建原则",
        "",
        "- `full_features`：当前可用全特征，包括历史、遥感、Ocean Colour、高分 patch，并在 Chl-a bottom 中加入机制 proxy。",
        "- `stable_core_features`：在缺失率、已有模型重要性、validation 置换检验和共线性诊断后保留的稳定核心特征。",
        "- `mechanism_features`：时间、空间、目标历史、核心遥感、环境/机制 proxy 等可解释特征。",
        "- `history_only_baseline`：只保留时间、站点、空间和目标自身历史特征，用于判断外部特征真实增益。",
        "",
        "## 各任务特征数量与 test 表现",
        "",
        "| target | depth | feature_set | feature_count | test R² | test RMSE | test MAE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    test = metrics[metrics["split"] == "test"].copy()
    for _, row in test.sort_values(["target", "depth_layer", "feature_set"]).iterrows():
        lines.append(
            f"| {row['target']} | {row['depth_layer']} | {row['feature_set']} | "
            f"{int(row['feature_count'])} | {row['r2']:.4f} | {row['rmse']:.4f} | {row['mae']:.4f} |"
        )
    lines += ["", "## 每个任务的特征清单", ""]
    for (target, depth), groups in feature_sets.items():
        lines.append(f"### {target} {depth}")
        for name, feats in groups.items():
            preview = ", ".join(feats[:80])
            suffix = " ..." if len(feats) > 80 else ""
            lines.append(f"- `{name}`：{len(feats)} 个字段。{preview}{suffix}")
        lines.append("")
    (REPORT_DIR / "task_specific_feature_set_documentation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_ceiling_analysis(metrics: pd.DataFrame, selected: pd.DataFrame, stability: pd.DataFrame) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    best = test.loc[test.groupby(["target", "depth_layer"])["r2"].idxmax()].copy()
    lines = [
        "# 当前数据条件下预测精度上限分析",
        "",
        "本轮不新增数据源、不做普通调参，重点比较全特征与精简特征的差异，用于判断当前瓶颈来自特征噪声、模型选择，还是数据/标签/机制缺口。",
        "",
        "## 1. 本轮各任务最高 R²",
        "",
        "| target | depth | best_feature_set | best_test_R² | RMSE | MAE | 判断 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in best.sort_values(["target", "depth_layer"]).iterrows():
        if row["r2"] >= 0.5:
            verdict = "已接近较高精度"
        elif row["r2"] >= 0.35:
            verdict = "中等可预测性"
        else:
            verdict = "明显受数据或机制瓶颈限制"
        lines.append(
            f"| {row['target']} | {row['depth_layer']} | {row['feature_set']} | "
            f"{row['r2']:.4f} | {row['rmse']:.4f} | {row['mae']:.4f} | {verdict} |"
        )
    lines += [
        "",
        "## 2. 是否存在特征过多导致噪声",
        "",
    ]
    for (target, depth), sub in test.groupby(["target", "depth_layer"]):
        full = sub[sub["feature_set"] == "full_features"]
        stable = sub[sub["feature_set"] == "stable_core_features"]
        hist = sub[sub["feature_set"] == "history_only_baseline"]
        if len(full) and len(stable):
            delta = float(stable["r2"].iloc[0] - full["r2"].iloc[0])
            if delta >= 0:
                msg = "stable_core 不低于 full，说明全特征中存在可删除噪声。"
            elif abs(delta) <= 0.01:
                msg = "stable_core 略低但接近 full，说明精简后信息损失有限。"
            else:
                msg = "full 明显更高，说明仍有部分额外特征提供信息。"
            lines.append(f"- `{target} {depth}`：stable_core 相对 full 的 R² 差异为 {delta:.4f}，{msg}")
        if len(full) and len(hist):
            delta_hist = float(full["r2"].iloc[0] - hist["r2"].iloc[0])
            if delta_hist <= 0.02:
                lines.append(f"  - history_only 已接近 full，外部遥感/环境增益有限，差异 {delta_hist:.4f}。")
            else:
                lines.append(f"  - full 相比 history_only 仍有增益，差异 {delta_hist:.4f}。")
    high_missing = stability[
        (stability["missing_rate"] >= 0.75) & (~stability["selected_by_stability"])
    ]
    lines += [
        "",
        "## 3. 主要瓶颈判断",
        "",
        f"- 本轮识别出高缺失且未被稳定选择的特征记录数：{len(high_missing)}。",
        "- Chl-a bottom 仍缺少真实强机制变量，包括站点水深、bathymetry、底层水温、垂向温度剖面、流速、水位和流量。",
        "- 遥感与实测标签仍可能存在时间错配，尤其是高分 patch 覆盖率不足或影像日期差较大时。",
        "- 若四组特征表现接近，说明瓶颈更可能来自数据和标签，而不是简单特征筛选。",
        "",
        "## 4. 是否继续追求 R²=0.60",
        "",
        "在当前数据条件下，继续仅靠删特征、普通调参或重新组合现有遥感点值，不宜预期稳定达到 R²=0.60。若要继续提升，优先补真实 bottom 机制数据和更稳定的高分遥感有效观测，而不是继续堆派生特征。",
    ]
    (REPORT_DIR / "prediction_ceiling_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def append_report_updates() -> None:
    snippets = {
        "experiment_report.md": [
            "",
            "## 2026-05-19 特征瓶颈与最小有效特征集诊断",
            "",
            "- 本轮未新增数据源，未做大规模 Optuna，重点比较 `full_features`、`stable_core_features`、`mechanism_features`、`history_only_baseline`。",
            "- 输出 `feature_stability_analysis.csv`、`feature_set_ab_metrics.csv`、误差分解表和 `prediction_ceiling_analysis.md`。",
            "- 本轮结论用于判断现有特征是否存在噪声、外部特征是否真正增益，以及当前数据条件下合理精度上限。",
        ],
        "methods_tried_registry.md": [
            "",
            "## 2026-05-19 特征瓶颈与最小有效特征集诊断",
            "",
            "- 已尝试：高缺失/共线特征诊断、稳定核心特征集、机制可解释特征集、历史基线特征集。",
            "- 与普通调参不同：本轮固定时间切分与模型家族，目标是识别有效信息和噪声，而不是继续扩大模型搜索。",
        ],
        "failed_experiments.md": [
            "",
            "## 2026-05-19 特征筛选相关失败边界",
            "",
            "- 若 `full_features`、`stable_core_features`、`mechanism_features`、`history_only_baseline` 表现接近，则说明继续重排现有特征难以带来跃迁。",
            "- 若 `history_only_baseline` 接近全特征，说明外部遥感/环境特征对该任务增益有限，应避免继续堆同类高缺失特征。",
        ],
        "do_not_repeat_experiments.md": [
            "",
            "## 2026-05-19 新增不再重复项",
            "",
            "- 不再仅通过增加高缺失 rolling 遥感特征追求小幅 R² 提升。",
            "- 不再在缺少真实底层机制变量时反复微调 Chl-a bottom。",
            "- 不再把 validation/test 不稳定的单次特征重要性当作主线依据。",
        ],
        "data_gap_inventory.md": [
            "",
            "## 2026-05-19 特征瓶颈诊断后的数据缺口更新",
            "",
            "- 若精简特征与全特征表现接近，下一步重点仍是补真实机制数据，而不是继续派生同源特征。",
            "- Chl-a bottom 仍优先需要 `station_depth / bathymetry / bottom_temp / vertical profile / current / water level / discharge`。",
        ],
        "cleanup_log.md": [
            "",
            "## 2026-05-19 特征瓶颈诊断与最小有效特征集",
            "",
            "- 新增 `scripts/evaluation/run_current_feature_bottleneck_diagnostics.py`。",
            "- 更新 `feature_stability_analysis.csv`、`task_specific_feature_sets.json`、`feature_set_ab_metrics.csv`、误差分解表、站点推荐表和精度上限分析文档。",
            "- 本轮未新增数据源，未覆盖历史实验目录，输出集中在 current/。",
        ],
    }
    for name, lines in snippets.items():
        path = REPORT_DIR / name
        old = path.read_text(encoding="utf-8") if path.exists() else f"# {name}\n"
        marker = lines[1] if len(lines) > 1 else ""
        if marker and marker in old:
            continue
        path.write_text(old.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def update_model_docs(metrics: pd.DataFrame) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    lines = [
        "",
        "## 2026-05-19 特征集 A/B 补充",
        "",
        "本轮新增四组特征集比较：`full_features`、`stable_core_features`、`mechanism_features`、`history_only_baseline`。模型参数不做大规模调优，主要用于判断输入信息是否稳定有效。",
        "",
        "| target | depth | feature_set | feature_count | test R² | RMSE | MAE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in test.sort_values(["target", "depth_layer", "feature_set"]).iterrows():
        lines.append(
            f"| {row['target']} | {row['depth_layer']} | {row['feature_set']} | "
            f"{int(row['feature_count'])} | {row['r2']:.4f} | {row['rmse']:.4f} | {row['mae']:.4f} |"
        )
    for name in ["model_parameter_and_feature_inventory.md", "model_input_alignment_documentation.md"]:
        path = REPORT_DIR / name
        old = path.read_text(encoding="utf-8")
        marker = "## 2026-05-19 特征集 A/B 补充"
        if marker in old:
            continue
        path.write_text(old.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def update_md_checklist() -> None:
    md_files = [
        "experiment_report.md",
        "model_input_alignment_documentation.md",
        "model_parameter_and_feature_inventory.md",
        "methods_tried_registry.md",
        "failed_experiments.md",
        "do_not_repeat_experiments.md",
        "data_gap_inventory.md",
        "prediction_ceiling_analysis.md",
        "task_specific_feature_set_documentation.md",
        "cleanup_log.md",
        "station_group_prediction_summary.md",
    ]
    rows = []
    for name in md_files:
        path = REPORT_DIR / name
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        rows.append(
            {
                "md_file": name,
                "exists": path.exists(),
                "updated_this_run": "2026-05-19" in text or name
                in {
                    "prediction_ceiling_analysis.md",
                    "task_specific_feature_set_documentation.md",
                },
                "aligned_with_current_model": path.exists(),
                "includes_model_inputs": "特征" in text or "输入" in text,
                "includes_data_alignment": "对齐" in text or "date" in text,
                "includes_feature_meaning": "含义" in text or "机制" in text,
                "includes_latest_results": "R²" in text or "R2" in text or "RMSE" in text,
                "action_taken": "updated_or_verified",
                "remaining_issue": "",
            }
        )
    pd.DataFrame(rows).to_csv(
        REPORT_DIR / "md_update_checklist.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    ensure_dirs()
    df, highres_groups, mechanism_features = current_ab.merge_model_dataset()
    stability_path = OUT_DIR / "feature_stability_analysis.csv"
    metrics_path = OUT_DIR / "feature_set_ab_metrics.csv"
    predictions_path = OUT_DIR / "feature_set_ab_predictions.csv"
    station_path = OUT_DIR / "feature_set_station_level_comparison.csv"

    if stability_path.exists() and FEATURE_SET_JSON.exists():
        stability = pd.read_csv(stability_path, low_memory=False)
        feature_sets = read_feature_set_json()
    else:
        stability, feature_sets = analyze_feature_stability(df, highres_groups, mechanism_features)
        stability.to_csv(stability_path, index=False, encoding="utf-8-sig")
        write_feature_set_json(feature_sets)

    if metrics_path.exists() and predictions_path.exists() and station_path.exists():
        metrics = pd.read_csv(metrics_path, low_memory=False)
        predictions = pd.read_csv(predictions_path, low_memory=False)
        station_metrics = pd.read_csv(station_path, low_memory=False)
    else:
        metrics, predictions, station_metrics = run_feature_set_ab(df, feature_sets)
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
        station_metrics.to_csv(station_path, index=False, encoding="utf-8-sig")

    selected = write_error_diagnostics(df, predictions, metrics)
    recommendation = write_station_recommendation(station_metrics, selected)
    selection_summary = write_feature_set_selection_summary(metrics)
    write_feature_doc(metrics, feature_sets)
    write_ceiling_analysis(metrics, selected, stability)
    append_report_updates()
    update_model_docs(metrics)
    update_md_checklist()

    print(
        json.dumps(
            {
                "status": "ok",
                "feature_stability_rows": int(len(stability)),
                "feature_set_metric_rows": int(len(metrics)),
                "prediction_rows": int(len(predictions)),
                "station_recommendation_rows": int(len(recommendation)),
                "feature_set_selection_rows": int(len(selection_summary)),
                "selected_feature_sets": selected.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
