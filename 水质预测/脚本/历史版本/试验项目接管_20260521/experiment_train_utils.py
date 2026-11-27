from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_MAP = {
    "chla": "chla_ug_l",
    "secchi": "secchi_depth_m",
    "tn": "tn_mg_l",
    "tp": "tp_mg_l",
}

FEATURE_SET_CHOICES = ["baseline", "f1_history", "f2_remote", "f3_history_remote"]

DEFAULT_MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": 4,
}


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.replace("\ufeff", "") for c in df.columns})


def normalize_bool_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        lowered = out[col].dropna().astype(str).str.strip().str.lower()
        if lowered.empty:
            continue
        if set(lowered.unique()).issubset({"true", "false", "0", "1"}):
            out[col] = (
                out[col].astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
            )
    return out


def prepare_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    data = normalize_columns(data)
    data = normalize_bool_like_columns(data)
    data["water_zone_en"] = data["water_zone_en"].fillna("unknown")
    if "water_zone_code" not in data.columns:
        data["water_zone_code"] = pd.Categorical(data["water_zone_en"]).codes.astype("int16")
    return data


def parse_feature_list(raw: str) -> list[str]:
    if pd.isna(raw) or not str(raw).strip():
        return []
    value = json.loads(raw)
    return [str(x) for x in value]


def load_manifest(path: Path, targets: list[str] | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(path, low_memory=False)
    manifest = normalize_columns(manifest)
    if targets is not None:
        manifest = manifest[manifest["target"].isin(targets)].copy()
    manifest["base_features"] = manifest["features_json"].map(parse_feature_list)
    return manifest


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = num / den
    invalid = den.isna() | (den == 0)
    out = out.mask(invalid)
    return out.replace([np.inf, -np.inf], np.nan)


def make_model(model_params: dict[str, object]):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("Missing xgboost. Install it with `pip install xgboost`.") from exc

    params = dict(DEFAULT_MODEL_PARAMS)
    params.update(model_params)
    return XGBRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        objective="reg:squarederror",
        random_state=int(params["random_state"]),
        n_jobs=int(params["n_jobs"]),
    )


def make_classifier(model_params: dict[str, object]):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit("Missing xgboost. Install it with `pip install xgboost`.") from exc

    params = dict(DEFAULT_MODEL_PARAMS)
    params.update(model_params)
    return XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(params["random_state"]),
        n_jobs=int(params["n_jobs"]),
    )


def fit_combo(
    data: pd.DataFrame,
    combo: dict[str, object],
    features: list[str],
    model_params: dict[str, object] | None = None,
    sample_weight_train: np.ndarray | None = None,
    force_target_transform: str | None = None,
    min_train_rows: int = 200,
    min_test_rows: int = 50,
    train_end_year: int = 2022,
    test_start_year: int = 2023,
    test_end_year: int = 2024,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    target_key = str(combo["target"])
    target_col = str(combo["target_column"])
    depth = str(combo["depth"])
    feature_set_name = str(combo["feature_set"])
    features = [f for f in features if f in data.columns]

    subset = data[(data["depth"] == depth) & data[target_col].notna()].copy()
    subset = subset.sort_values(["date", "station"]).reset_index(drop=True)
    train = subset[subset["year"] <= train_end_year].copy()
    test = subset[subset["year"].between(test_start_year, test_end_year)].copy()

    result: dict[str, object] = {
        "target": target_key,
        "target_column": target_col,
        "depth": depth,
        "feature_set": feature_set_name,
        "feature_count": len(features),
        "features_json": json.dumps(features, ensure_ascii=False),
        "train_rows": len(train),
        "test_rows": len(test),
    }

    if not features:
        result["status"] = "skipped_no_features"
        return result, pd.DataFrame(), pd.DataFrame()
    if len(train) < min_train_rows or len(test) < min_test_rows:
        result["status"] = "skipped_insufficient_rows"
        return result, pd.DataFrame(), pd.DataFrame()

    model = make_model(model_params or {})
    positive_target = train[target_col].dropna().ge(0).all() and test[target_col].dropna().ge(0).all()
    use_log1p = force_target_transform == "log1p" or (
        force_target_transform is None and positive_target
    )
    if use_log1p:
        train_y = np.log1p(train[target_col].to_numpy())
        model.fit(train[features], train_y, sample_weight=sample_weight_train)
        pred = np.expm1(model.predict(test[features]))
        pred = np.clip(pred, a_min=0, a_max=None)
        y = test[target_col].to_numpy()
        result["target_transform"] = "log1p"
        result["r2_log_space"] = float(r2_score(np.log1p(y), np.log1p(pred)))
    else:
        model.fit(train[features], train[target_col], sample_weight=sample_weight_train)
        pred = model.predict(test[features])
        y = test[target_col].to_numpy()
        result["target_transform"] = "none"

    result.update(
        {
            "status": "ok",
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
            "nse": float(nse(y, pred)),
        }
    )

    pred_df = test[
        ["obs_id", "sample_id", "station", "station_name_en", "water_zone_en", "date", "year", "month", "depth", target_col]
    ].copy()
    pred_df["target"] = target_key
    pred_df["feature_set"] = feature_set_name
    pred_df["observed"] = pred_df[target_col]
    pred_df["predicted"] = pred
    pred_df = pred_df.drop(columns=[target_col])

    importance_df = pd.DataFrame(
        {
            "target": target_key,
            "target_column": target_col,
            "depth": depth,
            "feature_set": feature_set_name,
            "feature": features,
            "importance": model.feature_importances_.tolist(),
        }
    )
    return result, pred_df, importance_df


def subset_regression_metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    obs = pd.to_numeric(observed, errors="coerce")
    pred = pd.to_numeric(predicted, errors="coerce")
    mask = obs.notna() & pred.notna()
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0:
        return {
            "rows": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "mean_error": np.nan,
            "underestimation_rate": np.nan,
        }
    return {
        "rows": int(len(obs)),
        "rmse": float(np.sqrt(mean_squared_error(obs, pred))),
        "mae": float(mean_absolute_error(obs, pred)),
        "r2": float(r2_score(obs, pred)) if len(obs) >= 2 else np.nan,
        "mean_error": float((pred - obs).mean()),
        "underestimation_rate": float((pred < obs).mean()),
    }


def compare_metrics(
    baseline_metrics: pd.DataFrame,
    experiment_metrics: pd.DataFrame,
) -> pd.DataFrame:
    compare = baseline_metrics.merge(
        experiment_metrics,
        on=["target", "target_column", "depth", "feature_set", "status"],
        suffixes=("_baseline", "_experiment"),
        how="outer",
    )
    for metric in ["feature_count", "train_rows", "test_rows", "rmse", "mae", "r2", "nse"]:
        compare[f"{metric}_delta"] = compare[f"{metric}_experiment"] - compare[f"{metric}_baseline"]
    return compare


def summarize_added_feature_usage(
    importance_df: pd.DataFrame,
    added_feature_map: dict[tuple[str, str, str], list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, item in importance_df.iterrows():
        key = (str(item["target"]), str(item["depth"]), str(item["feature_set"]))
        added = set(added_feature_map.get(key, []))
        feature = str(item["feature"])
        if feature not in added:
            continue
        rows.append(
            {
                "feature": feature,
                "used_in_combo": int(float(item["importance"]) > 0),
                "importance": float(item["importance"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "used_combo_count", "mean_importance", "max_importance"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("feature", as_index=False)
        .agg(
            used_combo_count=("used_in_combo", "sum"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
        )
        .sort_values(["used_combo_count", "mean_importance", "feature"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
