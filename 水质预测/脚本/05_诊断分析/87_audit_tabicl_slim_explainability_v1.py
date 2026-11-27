"""十项TabICLv2精简解释实验的只读资产审计与事前样本登记。

此脚本只读取冻结资产，不加载模型，不执行预测、SHAP或ALE。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v1"
MAIN = ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv"
FEATURES = ROOT / "runs/tabiclv2_pilot/01_strict_feature_manifest.csv"
OLD = ROOT / "runs/tabicl_v2_final_explanation_complete_v1"
V3 = ROOT / "runs/tabicl_v2_explainability_v3"
FOLD = "fold_1"
SEED = 42
QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
TASKS = (
    "chla_surface", "chla_middle", "chla_bottom", "secchi",
    "tn_surface", "tn_middle", "tn_bottom",
    "tp_surface", "tp_middle", "tp_bottom",
)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_ids(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def sha_array(values) -> str:
    data = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(str(data.dtype).encode() + str(data.shape).encode() + data.tobytes()).hexdigest()


def paths(task: str) -> tuple[Path, Path]:
    if task == "secchi":
        folder = ROOT / "runs/modern_baseline_gate_v1"
        return folder / "01_secchi_manifest.csv", folder / "predictions/secchi_tabiclv2_recovered.csv"
    family, depth = task.split("_")
    folder = ROOT / f"runs/{family}_family_v2"
    number = {"surface": "03", "middle": "04", "bottom": "05"}[depth]
    return folder / f"{number}_{task}_manifest.csv", folder / "predictions" / f"{task}_tabiclv2_fold_1.csv"


def select_quantile_samples(frozen: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    frame = frozen[["obs_id", "y_pred"]].merge(
        test[["obs_id", "station_id", "date"]], on="obs_id", validate="one_to_one"
    )
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    chosen: set[str] = set()
    stations: set[str] = set()
    years: set[int] = set()
    results = []
    for quantile in QUANTILES:
        value = float(frame["y_pred"].quantile(quantile))
        candidates = frame.loc[~frame.obs_id.isin(chosen)].copy()
        candidates["distance"] = (candidates.y_pred - value).abs()
        # 先确保预测分位区间，再在附近优先选新站点和新年份；不读取测试标签。
        window = candidates.nsmallest(min(50, len(candidates)), "distance").copy()
        window["new_station"] = (~window.station_id.astype(str).isin(stations)).astype(int)
        window["new_year"] = (~window.year.astype(int).isin(years)).astype(int)
        window = window.sort_values(
            ["new_station", "new_year", "distance", "obs_id"],
            ascending=[False, False, True, True], kind="stable",
        )
        row = window.iloc[0]
        obs = str(row.obs_id)
        chosen.add(obs)
        stations.add(str(row.station_id))
        years.add(int(row.year))
        results.append({
            "obs_id": obs, "station_id": str(row.station_id),
            "date": str(pd.to_datetime(row.date).date()),
            "year": int(row.year), "frozen_prediction": float(row.y_pred),
            "target_quantile": quantile, "quantile_target_value": value,
            "abs_prediction_distance": float(row.distance),
            "phase": "CORE_3" if quantile in (0.1, 0.5, 0.9) else "OPTIONAL_2",
            "selection_uses_test_label": False,
        })
    return results


def historical_shap(task: str, feature_order: list[str]) -> tuple[int, str]:
    directory = OLD / "shap_values" / task / FOLD
    if not directory.is_dir():
        return 0, ""
    valid = []
    for path in sorted(directory.glob("*_seed42.csv.gz")):
        frame = pd.read_csv(path)
        if (len(frame) == 77 and frame.feature.astype(str).tolist() == feature_order
                and float(frame.additivity_abs_error.iloc[0]) <= 0.02
                and frame.protocol.iloc[0] == "TABICLV2_POINTWISE_EXPLANATION_V1"):
            valid.append(str(frame.obs_id.iloc[0]))
    return len(valid), ";".join(valid)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feature_table = pd.read_csv(FEATURES)
    features = feature_table.loc[feature_table.included_in_pilot.astype(bool), "feature"].astype(str).tolist()
    if len(features) != 77 or len(set(features)) != 77:
        raise RuntimeError("冻结特征不是77个唯一变量")
    feature_hash = sha_ids(features)
    checkpoint = ROOT / "runs/tabicl_explanation_portable_bundle_v2/model/tabicl-regressor-v2-20260212.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"预训练checkpoint缺失: {checkpoint}")
    checkpoint_hash = sha_file(checkpoint)
    reproduction = pd.read_csv(V3 / "00_inference_reproduction.csv")
    old_env = pd.read_csv(OLD / "environment_manifest.csv")
    old_ale = pd.read_csv(OLD / "06_ale_1d_support_and_stability.csv")
    old_repeat = pd.read_csv(OLD / "ale_1d_repeatability.csv")
    runtimes = pd.read_csv(OLD / "shap_runtime.csv")
    per_sample_hours = float(runtimes.elapsed_seconds.mean() / 3600)
    data = pd.read_csv(
        MAIN, usecols=list(dict.fromkeys(["obs_id", "station", "date", "depth",
            "chla_ug_l", "secchi_depth_m", "tn_mg_l", "tp_mg_l", *features])),
        dtype={"obs_id": "string", "station": "string"}, low_memory=False,
    )
    data["obs_id"] = data.obs_id.astype("string").str.strip()
    data["station_id"] = data.station.astype("string").str.strip()
    audit, samples, budget = [], [], []
    for task in TASKS:
        split_path, prediction_path = paths(task)
        missing = [str(path) for path in (split_path, prediction_path) if not path.is_file()]
        depth = None if task == "secchi" else task.split("_")[1]
        target = {"chla": "chla_ug_l", "tn": "tn_mg_l", "tp": "tp_mg_l", "secchi": "secchi_depth_m"}[task.split("_")[0]]
        row = {"task": task, "fold": FOLD, "target": target,
            "depth": depth or "single_task", "p1_protocol": "TABICLV2_POINTWISE_EXPLANATION_V1",
            "seed": SEED, "feature_count": len(features), "feature_order_hash": feature_hash,
            "feature_manifest_sha256": sha_file(FEATURES),
            "checkpoint_sha256": checkpoint_hash, "model_checkpoint_path": str(checkpoint),
            "split_path": str(split_path), "prediction_path": str(prediction_path),
            "split_sha256": sha_file(split_path) if split_path.is_file() else "",
            "frozen_prediction_sha256": sha_file(prediction_path) if prediction_path.is_file() else "",
            "output_scale": "original_target", "background_n": 16,
            "background_source": "current_fold_training_only",
            "shap_masker": "shap.maskers.Independent",
            "shap_algorithm": "PermutationExplainer", "shap_max_evals": 155,
            "missing_assets": ";".join(missing),
        }
        if missing:
            row.update(status="BLOCKED", reason="冻结split或预测文件缺失")
            audit.append(row)
            continue
        man = pd.read_csv(split_path, dtype={"obs_id": "string"})
        man = man.loc[man.fold.astype(str).eq(FOLD)].copy()
        man["obs_id"] = man.obs_id.astype("string").str.strip()
        ids_train = man.loc[man.role.eq("train"), "obs_id"].astype(str).tolist()
        ids_test = man.loc[man.role.eq("test"), "obs_id"].astype(str).tolist()
        if not ids_train or not ids_test or len(set(ids_train) & set(ids_test)):
            row.update(status="BLOCKED", reason="fold 1 obs_id划分缺失或重叠")
            audit.append(row)
            continue
        frame = data if depth is None else data.loc[data.depth.astype(str).str.lower().eq(depth)]
        frame = frame.loc[pd.to_numeric(frame[target], errors="coerce").notna()].copy()
        if task == "secchi":
            index = frame.set_index(frame.obs_id.astype(str), drop=False)
            train = index.loc[ids_train].reset_index(drop=True)
            test = index.loc[ids_test].reset_index(drop=True)
        else:
            train = frame.loc[frame.obs_id.astype(str).isin(set(ids_train))].copy().reset_index(drop=True)
            test = frame.loc[frame.obs_id.astype(str).isin(set(ids_test))].copy().reset_index(drop=True)
        frozen = pd.read_csv(prediction_path, dtype={"obs_id": "string"})
        if task == "secchi":
            frozen = frozen.loc[frozen.fold.astype(str).eq(FOLD)].copy()
        frozen["obs_id"] = frozen.obs_id.astype("string").str.strip()
        valid_ids = (len(train) == len(ids_train) and len(test) == len(ids_test)
                     and len(frozen) == len(test) and set(frozen.obs_id) == set(test.obs_id))
        station_disjoint = not bool(set(train.station_id.astype(str)) & set(test.station_id.astype(str)))
        if not valid_ids or not station_disjoint:
            row.update(status="BLOCKED", reason="输入obs_id或训练/测试站点隔离检查失败")
            audit.append(row)
            continue
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        imp.fit(train[features].apply(pd.to_numeric, errors="coerce"))
        row.update(train_n=len(train), test_n=len(test),
            train_stations=train.station_id.nunique(), test_stations=test.station_id.nunique(),
            train_obs_order_hash=sha_ids(train.obs_id.astype(str)),
            test_obs_order_hash=sha_ids(test.obs_id.astype(str)),
            imputer_statistics_hash=sha_array(imp.statistics_),
            train_context_hash="PENDING_P1_CONTEXT_RECONSTRUCTION",
            station_disjoint=station_disjoint,
            frozen_prediction_min=float(frozen.y_pred.min()),
            frozen_prediction_max=float(frozen.y_pred.max()))
        old_result = reproduction.loc[(reproduction.task == task) & (reproduction.fold == FOLD)]
        historical_status = str(old_result.status.iloc[0]) if len(old_result) else "NOT_TESTED"
        row["historical_batch_reproduction"] = historical_status
        row["historical_max_abs_error"] = float(old_result.max_abs_error.iloc[0]) if len(old_result) else np.nan
        row["historical_mean_abs_error"] = float(old_result.mean_abs_error.iloc[0]) if len(old_result) else np.nan
        env = old_env.loc[(old_env.task == task) & (old_env.fold == FOLD)]
        if len(env):
            row["historical_p1_context_hash"] = str(env.context_hash.iloc[0])
            row["historical_p1_imputer_hash"] = str(env.imputer_stats_hash.iloc[0])
            row["imputer_matches_historical_p1"] = row["imputer_statistics_hash"] == row["historical_p1_imputer_hash"]
        shap_n, shap_ids = historical_shap(task, features)
        row["reusable_p1_shap_n"] = shap_n
        row["reusable_p1_shap_obs_ids"] = shap_ids
        ale = old_ale.loc[(old_ale.task == task) & (old_ale.fold == FOLD)]
        row["historical_p1_ale_features"] = ";".join(ale.feature.astype(str))
        row["historical_p1_ale_repeatability"] = ";".join(
            f"{r.feature}:{r.repeatability_status}" for r in old_repeat.loc[
                (old_repeat.task == task) & (old_repeat.fold == FOLD)].itertuples())
        if historical_status == "REPRODUCTION_FAILED":
            row.update(status="BLOCKED", reason="fold 1历史严格逐样本复现失败；禁止换折或覆盖旧结论")
        elif historical_status == "EXACT_REPRODUCTION":
            row.update(status="PENDING_P1_GATE", reason="旧批量接口通过；新P1接口仍需独立资格核验")
        else:
            row.update(status="PENDING_REPRODUCTION_GATE", reason="此任务无fold 1严格复现记录")
        audit.append(row)
        for item in select_quantile_samples(frozen, test):
            samples.append({"task": task, "fold": FOLD, **item,
                "pre_registered_before_explanation": True,
                "historical_p1_shap_exists": item["obs_id"] in shap_ids.split(";")})
        budget.append({"task": task, "fold": FOLD, "gate_status": row["status"],
            "core_shap_samples": 3, "optional_shap_samples": 2,
            "historical_reusable_shap_n": shap_n,
            "rtx2070_observed_hours_per_sample": per_sample_hours,
            "core_shap_gpu_hours_chla_speed_proxy": 3 * per_sample_hours,
            "five_shap_gpu_hours_chla_speed_proxy": 5 * per_sample_hours,
            "ale_features_max": 2, "ale_query_rows_planned": "300-600 if support permits",
            "ale_model_calls_per_feature_rough": "600-1200",
            "caveat": "SHAP速度仅来自Chl-a表层两样本；其他任务和GPU不得当作实测"})
    pd.DataFrame(audit).to_csv(OUT / "00_frozen_protocol_task_registry.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(samples).to_csv(OUT / "01_preregistered_shap_samples.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(budget).to_csv(OUT / "02_compute_budget.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"task": r["task"], "fold": FOLD, "status": r["status"],
        "reuse_shap_n": r.get("reusable_p1_shap_n", 0),
        "reuse_ale_features": r.get("historical_p1_ale_features", ""),
        "reason": r["reason"]} for r in audit]).to_csv(
        OUT / "03_reuse_pending_blocked.csv", index=False, encoding="utf-8-sig")
    counts = pd.Series([r["status"] for r in audit]).value_counts().to_dict()
    report = ["# TabICLv2预测可解释性精简正式实验V1：只读资产审计", "",
        "## 已冻结范围", "", "十项正式任务均预登记fold 1；77特征、seed 42、训练折16行背景、",
        "PermutationExplainer max_evals=155、P1逐行预测、原始目标输出尺度。",
        "本审计没有执行模型推理、SHAP或ALE，也没有修改冻结结果。", "",
        "## 资格状态", "", f"状态计数：`{json.dumps(counts, ensure_ascii=False)}`。", "",
        "`chla_surface/fold_1`既有严格复现失败，且ALE重复性超过1e-6；按本轮硬门槛标记BLOCKED。",
        "已有两条77维P1 SHAP及两条ALE保留为历史计算资产，不能据此把该折升级为正式通过。", "",
        "Secchi、TN surface、TP middle的旧批量预测曾严格复现；P1逐查询函数与批量接口不是同一个函数，",
        "仍需独立P1资格核验。另六项没有相应严格复现记录，须先核验才可计算解释。", "",
        "## 预算依据", "",
        f"两条已完成P1 SHAP的平均实测耗时为{per_sample_hours:.3f}小时/样本。",
        "预算表仅以RTX2070的Chl-a样本作为速度代理，不是新任务或RTX4060实测，也不是上限。",
        "ALE先做CPU支持审查，仅对合格变量发起GPU调用。", "",
        "## 科学边界", "",
        "单折、每任务3至5个样本只能形成代表性局部解释，不能称全测试集全局重要性或五折稳定性。",
        "历史V1/V2数值不稳定结论和失败折次原样保留。", ""]
    (OUT / "04_asset_audit_decision.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"task_count": len(audit), "statuses": counts,
        "sample_rows": len(samples), "observed_shap_hours_per_sample": per_sample_hours,
        "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
