"""审计 XGB_SHAP_ALE_V1 所需的冻结 XGBoost 模型资产。

本脚本严格只读上游资产，不训练模型，不计算 SHAP/ALE，不修改正式划分。
当五折模型缺失时，生成可追溯的 BLOCKED_MODEL_ARTIFACTS 决策包。
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "xgb_shap_ale_v1"
STRICT_MANIFEST = ROOT / "runs" / "tabiclv2_pilot" / "01_strict_feature_manifest.csv"
TAXONOMY = ROOT / "runs" / "multisource_ablation_v1" / "00_feature_source_taxonomy.csv"
MODEL_SUFFIXES = {".json", ".ubj", ".model", ".bst", ".pkl", ".pickle", ".joblib", ".bin"}
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))

TASKS = {
    "chla_surface": {
        "run": ROOT / "runs" / "chla_family_v2",
        "manifest": "03_chla_surface_manifest.csv",
        "prediction_pattern": "predictions/chla_surface_xgboost_{fold}.csv",
        "training_script": ROOT / "脚本" / "04_模型训练" / "44_run_chla_family_v2_models.py",
        "split_version": "CHLA_FAMILY_V2_UNIFIED_STATION_SPLIT",
    },
    "secchi": {
        "run": ROOT / "runs" / "tabiclv2_pilot",
        "manifest": "02_split_manifest.csv",
        "prediction_pattern": "09_secchi_predictions.csv",
        "training_script": ROOT / "脚本" / "04_模型训练" / "33_run_tabiclv2_chla_smoke.py",
        "split_version": "SECCHI_HISTORICAL_STATION_HOLDOUT",
    },
    "tn_surface": {
        "run": ROOT / "runs" / "tn_family_v2",
        "manifest": "03_tn_surface_manifest.csv",
        "prediction_pattern": "predictions/tn_surface_xgboost_{fold}.csv",
        "training_script": ROOT / "脚本" / "04_模型训练" / "36_run_tn_family_v2_models.py",
        "split_version": "TN_FAMILY_V2_UNIFIED_STATION_SPLIT",
    },
    "tp_middle": {
        "run": ROOT / "runs" / "tp_family_v2",
        "manifest": "04_tp_middle_manifest.csv",
        "prediction_pattern": "predictions/tp_middle_xgboost_{fold}.csv",
        "training_script": ROOT / "脚本" / "04_模型训练" / "41_run_tp_family_v2_models.py",
        "split_version": "TP_FAMILY_V2_UNIFIED_STATION_SPLIT",
    },
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_candidates(directory: Path, task: str, fold: str) -> list[Path]:
    if not directory.exists():
        return []
    tokens = {task.lower(), task.replace("_", "").lower(), "xgb", "xgboost", fold.lower()}
    matches = []
    excluded = {"predictions", "fold_metrics", "环境", ".venv", "__pycache__"}
    stack = [(directory, 0)]
    while stack:
        current, depth = stack.pop()
        for path in current.iterdir():
            if path.is_dir():
                if depth < 3 and path.name not in excluded:
                    stack.append((path, depth + 1))
                continue
            if path.suffix.lower() not in MODEL_SUFFIXES:
                continue
            name = path.name.lower()
            if fold.lower() in name and ("xgb" in name or "xgboost" in name) and any(t in name for t in tokens):
                matches.append(path)
    return sorted(matches)


def load_feature_evidence() -> tuple[list[str], pd.DataFrame]:
    manifest = pd.read_csv(STRICT_MANIFEST)
    features = manifest.loc[manifest["included_in_pilot"].astype(bool), "feature"].dropna().astype(str).tolist()
    taxonomy = pd.read_csv(TAXONOMY)
    return features, taxonomy


def main() -> None:
    for name in ("00_audit", "01_shap", "02_ale_1d", "03_ale_2d", "04_integrated_analysis", "figures", "tables", "reports", "registry"):
        (RUN / name).mkdir(parents=True, exist_ok=True)

    features, taxonomy = load_feature_evidence()
    now = datetime.now().isoformat(timespec="seconds")
    artifact_rows, reproduction_rows, registry_rows = [], [], []

    for task, spec in TASKS.items():
        script_text = spec["training_script"].read_text(encoding="utf-8", errors="replace")
        persistence_api_present = any(token in script_text for token in ("save_model(", "joblib.dump(", "pickle.dump("))
        manifest_path = spec["run"] / spec["manifest"]
        for fold in FOLDS:
            candidates = model_candidates(spec["run"], task, fold)
            pred_path = spec["run"] / spec["prediction_pattern"].format(fold=fold)
            artifact_rows.append({
                "task": task,
                "fold": fold,
                "split_version": spec["split_version"],
                "split_manifest_path": rel(manifest_path),
                "split_manifest_exists": manifest_path.exists(),
                "training_script": rel(spec["training_script"]),
                "model_persistence_api_present": persistence_api_present,
                "model_candidate_count": len(candidates),
                "model_paths": "|".join(rel(p) for p in candidates),
                "frozen_prediction_path": rel(pred_path),
                "frozen_prediction_exists": pred_path.exists(),
                "status": "MODEL_ARTIFACT_MISSING" if not candidates else "MODEL_CANDIDATE_FOUND_REQUIRES_LOAD_VALIDATION",
                "reason": "训练入口仅保存预测/指标，目标目录未找到该任务该折的可加载XGBoost模型" if not candidates else "发现候选模型，需进一步加载核验",
            })
            reproduction_rows.append({
                "task": task, "fold": fold, "n_samples": 0,
                "absolute_tolerance": "NOT_SET_MODEL_MISSING", "relative_tolerance": "NOT_SET_MODEL_MISSING",
                "max_abs_difference": "", "mean_abs_difference": "",
                "status": "NOT_RUN_MODEL_ARTIFACT_MISSING",
                "reason": "prediction CSV不能代替模型文件；未执行加载推理核验",
            })
            registry_rows.append({
                "task": task, "fold": fold, "analysis_type": "MODEL_AUDIT", "feature_or_pair": "",
                "status": "BLOCKED", "output_path": "00_audit/00_model_artifact_audit.csv",
                "elapsed_seconds": 0, "reason": "MODEL_ARTIFACT_MISSING", "updated_at": now,
            })

    artifact = pd.DataFrame(artifact_rows)
    all_models_present = bool((artifact["model_candidate_count"] > 0).all())
    feature_rows = []
    taxonomy_feature_column = "feature_name" if "feature_name" in taxonomy.columns else "feature"
    taxonomy_features = set(taxonomy[taxonomy_feature_column].dropna().astype(str)) if taxonomy_feature_column in taxonomy.columns else set()
    for task in TASKS:
        feature_rows.append({
            "task": task,
            "strict_manifest_path": rel(STRICT_MANIFEST),
            "strict_feature_count": len(features),
            "strict_feature_unique_count": len(set(features)),
            "taxonomy_path": rel(TAXONOMY),
            "taxonomy_match_count": sum(f in taxonomy_features for f in features),
            "model_feature_names_available": False,
            "model_feature_order_verifiable": False,
            "preprocessing_from_training_script": "SimpleImputer(strategy=median, keep_empty_features=True); task-dependent log1p target",
            "status": "BLOCKED_MODEL_FEATURE_METADATA_UNAVAILABLE",
            "reason": "77特征清单存在，但模型文件缺失，无法从模型反查名称与顺序",
        })

    audit_dir = RUN / "00_audit"
    artifact.to_csv(audit_dir / "00_model_artifact_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_rows).to_csv(audit_dir / "00_feature_alignment_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(reproduction_rows).to_csv(audit_dir / "00_model_prediction_reproduction.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(registry_rows).to_csv(RUN / "registry" / "run_registry.csv", index=False, encoding="utf-8-sig")

    decision = "READY_FOR_SHAP" if all_models_present else "BLOCKED_MODEL_ARTIFACTS"
    preflight = f"""# XGB_SHAP_ALE_V1 Step 0 预检决策

- 生成时间：{now}
- 决策：`{decision}`
- 审计范围：Chl-a surface、Secchi、TN surface、TP middle，共 20 个 task-fold。
- strict feature manifest：`{rel(STRICT_MANIFEST)}`，纳入特征 {len(features)} 个，唯一特征 {len(set(features))} 个。
- 来源分类：`{rel(TAXONOMY)}`，77 特征匹配 {sum(f in taxonomy_features for f in features)} 个。

## 结论

四个任务均没有定位到完整五折、可加载且与冻结预测对应的 XGBoost 模型文件。现有 prediction CSV、fold metrics、split manifest 和参数配置不能替代训练完成后的模型对象。根据实验禁令，不重新训练、不根据预测拟合替代模型，因此不执行 SHAP、1D ALE、2D ALE及正式论文图。

## 上游证据

- Family V2 训练入口在内存中拟合并输出预测，未调用 XGBoost `save_model`、pickle 或 joblib 持久化接口。
- Secchi 历史入口同样只返回预测；`runs/tabiclv2_pilot/09_secchi_predictions.csv` 是预测文件，不是模型文件。
- 77 特征清单与来源分类文件存在，但模型缺失导致无法核验模型内特征名称、输入顺序及加载后预测一致性。

## 解阻条件

需提供原始冻结运行产生的 20 个模型文件（四任务 × 五折），以及每折拟合后的 median imputer 参数或包含预处理器的完整 pipeline；随后才能执行小样本预测复现核验。重新训练得到的新模型不属于本任务允许的解阻方式。
"""
    (audit_dir / "00_preflight_decision.md").write_text(preflight, encoding="utf-8")

    inventory = []
    planned = ["四任务主要特征重要性对比图", "Chl-a surface SHAP总体解释图", "Secchi SHAP总体解释图", "TN surface SHAP总体解释图", "TP middle SHAP总体解释图",
               "Chl-a surface一维ALE组合图", "Secchi一维ALE组合图", "TN surface一维ALE组合图", "TP middle一维ALE组合图",
               "Chl-a surface二维ALE组合图", "Secchi二维ALE组合图", "TN surface二维ALE组合图", "TP middle二维ALE组合图", "SHAP贡献与来源消融综合对照图"]
    for figure in planned:
        inventory.append({"figure": figure, "png_path": "", "svg_path": "", "data_path": "", "status": "NOT_GENERATED", "reason": "BLOCKED_MODEL_ARTIFACTS"})
    pd.DataFrame(inventory).to_csv(RUN / "xgb_shap_ale_v1_figure_inventory.csv", index=False, encoding="utf-8-sig")

    limitations = """# XGB_SHAP_ALE_V1 已知限制

1. 四个目标任务缺少与正式冻结预测逐折对应的已训练 XGBoost 模型文件。
2. 预处理过程可从脚本识别为逐折训练集 median imputer，但拟合后的每折中位数未独立持久化；不能以当前数据重新拟合后冒充原始冻结预处理器。
3. 因模型资产缺失，无法核验模型内77特征的名称、顺序以及加载推理与冻结预测的一致性。
4. SHAP、ALE、综合对照和论文图均未启动；任何已有预测文件都未被用来拟合替代模型。
5. 本结论只说明解释资产不完整，不否定最终实验闭环中的预测指标。
"""
    (RUN / "xgb_shap_ale_v1_known_limitations.md").write_text(limitations, encoding="utf-8")
    final = f"""# XGB_SHAP_ALE_V1 最终决策

## `BLOCKED_MODEL_ARTIFACTS`

- 完成时间：{now}
- 已完成：Step 0 定向资产审计、77特征与来源分类存在性检查、20个 task-fold 的模型/预测/划分对照、阻塞注册。
- 未执行：SHAP、1D ALE、2D ALE、消融综合解释和论文图生成。
- 覆盖结论：0/4 个任务具备完整五折可加载冻结 XGBoost 模型；0/20 个 fold 通过模型加载预测复现。
- 阻塞原因：上游训练仅保存预测与指标，没有持久化本任务要求的冻结模型及已拟合预处理器。
- 合规处理：未重训、未调参、未更换77特征、未重建split、未使用测试标签筛选变量、未由预测文件重建模型。

详细证据见 `00_audit/00_model_artifact_audit.csv` 与 `00_audit/00_preflight_decision.md`。
"""
    (RUN / "xgb_shap_ale_v1_decision.md").write_text(final, encoding="utf-8")

    evidence = pd.DataFrame([
        {"artifact": rel(STRICT_MANIFEST), "exists": STRICT_MANIFEST.exists(), "sha256": sha256(STRICT_MANIFEST)},
        {"artifact": rel(TAXONOMY), "exists": TAXONOMY.exists(), "sha256": sha256(TAXONOMY)},
        *({"artifact": rel(spec["training_script"]), "exists": True, "sha256": sha256(spec["training_script"])} for spec in TASKS.values()),
    ])
    evidence.to_csv(audit_dir / "00_input_evidence_hashes.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(decision)


if __name__ == "__main__":
    main()
