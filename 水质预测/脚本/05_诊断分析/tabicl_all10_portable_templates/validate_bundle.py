"""只读核验便携运行包中的哈希、十任务输入和77特征。"""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"chla_surface","chla_middle","chla_bottom","secchi","tn_surface","tn_middle","tn_bottom","tp_surface","tp_middle","tp_bottom"}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def main() -> int:
    issues=[]
    manifest=ROOT/"file_manifest_sha256.csv"
    if not manifest.is_file(): issues.append("file_manifest_sha256.csv缺失")
    else:
        for row in pd.read_csv(manifest).itertuples(index=False):
            path=(ROOT/row.relative_path).resolve()
            if ROOT.resolve() not in path.parents: issues.append(f"路径越界：{row.relative_path}")
            elif not path.is_file() or sha(path)!=row.sha256: issues.append(f"缺失或哈希错误：{row.relative_path}")
    registry=ROOT/"runs/tabicl_v2_slim_explainability_v1/00_frozen_protocol_task_registry.csv"
    samples=ROOT/"runs/tabicl_v2_slim_explainability_v1/01_preregistered_shap_samples.csv"
    features=ROOT/"runs/tabiclv2_pilot/01_strict_feature_manifest.csv"
    if all(p.is_file() for p in (registry,samples,features)):
        r=pd.read_csv(registry); s=pd.read_csv(samples); f=pd.read_csv(features)
        if set(r.task.astype(str))!=TASKS: issues.append("任务注册表不是预期十任务")
        if set(s.task.astype(str))!=TASKS: issues.append("解释样本manifest未覆盖十任务")
        names=f.loc[f.included_in_pilot.astype(bool),"feature"].astype(str).tolist()
        if len(names)!=77 or len(set(names))!=77: issues.append("strict feature manifest并非77个唯一变量")
        model=ROOT/"model/tabicl-regressor-v2-20260212.ckpt"
        if not model.is_file() or sha(model) not in set(r.checkpoint_sha256.astype(str)): issues.append("checkpoint与冻结注册表hash不一致")
    else: issues.append("必要输入manifest缺失")
    for path in ("数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv","models/tabiclv2_adapter.py",
                 "脚本/05_诊断分析/101_resume_tabicl_all10_explanation_v2.py",
                 "脚本/05_诊断分析/实时进度看板/index.html"):
        if not (ROOT/path).is_file(): issues.append(f"运行输入缺失：{path}")
    report={"status":"PASS" if not issues else "FAIL","issues":issues,
            "task_count":10,"strict_feature_count":77,"checked_at":pd.Timestamp.now().isoformat(timespec="seconds")}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    (ROOT/"portable_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return 0 if not issues else 1

if __name__=="__main__": raise SystemExit(main())
