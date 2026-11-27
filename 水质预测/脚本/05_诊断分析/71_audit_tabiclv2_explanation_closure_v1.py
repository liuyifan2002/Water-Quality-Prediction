"""审计 V3 解释缺口、样本代表性与补全成本，不执行模型训练。"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "runs/tabicl_v2_explainability_v3"
OUT = ROOT / "runs/tabicl_v2_explainability_closure_v1"
BASE_PATH = ROOT / "脚本/05_诊断分析/66_run_tabiclv2_explainability_v2.py"


def load_base():
    spec = importlib.util.spec_from_file_location("closure_base", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def read_shap_rows():
    frames = [pd.read_csv(p) for p in V3.glob("shap_values/*/*/*.csv.gz")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def split_half_stability(frame):
    rows=[]
    for (task,fold),g in frame.groupby(["task","fold"]):
        ids=list(g.obs_id.drop_duplicates()); pairs=[]
        # Exhaustive 2-vs-remaining splits for n=5; deterministic cap for larger n.
        comb=list(itertools.combinations(ids,max(2,len(ids)//2)))[:100]
        for left in comb:
            a=g[g.obs_id.isin(left)].groupby("feature").shap_value.apply(lambda x:np.abs(x).mean())
            b=g[~g.obs_id.isin(left)].groupby("feature").shap_value.apply(lambda x:np.abs(x).mean())
            common=a.index.intersection(b.index); rho=spearmanr(a.loc[common],b.loc[common]).statistic
            ta=set(a.nlargest(10).index); tb=set(b.nlargest(10).index); jac=len(ta&tb)/len(ta|tb)
            pairs.append((rho,jac))
        rows.append({"task":task,"fold":fold,"n_explanation_samples":len(ids),
            "median_split_half_spearman":np.nanmedian([x[0] for x in pairs]),
            "minimum_split_half_spearman":np.nanmin([x[0] for x in pairs]),
            "median_top10_jaccard":np.nanmedian([x[1] for x in pairs]),
            "minimum_top10_jaccard":np.nanmin([x[1] for x in pairs]),
            "ranking_support":"SUPPORTED" if len(ids)>=10 and np.nanmedian([x[0] for x in pairs])>=.70 and np.nanmedian([x[1] for x in pairs])>=.50 else "EXPLORATORY"})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    features=BASE.load_features(); rep=pd.read_csv(V3/"00_inference_reproduction.csv"); ale_rep=pd.read_csv(V3/"00_ale_inference_reproduction.csv")
    runtime=pd.read_csv(V3/"01_shap_runtime_benchmark.csv"); manifest=pd.read_csv(V3/"02_tabicl_shap_values_manifest.csv")
    bg=pd.read_csv(V3/"background/background_manifest.csv"); ex=pd.read_csv(V3/"explanation/explanation_sample_manifest.csv")
    a1=pd.read_csv(V3/"ale_1d/ale_1d_by_fold.csv"); a2=pd.read_csv(V3/"ale_2d/ale_2d_by_fold.csv")
    shap=read_shap_rows(); median_seconds=float(runtime.elapsed_seconds.median())
    gaps=[]; represent=[]
    for task in BASE.TASKS:
        for i in range(1,6):
            fold=f"fold_{i}"; r=rep[(rep.task==task)&(rep.fold==fold)]
            shap_n=manifest[(manifest.task==task)&(manifest.fold==fold)].obs_id.nunique()
            bg_n=bg[(bg.task==task)&(bg.fold==fold)].obs_id.nunique(); ale_features=a1[(a1.task==task)&(a1.fold==fold)].feature.nunique()
            ale2_pairs=a2[(a2.task==task)&(a2.fold==fold)][["feature_1","feature_2"]].drop_duplicates().shape[0]
            status=r.status.iloc[0] if len(r) else "NOT_RUN"; reason="" if status=="EXACT_REPRODUCTION" else "未通过事前固定逐样本预测复现阈值"
            gaps.append({"task":task,"fold":fold,"inference_status":status,
                "max_abs_reproduction_error":r.max_abs_error.iloc[0] if len(r) else np.nan,
                "shap_samples_completed":shap_n,"shap_background_n":bg_n,"shap_status":"COMPLETE" if shap_n>=5 else "MISSING",
                "shap_additivity_max_error":runtime[(runtime.task==task)&(runtime.fold==fold)].additivity_abs_error.max(),
                "ale_1d_features_completed":ale_features,"ale_1d_status":"COMPLETE" if ale_features>=4 else "MISSING",
                "ale_2d_pairs_completed":ale2_pairs,"uncompleted_reason":reason,
                "estimated_shap_minutes_for_5_samples":0 if shap_n>=5 else median_seconds*5/60,
                "estimated_ale_minutes":0 if ale_features>=4 else 4*6/60})
            if shap_n:
                train,test=BASE.load_fold(task,fold,features); q=ex[(ex.task==task)&(ex.fold==fold)].drop_duplicates("obs_id")
                picked=test[test.obs_id.astype(str).isin(q.obs_id.astype(str))].copy(); years=pd.to_datetime(picked.date).dt.year
                # Fraction of features for which selected samples span at least the test 10th-90th percentile interval endpoints.
                cover=[]
                for f in features:
                    lo,hi=np.nanquantile(test[f],[.1,.9]); vals=picked[f].to_numpy(float)
                    cover.append(bool(np.nanmin(vals)<=lo and np.nanmax(vals)>=hi))
                pred=shap[(shap.task==task)&(shap.fold==fold)].drop_duplicates("obs_id").formal_full_batch_prediction
                represent.append({"task":task,"fold":fold,"n_samples":len(picked),"n_stations":picked.station_id.nunique(),
                    "n_years":years.nunique(),"year_min":years.min(),"year_max":years.max(),
                    "feature_p10_p90_span_fraction":np.mean(cover),"prediction_min":pred.min(),"prediction_max":pred.max(),
                    "selection_used_test_label":False})
    gap=pd.DataFrame(gaps); gap.to_csv(OUT/"00_explanation_gap_audit.csv",index=False,encoding="utf-8-sig")
    repdf=pd.DataFrame(represent); stab=split_half_stability(shap)
    repdf.merge(stab,on=["task","fold"],how="left").to_csv(OUT/"01_shap_sample_representativeness.csv",index=False,encoding="utf-8-sig")
    stab.to_csv(OUT/"02_shap_sampling_sensitivity.csv",index=False,encoding="utf-8-sig")
    missing=gap[gap.shap_status=="MISSING"]
    text=f"""# TabICLv2解释闭环计划\n\n## 实际缺口\n\n- 当前SHAP完成：{int((gap.shap_status=='COMPLETE').sum())}/20折。\n- 当前一维ALE完成：{int((gap.ale_1d_status=='COMPLETE').sum())}/20折。\n- 尚缺SHAP：{', '.join((missing.task+' '+missing.fold).tolist()) or '无'}。\n- 补齐5个SHAP样本的折中实测预计约{median_seconds*5/60:.1f}分钟/折；仅在原严格预测复现门槛通过后运行。\n\n## 代表性判据\n\n每折原有5个样本同时要求站点、年份和环境范围覆盖；抽样稳定性采用确定性的分半Spearman与前10 Jaccard。只有样本数不少于10、分半Spearman中位数不低于0.70且前10 Jaccard中位数不低于0.50时，才标记为SUPPORTED；否则保留EXPLORATORY。\n\n## 执行边界\n\n不重新训练、不更换77特征、不改变背景16、PermutationExplainer或max_evals=155；失败折不放宽复现阈值。新增样本按原不使用测试标签的站点-年份覆盖加77维最大最小距离规则确定，已完成obs_id不重复计算。\n"""
    (OUT/"00_explanation_closure_plan.md").write_text(text,encoding="utf-8")
    print(gap.groupby(["shap_status","ale_1d_status"]).size())


if __name__=="__main__": main()
