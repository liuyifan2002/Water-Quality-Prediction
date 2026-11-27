"""Summarize frozen Random Strict test predictions and compare Station V2 separately."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"runs/random_10task_v1"
STATION=ROOT/"runs/final_experiment_closure_audit_v1/final_core_results_table_v2.csv"
ORDER=["chla_surface","chla_middle","chla_bottom","secchi","tn_surface","tn_middle","tn_bottom","tp_surface","tp_middle","tp_bottom"]

def main():
    paths=[OUT/"04_metrics"/f"{task}.csv" for task in ORDER]
    missing=[p.name for p in paths if not p.exists()]
    if missing:raise FileNotFoundError(f"未完成任务: {missing}")
    m=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    if len(m)!=20 or m.groupby("task").model.nunique().ne(2).any():raise ValueError("十任务两模型结果不完整")
    m["task"]=pd.Categorical(m.task,ORDER,ordered=True)
    def inner_before_after(row):
        try:
            vals=json.loads(row.inner_val_r2)
            if not vals:return pd.Series([float("nan")]*3)
            return pd.Series([vals[0],max(vals),max(vals)-vals[0]])
        except (TypeError, ValueError):return pd.Series([float("nan")]*3)
    m[["xgb_inner_val_baseline_R2","xgb_inner_val_selected_R2","xgb_inner_val_delta_R2"]]=m.apply(inner_before_after,axis=1)
    m=m.sort_values(["task","model"])
    m.to_csv(OUT/"02_run_registry.csv",index=False,encoding="utf-8-sig")
    m.to_csv(OUT/"06_model_comparison.csv",index=False,encoding="utf-8-sig")
    w=m.pivot(index="task",columns="model",values=["R2","RMSE","MAE","n_test"]).reset_index()
    w.columns=["task" if b=="" else f"{a}_{b}" for a,b in w.columns]
    w["delta_R2_tabicl_minus_xgb"]=w.R2_tabicl-w.R2_xgb
    w["test_best_observed_model"]=w.apply(lambda x:"TabICLv2" if x.R2_tabicl>x.R2_xgb else "XGBoost",axis=1)
    w["test_best_observed_R2"]=w[["R2_tabicl","R2_xgb"]].max(axis=1)
    w["at_least_0_5"]=w.test_best_observed_R2.ge(.5)
    w.to_csv(OUT/"05_random_10task_summary.csv",index=False,encoding="utf-8-sig")
    s=pd.read_csv(STATION)
    s=s.loc[s.feature_set.eq("X_strict_77") & s.task.isin(ORDER),["task","model","split_version","n_obs","pooled_R2","fold_R2_mean","RMSE","MAE"]].copy()
    s["model"]=s.model.map({"XGBoost":"xgb","TabICLv2":"tabicl"})
    comparison=m[["task","model","n_test","R2","RMSE","MAE"]].merge(s,on=["task","model"],how="left",validate="one_to_one",suffixes=("_random","_station"))
    if comparison.pooled_R2.isna().any():raise ValueError("Station V2 指标不全")
    comparison["delta_R2_random_minus_station"]=comparison.R2-comparison.pooled_R2
    comparison["comparison_note"]="Random 单次 seed42 测试集 vs Station V2 五折 pooled OOF；样本范围和评价对象不同，不作同协议效应归因"
    comparison.to_csv(OUT/"07_station_v2_comparison.csv",index=False,encoding="utf-8-sig")
    n=int(w.at_least_0_5.sum())
    low=w.loc[~w.at_least_0_5,["task","test_best_observed_R2"]]
    text=["# RANDOM_10TASK_V1 结果判定","",f"- 十项任务均有 XGBoost 与 TabICLv2 的独立 Random Strict 测试集预测。",f"- 按两模型测试集 R² 的较高值进行描述，达到 0.5 的任务为 {n}/10。此“测试集观察最佳”仅用于结果描述，不能当作经独立测试验证的模型选择规则。","- XGBoost 参数候选仅用训练集内部固定划分选择；历史结果保持原样。","- Station V2 是五折留站 pooled OOF；Random 是单次样本随机划分，两者样本范围及评价对象不同。","", "## 未达到 0.5 的任务", ""]
    text += [f"- {r.task}: {r.test_best_observed_R2:.3f}" for r in low.itertuples(index=False)] or ["- 无"]
    (OUT/"08_random_10task_decision.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(w[["task","R2_xgb","R2_tabicl","test_best_observed_model","test_best_observed_R2","at_least_0_5"]].to_string(index=False))
    print(f"R2>=0.5: {n}/10")

if __name__=="__main__":main()
