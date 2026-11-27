"""独立读取TabICLv2最终解释实验落盘结果并复核关键数值。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"runs/tabicl_v2_final_explanation_complete_v1"
FEATURES=ROOT/"runs/tabiclv2_pilot/01_strict_feature_manifest.csv"

def plot_actual_results():
    """只绘制已落盘实际数据；覆盖/协议状态同时写入图件清单。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    png=OUT/"figures/png";svg=OUT/"figures/svg";png.mkdir(parents=True,exist_ok=True);svg.mkdir(parents=True,exist_ok=True)
    inventory=[]
    def save(fig,key,title,scope):
        fig.savefig(png/f"{key}.png",dpi=300,bbox_inches="tight",facecolor="white")
        fig.savefig(svg/f"{key}.svg",bbox_inches="tight",facecolor="white")
        plt.close(fig)
        inventory.append({"figure":key,"title":title,"scope":scope,"status":"GENERATED_FROM_SAVED_DATA",
            "png":str((png/f"{key}.png").relative_to(OUT)),"svg":str((svg/f"{key}.svg").relative_to(OUT))})
    pa=OUT/"01_pointwise_inference_validation.csv"
    if pa.is_file() and pa.stat().st_size:
        a=pd.read_csv(pa)
        if len(a):
            fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(a));y=a.max_cross_process_abs_diff.to_numpy(float)
            colors=["#27864b" if s=="STABLE" else "#bb4a3a" for s in a.status]
            ax.bar(x,y,color=colors);ax.axhline(1e-6,color="#333333",linestyle="--",label="strict threshold 1e-6")
            ax.set_yscale("symlog",linthresh=1e-7);ax.set_xticks(x,[f"{r.task}\n{r.fold}" for r in a.itertuples()],rotation=65,ha="right",fontsize=8)
            ax.set_ylabel("Maximum cross-process absolute prediction difference");ax.set_title("Pointwise prediction repeatability by task and fold")
            ax.legend(frameon=False);ax.grid(axis="y",alpha=.2);save(fig,"pointwise_repeatability_by_fold","Pointwise repeatability by fold","20-fold protocol diagnostics")
    pp=OUT/"02_original_vs_pointwise_prediction.csv"
    if pp.is_file() and pp.stat().st_size:
        d=pd.read_csv(pp)
        if len(d):
            fig,ax=plt.subplots(figsize=(6,6));
            for task,g in d.groupby("task"):
                ax.scatter(g.y_pred,g.pointwise_prediction,s=12,alpha=.45,label=task)
            lo=float(np.nanmin(np.r_[d.y_pred,d.pointwise_prediction]));hi=float(np.nanmax(np.r_[d.y_pred,d.pointwise_prediction]));ax.plot([lo,hi],[lo,hi],"k--",lw=1)
            ax.set_xlabel("Historical frozen batch prediction (target units)");ax.set_ylabel("New pointwise prediction (target units)")
            ax.set_title("Historical vs pointwise predictions (probe/full rows as labeled in CSV)");ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
            save(fig,"historical_vs_pointwise_predictions","Historical versus pointwise predictions","obs_id-matched actual stored predictions")
    sf=OUT/"03_shap_full_results.csv"
    if sf.is_file() and sf.stat().st_size:
        d=pd.read_csv(sf)
        if len(d):
            tasks=sorted(d.task.unique());fig,axs=plt.subplots(len(tasks),1,figsize=(9,max(3,2.8*len(tasks))),squeeze=False)
            for ax,task in zip(axs[:,0],tasks):
                g=d[d.task==task].groupby("feature").shap_value.apply(lambda z:np.mean(np.abs(z))).nlargest(15).sort_values()
                ax.barh(g.index,g.values,color="#2a7894");ax.set_xlabel("Mean absolute SHAP (original target scale)");ax.set_title(task);ax.grid(axis="x",alpha=.18)
            fig.tight_layout();save(fig,"tabicl_shap_top_features_by_task","TabICLv2 model-agnostic SHAP top features","Only completed pointwise protocol samples; separate task panels")
    a1=OUT/"05_ale_1d_results.csv"
    if a1.is_file() and a1.stat().st_size:
        d=pd.read_csv(a1)
        if len(d):
            unit_map={}
            xu=ROOT/"runs/xgb_shap_ale_v2/01_shap/01_shap_importance_summary.csv"
            if xu.is_file():
                ud=pd.read_csv(xu)
                if "unit" in ud.columns: unit_map=ud.drop_duplicates("feature").set_index("feature").unit.astype(str).to_dict()
            for (task,feature),g in d.groupby(["task","feature"]):
                fig,ax=plt.subplots(figsize=(6,4))
                for fold,h in g.groupby("fold"):
                    h=h.sort_values("bin");ax.plot(h.center,h.ale,marker="o",ms=2,lw=1,label=fold)
                ax.axhline(0,color="#555555",lw=.7);ax.set_xlabel(f"{feature} ({unit_map.get(feature,'unit not specified in frozen taxonomy')})");ax.set_ylabel("ALE (original target units)")
                ax.set_title(f"{task}: {feature} — pointwise ALE; supported bins only");ax.legend(frameon=False,ncol=3,fontsize=8);ax.grid(alpha=.18)
                fig.tight_layout();save(fig,f"ale1_{task}_{feature}",f"1D ALE: {task}, {feature}","Only supported bins from saved results")
    a2=OUT/"07_ale_2d_results.csv"
    if a2.is_file() and a2.stat().st_size:
        d=pd.read_csv(a2)
        if len(d):
            for (task,f1,f2,grid),g in d.groupby(["task","feature_1","feature_2","grid"]):
                p1=g.pivot(index="cell_1",columns="cell_2",values="ale_2d").sort_index(ascending=False)
                fig,ax=plt.subplots(figsize=(5.5,4.5));im=ax.imshow(p1.to_numpy(float),aspect="auto",cmap="RdBu_r")
                ax.set_xlabel(f2);ax.set_ylabel(f1);ax.set_title(f"{task}: 2D ALE {grid}; supported rectangle only")
                fig.colorbar(im,ax=ax,label="Second-order ALE (target units)");fig.tight_layout()
                save(fig,f"ale2_{task}_{f1}_{f2}_{grid}",f"2D ALE: {task}, {f1} × {f2}","Supported rectangle only; sparse regions omitted")
    if inventory:
        pd.DataFrame(inventory).to_csv(OUT/"10_explanation_figure_inventory.csv",index=False,encoding="utf-8-sig")

def check(name, passed, value, criterion, detail=""):
    return {"check":name,"passed":bool(passed),"observed":value,"criterion":criterion,"detail":detail}

def main():
    rows=[]
    manifest=pd.read_csv(FEATURES)
    features=manifest.loc[manifest.included_in_pilot.astype(bool),"feature"].astype(str).tolist()
    rows.append(check("strict_feature_manifest_77",len(features)==77 and len(set(features))==77,len(features),"77 unique strict features"))
    p=OUT/"pointwise_probe_predictions.csv"
    if p.is_file() and p.stat().st_size:
        d=pd.read_csv(p)
        for (task,fold),g in d.groupby(["task","fold"]):
            counts=g.groupby("obs_id").replicate.nunique()
            rng=g.groupby("obs_id").pointwise_prediction.agg(lambda x:float(x.max()-x.min()))
            maxdiff=float(rng.max());meandiff=float(rng.mean())
            rows.append(check(f"{task}_{fold}_cross_process_repeat",bool((counts==3).all() and maxdiff<=1e-6 and meandiff<=1e-7),
                {"n_obs":len(counts),"max_diff":maxdiff,"mean_diff":meandiff},"3 processes; max<=1e-6; mean<=1e-7"))
    sf=OUT/"03_shap_full_results.csv"
    if sf.is_file() and sf.stat().st_size:
        s=pd.read_csv(sf)
        for (task,fold,obs),g in s.groupby(["task","fold","obs_id"]):
            good=(len(g)==77 and g.feature.astype(str).tolist()==features and np.isfinite(g.shap_value).all() and
                  np.isfinite(g.base_value).all() and np.isfinite(g.model_prediction).all())
            err=float(abs(g.base_value.iloc[0]+g.shap_value.sum()-g.model_prediction.iloc[0])) if len(g) else float("inf")
            rows.append(check(f"{task}_{fold}_{obs}_shap_shape_additivity",good and err<=.02,
                {"features":len(g),"additivity_abs_error":err},"77 ordered finite SHAP values; abs additivity error<=0.02"))
    runtime=OUT/"shap_runtime.csv"
    if runtime.is_file() and runtime.stat().st_size:
        r=pd.read_csv(runtime)
        for z in r.itertuples():
            calls=getattr(z,"actual_model_predict_calls",getattr(z,"model_predict_calls",0))
            if pd.isna(calls):calls=getattr(z,"model_predict_calls",0)
            rows.append(check(f"{z.task}_{z.fold}_{z.obs_id}_shap_runtime",int(calls)>0 and np.isfinite(float(z.elapsed_seconds)),
                {"model_predict_calls":int(calls),"elapsed_seconds":float(z.elapsed_seconds),"additivity_abs_error":float(z.additivity_abs_error)},
                "positive actual model-call count and finite elapsed/additivity record"))
    a1=OUT/"05_ale_1d_results.csv"
    if a1.is_file() and a1.stat().st_size:
        a=pd.read_csv(a1)
        for (task,fold,feature),g in a.groupby(["task","fold","feature"]):
            support=bool((g.n_samples>=30).all() and (g.n_stations>=5).all())
            center=float(np.average(g.ale,weights=g.n_samples.clip(lower=1)))
            rows.append(check(f"{task}_{fold}_{feature}_ale_coverage_centering",support and abs(center)<=1e-5,
                {"bins":len(g),"min_n":int(g.n_samples.min()),"min_stations":int(g.n_stations.min()),"weighted_center":center},
                "each bin n>=30 and stations>=5; weighted ALE center abs<=1e-5"))
    repeats=OUT/"ale_1d_repeatability.csv"
    if repeats.is_file() and repeats.stat().st_size:
        r=pd.read_csv(repeats)
        for z in r.itertuples():
            delta=float(z.max_local_ale_delta_range)
            rows.append(check(f"{z.task}_{z.fold}_{z.feature}_ale_repeatability",int(z.n_repeats)==3 and delta<=1e-6,
                {"n_repeats":int(z.n_repeats),"max_lower_delta":float(z.max_lower_prediction_delta),
                 "max_upper_delta":float(z.max_upper_prediction_delta),"max_local_effect_delta_range":delta},
                "three uncached P1 repeats; local-effect range<=1e-6"))
    a2=OUT/"07_ale_2d_results.csv"
    if a2.is_file() and a2.stat().st_size:
        a=pd.read_csv(a2)
        for (task,fold,x,y),g in a.groupby(["task","fold","feature_1","feature_2"]):
            supported=g.loc[~g.sparse_mask.astype(bool)]
            rows.append(check(f"{task}_{fold}_{x}_{y}_2d_support",bool(len(supported)>0),
                {"cells":len(g),"supported_cells":len(supported)},"only explicitly supported cells are interpreted"))
    val=pd.DataFrame(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    val.to_csv(OUT/"validation_checks.csv",index=False,encoding="utf-8-sig")
    plot_actual_results()
    failures=val.loc[~val.passed]
    (OUT/"independent_validation_report.md").write_text(
        "# 独立结果核验\n\n"+
        f"核验项：{len(val)}；通过：{int(val.passed.sum())}；未通过：{len(failures)}。\n\n"+
        "逐项数值见`validation_checks.csv`。核验脚本不调用训练/预测函数，仅读取已保存结果，避免与生成程序共用计算路径。\n",
        encoding="utf-8")
    print(f"checks={len(val)} passed={int(val.passed.sum())} failed={len(failures)}")

if __name__=="__main__":main()
