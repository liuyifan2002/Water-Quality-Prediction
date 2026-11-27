"""对XGB_SHAP_ALE_V2执行最终加性、覆盖和图件完整性验收。"""
from pathlib import Path
import importlib.util, joblib, json, numpy as np, pandas as pd, xgboost as xgb
from xgboost import XGBRegressor
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'runs/xgb_shap_ale_v2'; R=ROOT/'runs/xgb_interpretation_refit_v1'
spec=importlib.util.spec_from_file_location('p',ROOT/'脚本/05_诊断分析/56_refit_frozen_xgb_for_interpretation.py'); P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
features=P.features(); checks=[]
for task in P.TASKS:
 data=P.load_task_data(task,features)
 for fold in P.FOLDS:
  a=R/'artifacts'/task/fold; ids=pd.read_csv(a/'train_obs_ids.csv',dtype={'obs_id':'string'}).obs_id.astype(str); tr=data[data.obs_id.astype(str).isin(set(ids))].iloc[:256]
  m=XGBRegressor(); m.load_model(a/'xgboost_refit_model.ubj'); imp=joblib.load(a/'median_imputer.joblib'); z=imp.transform(tr[features].to_numpy(float)).astype(np.float32); booster=m.get_booster(); c=booster.predict(xgb.DMatrix(z,feature_names=features),pred_contribs=True); raw=m.predict(z); err=float(np.max(np.abs(c[:,:-1].sum(1)+c[:,-1]-raw)))
  checks.append({'task':task,'fold':fold,'n_checked':len(tr),'shap_additivity_max_abs_error':err,'threshold':1e-5,'status':'PASS' if err<=1e-5 else 'FAIL'})
pd.DataFrame(checks).to_csv(B/'00_preflight/00_shap_additivity_check.csv',index=False,encoding='utf-8-sig')
impf=pd.read_csv(B/'01_shap/01_shap_importance_by_fold.csv').drop(columns=['shap_additivity_max_abs_error'],errors='ignore').merge(pd.DataFrame(checks)[['task','fold','shap_additivity_max_abs_error']],on=['task','fold']); impf.to_csv(B/'01_shap/01_shap_importance_by_fold.csv',index=False,encoding='utf-8-sig')
inventory=pd.read_csv(B/'xgb_shap_ale_v2_figure_inventory.csv'); required=['xgb_shap_ale_v2_final_decision.md','xgb_shap_ale_v2_figure_inventory.csv','xgb_shap_ale_v2_feature_selection_summary.csv','xgb_shap_ale_v2_known_limitations.md','xgb_shap_ale_v2_thesis_results_draft.md']; qa={'models_checked':20,'additivity_pass':sum(x['status']=='PASS' for x in checks),'shap_features':impf.feature.nunique(),'shap_fold_rows':len(impf),'ale_1d_selected':pd.read_csv(B/'02_ale_1d/02_ale_1d_fold_consistency.csv').shape[0],'ale_2d_pairs':pd.read_csv(B/'03_ale_2d/03_ale_2d_fold_consistency.csv').shape[0],'figures':len(inventory),'figure_files_present':int(sum((ROOT/p).exists() for p in list(inventory.png_path)+list(inventory.svg_path))),'required_final_files_present':sum((B/p).exists() for p in required)}
(B/'reports/final_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(qa,ensure_ascii=False))
if qa['additivity_pass']!=20 or qa['shap_features']!=77 or qa['figure_files_present']!=34 or qa['required_final_files_present']!=5: raise SystemExit('FINAL_QA_FAILED')
