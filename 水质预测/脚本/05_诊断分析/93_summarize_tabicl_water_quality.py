"""Summarize frozen baselines and completed TabICLv2 adaptation candidates."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import importlib.metadata as package_metadata
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/tabicl_water_quality_adaptation_v1'
ZOO = ROOT / 'runs/random_tabular_model_zoo_v2'
SOURCE = ROOT / '脚本/04_模型训练/67_run_tabicl_water_quality_inner.py'
spec = importlib.util.spec_from_file_location('tabicl_adaptation_inner', SOURCE)
inner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inner)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(frame):
    y, p = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
    return dict(R2=r2_score(y, p), RMSE=mean_squared_error(y, p) ** .5,
                MAE=mean_absolute_error(y, p))


def baseline():
    old = pd.read_csv(ZOO / '08_all_models_test_comparison.csv')
    rows = []
    for task in inner.TASKS:
        row = old.loc[(old.task == task) & (old.model == 'tabicl')].iloc[0]
        oof = pd.read_csv(ZOO / '02_inner_oof' / f'{task}_tabicl.csv')
        s = score(oof)
        rows.append(dict(task=task, config_id='Z0', inner_oof_R2=s['R2'],
                         inner_oof_RMSE=s['RMSE'], inner_oof_MAE=s['MAE'],
                         old_test_R2=row.R2, old_test_RMSE=row.RMSE,
                         old_test_MAE=row.MAE, old_test_n=int(row.n_test)))
    pd.DataFrame(rows).to_csv(OUT / '01_baseline_tabicl_results.csv', index=False, encoding='utf-8-sig')


def candidates():
    rows = []
    for key, value in inner.CONFIGS.items():
        rows.append(dict(config_id=key, settings=json.dumps(value, ensure_ascii=False),
                         stage=('baseline' if key == 'Z0' else 'zero_shot' if key.startswith(('Z','P','N','O')) or key == 'MEDIAN' else 'target' if key in {'RAW','YEO'} else 'feature'),
                         note='Z1 none+power与官方默认一致，直接复用Z0' if key == 'Z0' else ''))
    pd.DataFrame(rows).to_csv(OUT / '02_candidate_configurations.csv', index=False, encoding='utf-8-sig')


def metrics():
    rows = []
    runtime = []
    for task in inner.TASKS:
        baseline_oof = pd.read_csv(ZOO / '02_inner_oof' / f'{task}_tabicl.csv')
        baseline_oof['fold'] = baseline_oof.inner_fold
        for config in [*inner.CONFIGS, 'FT1', 'FT1_CLIP', 'FT2', 'FT2_CLIP', 'FT3', 'FT3_CLIP',
                       'FT1_CHUNK2048', 'FT1_CHUNK2048_CLIP']:
            if config == 'Z0':
                parts = [baseline_oof]
            else:
                paths = sorted((OUT / '07_inner_oof_predictions').glob(f'{task}_{config}_fold*.csv'))
                if not paths:
                    continue
                parts = [pd.read_csv(p) for p in paths]
            joined = pd.concat(parts, ignore_index=True)
            folds = sorted(joined.fold.unique())
            fold_scores = [score(joined.loc[joined.fold == fold])['R2'] for fold in folds]
            rows.append(dict(task=task, config_id=config, folds=len(folds),
                             n=len(joined), **score(joined), fold_R2_mean=sum(fold_scores)/len(fold_scores),
                             fold_R2_sd=float(pd.Series(fold_scores).std(ddof=1)) if len(folds)>1 else float('nan')))
            if config != 'Z0':
                for fold in folds:
                    meta = OUT / 'inner_fold_metrics' / f'{task}_{config}_fold{fold}.json'
                    if meta.exists(): runtime.append(dict(stage='inner_cv', **json.loads(meta.read_text(encoding='utf-8'))))
    for task in inner.TASKS:
        final_meta = OUT / '08_final_test_predictions' / f'{task}_tabicl_adapted.json'
        if final_meta.exists():
            runtime.append(dict(stage='external_test', **json.loads(final_meta.read_text(encoding='utf-8'))))
    all_metrics = pd.DataFrame(rows)
    all_metrics.loc[all_metrics.config_id.isin(['Z0','Z2','Z3','Z4','Z5','P_shift','P_random','P_none','N16','N24','O3','O6','MEDIAN'])].to_csv(
        OUT / '03_inference_optimization_metrics.csv', index=False, encoding='utf-8-sig')
    all_metrics.loc[all_metrics.config_id.str.startswith('FT')].to_csv(
        OUT / '04_finetuning_metrics.csv', index=False, encoding='utf-8-sig')
    all_metrics.loc[all_metrics.config_id.isin(['Z0','F60','F40','I4'])].to_csv(
        OUT / '05_feature_adaptation_metrics.csv', index=False, encoding='utf-8-sig')
    all_metrics.loc[all_metrics.config_id.isin(['Z0','RAW','YEO','NOCLIP'])].to_csv(
        OUT / '06_target_transform_metrics.csv', index=False, encoding='utf-8-sig')
    all_metrics.to_csv(OUT / '03_all_inner_candidate_metrics.csv', index=False, encoding='utf-8-sig')
    if runtime:
        pd.DataFrame(runtime).to_csv(OUT / '12_runtime_and_gpu_metrics.csv', index=False, encoding='utf-8-sig')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline(); candidates(); metrics()
    checkpoint=ROOT/'runs/tabicl_explanation_portable_bundle_v2/model/tabicl-regressor-v2-20260212.ckpt'
    protocol = f'''# TabICLv2近岸水质领域适配实验协议\n\n- 实验：TABICLV2_WATER_QUALITY_ADAPTATION_V1，独立扩展实验。\n- 数据：正式主表原样读取，77项strict特征；不含water_zone_code。\n- 外部划分：复用RANDOM_10TASK_V1，split SHA256={sha(ROOT / 'runs/random_10task_v1/01_random_split_manifest.csv')}。\n- 内部划分：复用Model Zoo V2的01_inner_fold_manifest.csv，SHA256={sha(ZOO / '01_inner_fold_manifest.csv')}；全部调参仅使用外部训练集。\n- 特征顺序：Pilot 77项manifest，SHA256={sha(inner.FEATURE_MANIFEST)}。\n- 模型身份：tabicl=={package_metadata.version('tabicl')}，torch=={package_metadata.version('torch')}，回归权重SHA256={sha(checkpoint)}；权重与本机官方缓存哈希一致。\n- 原始基准：复用Model Zoo V2 TabICLv2内部OOF及外部测试预测；Z1与默认none+power相同，不重算。\n- 选择：首轮最多五种推理配置、同一内部折次初筛；最多两种进入完整五折。最终以完整五折pooled OOF R²为主，RMSE/MAE辅助。\n- 目标变换、缺失填补与特征选择均仅在各内部训练折拟合；原始水质单位上评价。\n- 外部测试集只在内部配置锁定后评价一次；不能根据测试分数改选。\n- 官方微调在隔离依赖目录内完成TN中层一个内部折代表试验；早停后未优于零样本基线，未扩展至十任务或参与外部测试。\n- 不修改正式主链、冻结预测、split、77项特征或历史模型资产。\n'''
    (OUT / '00_experiment_protocol.md').write_text(protocol, encoding='utf-8')


if __name__ == '__main__':
    main()
