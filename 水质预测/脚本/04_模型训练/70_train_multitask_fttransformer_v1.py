"""冻结站点折次上的独立与共享 FT-Transformer；不更改任何原始资产。"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/multitask_fttransformer_v1'
SOURCE = ROOT / '结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
FROZEN = SOURCE / '02_冻结候选'
INPUT = FROZEN / '基础输入_77特征.csv'
FEATURES = FROZEN / '77项正式输入清单.csv'
REGISTRY = SOURCE / '03_正式十任务重算/运行登记.csv'
META = FROZEN / '观测元数据.csv'
CONFIG = OUT / 'experiment_config.json'
TASKS = ['chla_surface', 'chla_middle', 'chla_bottom', 'secchi',
         'tn_surface', 'tn_middle', 'tn_bottom', 'tp_surface', 'tp_middle', 'tp_bottom']


class FTTransformer(nn.Module):
    def __init__(self, n_features, n_tasks, cfg):
        super().__init__()
        d = cfg['d_model']
        self.weight = nn.Parameter(torch.randn(n_features, d) * .02)
        self.bias = nn.Parameter(torch.randn(n_features, d) * .02)
        self.missing = nn.Parameter(torch.zeros(n_features, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * .02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=cfg['heads'],
            dim_feedforward=cfg['ffn_dim'], dropout=cfg['dropout'],
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg['layers'], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(d, d//2), nn.GELU(),
            nn.Dropout(cfg['dropout']), nn.Linear(d//2, 1)) for _ in range(n_tasks)])

    def forward(self, x, missing):
        tokens = x.unsqueeze(-1)*self.weight + self.bias + missing.unsqueeze(-1)*self.missing
        cls = self.cls.expand(len(x), -1, -1)
        z = self.norm(self.encoder(torch.cat([cls, tokens], dim=1))[:, 0])
        return torch.cat([head(z) for head in self.heads], dim=1)


def masked_loss(pred, target, valid, mode):
    errors = (pred-target).square()
    if mode == 'M1':
        return (errors*valid).sum()/valid.sum().clamp(min=1)
    per_task = (errors*valid).sum(dim=0)/valid.sum(dim=0).clamp(min=1)
    active = valid.sum(dim=0)>0
    return per_task[active].mean() if active.any() else pred.sum()*0


def prepare_x(x, fit_idx):
    fit = x[fit_idx]
    median = np.nanmedian(fit, axis=0)
    median = np.where(np.isfinite(median), median, 0)
    q25, q75 = np.nanpercentile(fit, [25, 75], axis=0)
    iqr = np.where(np.isfinite(q75-q25) & ((q75-q25)>1e-8), q75-q25, 1)
    missing = ~np.isfinite(x)
    transformed = np.clip((np.where(missing, median, x)-median)/iqr, -10, 10)
    return transformed.astype('float32'), missing.astype('float32')


def prepare_y(y, fit_idx, valid):
    mean = np.zeros(y.shape[1]); std = np.ones(y.shape[1])
    for i in range(y.shape[1]):
        values = y[fit_idx, i][valid[fit_idx, i]]
        if not len(values):
            raise ValueError(f'task {i}: no training labels')
        mean[i] = values.mean()
        std[i] = max(values.std(), 1e-6)
    standardized = np.where(valid, (np.nan_to_num(y)-mean)/std, 0).astype('float32')
    return standardized, mean, std


def read_assets():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    if hashlib.sha256(INPUT.read_bytes()).hexdigest() != cfg['input_sha256']:
        raise ValueError('正式修复版输入 SHA256 不符')
    if hashlib.sha256(FEATURES.read_bytes()).hexdigest() != cfg['feature_manifest_sha256']:
        raise ValueError('正式77特征清单 SHA256 不符')
    names = pd.read_csv(FEATURES).sort_values('顺序').feature.tolist()
    if len(names) != 77 or len(set(names)) != 77 or 'water_zone_code' in names:
        raise ValueError('77项 strict 特征清单异常')
    frame = pd.read_csv(INPUT, usecols=['obs_id', 'station', 'date', 'depth',
        'chla_ug_l', 'secchi_depth_m', 'tn_mg_l', 'tp_mg_l', *names],
        dtype={'obs_id': str}, low_memory=False).set_index('obs_id')
    if not frame.index.is_unique or len(frame) != 30719:
        raise ValueError('正式输入行数或 obs_id 不符')
    registry = pd.read_csv(REGISTRY)
    registered = registry.loc[registry.status.eq('OK') & registry.model.eq('xgboost')]
    if set(registered.task) != set(TASKS) or not registered.dataset_version.eq(cfg['data_version']).all():
        raise ValueError('任务注册表或数据版本不符')
    manifests = {}
    targets = {}
    for task in TASKS:
        rows = registered.loc[registered.task.eq(task)]
        if len(rows) != 5 or rows.target.nunique() != 1 or rows.manifest_path.nunique() != 1:
            raise ValueError(f'{task}: 注册表不完整')
        targets[task] = rows.target.iloc[0]
        if targets[task] not in frame:
            raise ValueError(f'{task}: 正式目标字段不存在')
        manifest_path = Path(rows.manifest_path.iloc[0])
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != rows.manifest_sha256.iloc[0]:
            raise ValueError(f'{task}: 冻结manifest哈希不符')
        m = pd.read_csv(manifest_path, dtype={'obs_id': str})
        if set(m.fold) != {f'fold_{i}' for i in range(1,6)}:
            raise ValueError(f'{task}: 折次不完整')
        manifests[task] = m
    region = pd.read_csv(META, usecols=['obs_id', 'water_zone_en'],
                         dtype={'obs_id': str}).drop_duplicates('obs_id').set_index('obs_id')
    return cfg, names, frame, manifests, targets, region


def fold_arrays(frame, manifests, targets, fold, seed, cfg, task=None):
    # 同一站点若在任一正式任务中是本折测试站点，则该站点所有标签禁入共享训练。
    forbidden = set().union(*(set(m.loc[m.fold.eq(fold) & m.role.eq('test'), 'station_id'])
                              for m in manifests.values()))
    candidates = sorted(set(frame.station) - forbidden)
    rng = np.random.default_rng(42 + int(fold[-1]))
    n_val = max(1, round(len(candidates)*cfg['validation_station_fraction']))
    val_stations = set(rng.choice(candidates, size=n_val, replace=False))
    indexed = frame.reset_index().set_index('obs_id')
    obs = pd.Index(indexed.index)
    n_tasks = 1 if task else len(TASKS)
    active_tasks = [task] if task else TASKS
    y = np.full((len(obs), n_tasks), np.nan, dtype=float)
    allowed = np.zeros_like(y, dtype=bool)
    test_ids = {}
    for j, name in enumerate(active_tasks):
        m = manifests[name].loc[manifests[name].fold.eq(fold)]
        train = m.loc[m.role.eq('train'), 'obs_id'].astype(str)
        test = m.loc[m.role.eq('test'), 'obs_id'].astype(str)
        if not set(test).issubset(set(obs)) or set(test)&set(train):
            raise ValueError(f'{name}/{fold}: 测试ID错位')
        test_ids[name] = pd.Index(test)
        ix = obs.get_indexer(train)
        if (ix < 0).any():
            raise ValueError(f'{name}/{fold}: 训练ID错位')
        allowed[ix, j] = True
        y[:, j] = indexed[targets[name]].to_numpy(float)
    station = indexed.station.to_numpy(str)
    train_station = ~np.isin(station, list(forbidden | val_stations))
    val_station = np.isin(station, list(val_stations))
    label_ok = allowed & np.isfinite(y)
    train_idx = np.flatnonzero(train_station & label_ok.any(axis=1))
    val_idx = np.flatnonzero(val_station & label_ok.any(axis=1))
    if not len(train_idx) or not len(val_idx):
        raise ValueError(f'{fold}: 无训练/验证样本')
    x = indexed[frame.columns.difference(['station','date','depth','chla_ug_l',
        'secchi_depth_m','tn_mg_l','tp_mg_l'], sort=False)].apply(pd.to_numeric, errors='coerce')
    # 特征顺序由调用方通过 frame 中已按清单读取的列位置确定。
    return obs, x, y, label_ok, train_idx, val_idx, test_ids, forbidden, val_stations


def train_one(frame, manifests, targets, names, cfg, fold, seed, variant, region, device):
    task = variant.split(':',1)[1] if variant.startswith('M0:') else None
    mode = 'M0' if task else variant
    obs, x_frame, y, label_ok, train_idx, val_idx, test_ids, forbidden, val_stations = fold_arrays(
        frame, manifests, targets, fold, seed, cfg, task)
    x = x_frame.loc[:, names].to_numpy(float)
    x_scaled, missing = prepare_x(x, train_idx)
    y_scaled, mean, std = prepare_y(y, train_idx, label_ok)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = FTTransformer(len(names), y.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                   weight_decay=cfg['weight_decay'])
    def loader(indices, shuffle):
        dataset = TensorDataset(torch.from_numpy(x_scaled[indices]),
            torch.from_numpy(missing[indices]), torch.from_numpy(y_scaled[indices]),
            torch.from_numpy(label_ok[indices].astype('float32')))
        return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type=='cuda'))
    training = loader(train_idx, True); validation = loader(val_idx, False)
    best = float('inf'); best_state = None; patience = 0; epoch_used = 0
    started = time.perf_counter()
    for epoch in range(cfg['max_epochs']):
        model.train()
        for xb, mb, yb, vb in training:
            xb, mb, yb, vb = (z.to(device, non_blocking=True) for z in (xb, mb, yb, vb))
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(xb, mb), yb, vb, 'M1' if mode=='M1' else 'M2')
            if not torch.isfinite(loss): raise ValueError('nonfinite train loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        model.eval(); total=0; denominator=0
        if mode=='M1':
            with torch.no_grad():
                for xb, mb, yb, vb in validation:
                    xb, mb, yb, vb = (z.to(device, non_blocking=True) for z in (xb, mb, yb, vb))
                    total += float(((model(xb, mb)-yb).square()*vb).sum())
                    denominator += float(vb.sum())
        if mode!='M1':
            val_predictions = predict(model, x_scaled[val_idx], missing[val_idx], cfg, device)
            errors = (val_predictions-y_scaled[val_idx])**2
            task_losses = [errors[label_ok[val_idx,j], j].mean() for j in range(y.shape[1])
                           if label_ok[val_idx,j].any()]
            current = float(np.mean(task_losses))
        else:
            current = total/max(denominator,1)
        if current < best-1e-5:
            best = current; epoch_used = epoch+1; patience = 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg['patience']: break
    if best_state is None: raise ValueError('未产生有效模型权重')
    model.load_state_dict(best_state); model.eval()
    out_rows=[]
    active_tasks=[task] if task else TASKS
    for j,name in enumerate(active_tasks):
        ids = test_ids[name]
        indices = obs.get_indexer(ids)
        pred = predict(model, x_scaled[indices], missing[indices], cfg, device)[:,j]*std[j]+mean[j]
        pred = np.maximum(pred, 0)
        truth = y[indices,j]
        if not np.isfinite(truth).all(): raise ValueError(f'{name}/{fold}: 测试标签缺失')
        out_rows.append(pd.DataFrame({'obs_id': ids, 'task': name, 'fold': fold,
            'seed': seed, 'variant': mode, 'y_true': truth, 'y_pred': pred,
            'station_id': frame.loc[ids, 'station'].to_numpy(),
            'water_zone': region.loc[ids, 'water_zone_en'].to_numpy()}))
    weights = OUT/'weights';weights.mkdir(exist_ok=True, parents=True)
    file_name = f'{variant.replace(":", "_")}_{fold}_seed{seed}.pt'
    torch.save({'model_state':best_state, 'feature_names': names, 'tasks':active_tasks,
                'target_mean':mean, 'target_std':std, 'best_epoch':epoch_used,
                'forbidden_test_station_count':len(forbidden),
                'validation_station_count':len(val_stations), 'config':cfg}, weights/file_name)
    return pd.concat(out_rows, ignore_index=True), {'variant':mode, 'task':task or 'all',
        'fold':fold,'seed':seed,'best_epoch':epoch_used,'best_validation_loss':best,
        'n_training_rows':len(train_idx),'n_validation_rows':len(val_idx),
        'forbidden_test_station_count':len(forbidden),'runtime_sec':time.perf_counter()-started,
        'weight_file':str((weights/file_name).relative_to(ROOT))}


def predict(model, x, missing, cfg, device):
    pieces=[]
    with torch.no_grad():
        for start in range(0,len(x),cfg['batch_size']):
            xb=torch.from_numpy(x[start:start+cfg['batch_size']]).to(device)
            mb=torch.from_numpy(missing[start:start+cfg['batch_size']]).to(device)
            pieces.append(model(xb,mb).cpu().numpy())
    return np.concatenate(pieces)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(1,6))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--variant', choices=['M0','M1','M2'])
    args=parser.parse_args()
    cfg,names,frame,manifests,targets,region=read_assets()
    if cfg['device']=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('独立环境无法使用CUDA；未启动CPU大规模训练')
    device=torch.device(cfg['device'])
    torch.set_num_threads(4)
    folds=[args.fold] if args.fold else range(1,6)
    seeds=[args.seed] if args.seed is not None else cfg['seeds']
    variants=[args.variant] if args.variant else ['M0','M1','M2']
    output=OUT/'predictions';output.mkdir(parents=True, exist_ok=True)
    for f in folds:
      for seed in seeds:
       for mode in variants:
        jobs=[f'M0:{task}' for task in TASKS] if mode=='M0' else [mode]
        for job in jobs:
            path=output/f'{job.replace(":", "_")}_fold_{f}_seed{seed}.csv'
            if path.exists() and path.stat().st_size>100:
                print('REUSE',path.name,flush=True);continue
            rows,record=train_one(frame,manifests,targets,names,cfg,f'fold_{f}',seed,job,region,device)
            rows.to_csv(path,index=False,encoding='utf-8-sig')
            register=OUT/'run_registry.jsonl'
            with register.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(record,ensure_ascii=False)+'\n')
            print('DONE',path.name,'seconds',round(record['runtime_sec'],1),flush=True)


if __name__=='__main__': main()
