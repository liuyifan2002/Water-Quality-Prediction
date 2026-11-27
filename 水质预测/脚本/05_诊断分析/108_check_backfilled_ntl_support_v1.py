"""夜光补齐后确定性分箱支持预检；不计算模型响应或ALE。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '结果/诊断结果/图片数据完整性整改_v1_20261115'
E = OUT / '证据'

def select(frame, n, seed):
    rng = np.random.default_rng(seed)
    pools = {k: list(v) for k, v in frame.groupby('station', sort=True).groups.items()}
    selected = []
    while len(selected) < min(n, len(frame)):
        for station in sorted(pools):
            available = [i for i in pools[station] if i not in selected]
            if available:
                selected.append(int(rng.choice(available)))
                if len(selected) >= min(n, len(frame)):
                    break
    return np.asarray(selected, int)

def main():
    old = pd.read_csv(ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv', low_memory=False)
    new = pd.read_csv(OUT/'修复候选/正式77特征_夜光修复候选_未用于实验.csv', low_memory=False)
    assert old.obs_id.equals(new.obs_id)
    man = pd.read_csv(ROOT/'runs/tn_family_v2/03_tn_surface_manifest.csv')
    rows, cells, coverage = [], [], []
    for version, frame in [('原始冻结输入', old), ('补齐质量候选', new)]:
        for feature in ['ntl_avg_rad', 'ntl_avg_rad_lag2']:
            for dimension in ['station', 'year']:
                group = frame.station if dimension == 'station' else pd.to_datetime(frame.date).dt.year
                for key, g in frame.groupby(group):
                    valid = g.loc[g[feature].notna()]
                    coverage.append(dict(版本=version, 特征=feature, 维度=dimension, 分组=key, 记录数=len(g), 非缺失数=len(valid), 缺失率=float(g[feature].isna().mean()), 唯一值=g[feature].nunique(), 有效开始日期=valid.date.min() if len(valid) else 'NA', 有效结束日期=valid.date.max() if len(valid) else 'NA'))
        d = frame.loc[frame.tn_mg_l.notna() & frame.depth.eq('surface')]
        for k in range(1, 6):
            m = man[man.fold.eq(f'fold_{k}')]
            tr = d[d.obs_id.isin(m.loc[m.role.eq('train'), 'obs_id'])].reset_index(drop=True)
            te = d[d.obs_id.isin(m.loc[m.role.eq('test'), 'obs_id'])].reset_index(drop=True)
            idx = select(te, 64, 30200+k)
            for feature in ['ntl_avg_rad', 'ntl_avg_rad_lag2']:
                rawtr = tr[feature].to_numpy(float)
                rawte = te[feature].to_numpy(float)
                median = float(np.nanmedian(rawtr))
                xtr = np.where(np.isfinite(rawtr), rawtr, median).astype('float32')
                xte = np.where(np.isfinite(rawte), rawte, median).astype('float32')
                edges = np.unique(np.quantile(xtr, np.linspace(0, 1, 11)))
                b = np.clip(np.digitize(xte, edges[1:-1], right=True), 0, len(edges)-2)
                counts = np.bincount(b[idx], minlength=len(edges)-1)
                rows.append(dict(版本=version, 折次=k, 特征=feature, 训练缺失率=float(np.isnan(rawtr).mean()), 训练唯一值=len(np.unique(rawtr[np.isfinite(rawtr)])), 分箱数=len(edges)-1, 参考样本数=len(idx), 参考非缺失数=int(np.isfinite(rawte[idx]).sum()), 非空参考箱=int((counts>0).sum()), 说明='相同历史参考ID与分箱算法的支持预检；无模型响应计算'))
                for bi in range(len(edges)-1):
                    mask = b[idx] == bi
                    valid = mask & np.isfinite(rawte[idx])
                    fullvalid = (b == bi) & np.isfinite(rawte)
                    cells.append(dict(版本=version, 折次=k, 特征=feature, 箱=bi+1, 下界=edges[bi], 上界=edges[bi+1], 参考样本数=int(mask.sum()), 参考站点数=te.iloc[idx[mask]].station.nunique(), 参考真实非缺失数=int(valid.sum()), 参考真实站点数=te.iloc[idx[valid]].station.nunique(), 完整测试非缺失数=int(fullvalid.sum()), 完整测试真实站点数=te.loc[fullvalid, 'station'].nunique()))
    for filename, data in [('补齐前后_TN夜光分箱支持预检.csv', rows), ('补齐前后_TN夜光逐箱真实支持.csv', cells), ('补齐前后_逐站逐年覆盖.csv', coverage)]:
        pd.DataFrame(data).to_csv(E/filename, index=False, encoding='utf-8-sig')
    auth = json.loads((E/'既有项目与授权访问核查.json').read_text(encoding='utf-8'))
    auth['后续动作'] = '官方localhost授权回调成功；随后Initialize及94站132月同源提取成功；旧失败仅保留为历史过程'
    auth['当前状态'] = '已授权并完成12408条同源提取；项目中未保存密码或令牌'
    (E/'既有项目与授权访问核查.json').write_text(json.dumps(auth, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == '__main__':
    main()
