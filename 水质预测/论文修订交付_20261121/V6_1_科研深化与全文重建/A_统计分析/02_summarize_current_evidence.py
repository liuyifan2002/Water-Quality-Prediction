"""从冻结汇总和本目录配对统计生成中文研究分析，不启动模型。"""
from pathlib import Path
import hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
OUT=Path(__file__).resolve().parent

def f(x,n=3): return f'{x:.{n}f}'
def main():
    b=pd.read_csv(OUT/'十任务Bootstrap结果.csv')
    c=pd.read_csv(OUT/'77特征真实观测覆盖.csv')
    a=pd.read_csv(SRC/'04_来源消融重算/四任务六来源_新版本完整对照.csv')
    m=pd.read_csv(SRC/'08_重算结果核验/新候选_pooled与逐折指标.csv')
    z=pd.read_csv(SRC/'06_历史78特征分区重算/五折留站点验证_汇总.csv')
    taxonomy=ROOT/'runs/multisource_ablation_v1/00_feature_source_taxonomy.csv'
    assert hashlib.sha256(taxonomy.read_bytes()).hexdigest()=='b8141c13a0d376986000de8fc0f5d9bac9f33c9545820cf8b4bc1eee243de9bc'
    tax=pd.read_csv(taxonomy)
    assert set(tax.feature_name)==set(c.feature)
    cg=c.merge(tax[['feature_name','source_group']],left_on='feature',right_on='feature_name',validate='one_to_one')
    cov=cg.groupby('source_group').agg(n_features=('feature','count'),mean_coverage=('coverage_rate','mean'),minimum_coverage=('coverage_rate','min'),minimum_stations=('n_stations_valid','min')).reset_index()
    cov.to_csv(OUT/'六来源真实覆盖汇总.csv',index=False,encoding='utf-8-sig')
    assert set(a.dataset_version)=={'WQ77_REPAIRED_20261115_V1'} and len(a)==24
    assert len(z)==12 and (z.folds==5).all()
    print('modern',m[m['范围']=='现代模型'][['task','model','pooled_R2']].to_string(index=False))
    lines=['# 当前版本实验结果综合分析','',
           '本报告只使用 `WQ77_REPAIRED_20261115_V1` 的冻结预测和候选版实验汇总；未训练或重新预测。站点聚类 Bootstrap 使用固定配对 `obs_id`、五折测试样本、1000 次有放回站点抽样、随机种子 20260921、百分位 95% 区间。该种子沿用 Chl-a/TN 家族汇总程序；旧 Secchi 单独审查曾用 42，故本表十任务统一采用 20260921。Δ 指标均为 TabICLv2 减 XGBoost，负 ΔRMSE/ΔMAE 表示误差降低。','',
           '## 正式十任务预测与统计边界','',
           '| 任务 | XGB R² | TabICLv2 R² | ΔR² | ΔRMSE | ΔRMSE 95%CI | 站点胜率 |','|---|---:|---:|---:|---:|---:|---:|']
    for _,x in b.iterrows():
        lines.append(f"| {x.task} | {f(x.XGBoost_pooled_R2)} | {f(x.TabICLv2_pooled_R2)} | {f(x.delta_pooled_R2)} | {f(x.delta_pooled_RMSE,4)} | [{f(x.delta_RMSE_CI_low,4)}, {f(x.delta_RMSE_CI_high,4)}] | {f(100*x.station_RMSE_win_rate,1)}% |")
    lines+=['',
        'TabICLv2 的 pooled R² 在九项任务高于 XGBoost，TN 中层低 0.0115。Chl-a 三层和 Secchi 的 ΔRMSE 区间均小于零，提供较一致的误差降低证据；TN 表层、TN 中层、TN 底层及 TP 三层的区间跨零，不能称其误差具有明确统计优势。TN 中层总体 RMSE 略增，但多数站点改善，说明样本加权总体指标可能受少数站点较大误差影响；此句只是从站点与 pooled 指标并列观察得到的推断，具体驱动站点需参考逐站点表。不同水质指标量纲不同，不能横向比较绝对 RMSE。',
        'TP 表层的 TabICLv2 pooled R² 为 0.234，TP 中层为 0.378，明显低于 Secchi 的 0.627；这说明预测难度具有任务依赖性。近岸径流、潮汐交换、生物过程及观测尺度可能共同影响可预测性，但目前模型对比不能区分这些机制的因果作用。',
        '', '## 四模型比较（同协议冻结结果）','',
        '| 任务 | XGBoost | TabICLv2 | TabPFN | TabM |','|---|---:|---:|---:|---:|']
    for task in ['chla_surface','secchi','tn_surface','tp_middle']:
        vals={x.model: x.pooled_R2 for _,x in m[m.task==task].iterrows() if x['范围'] in ['正式十任务','现代模型']}
        assert set(vals)=={'xgboost','tabiclv2','tabpfn','tabm'},(task,vals)
        lines.append('| '+task+' | '+' | '.join(f(vals[k]) for k in ['xgboost','tabiclv2','tabpfn','tabm'])+' |')
    lines+=['','四模型比较限于上述四项代表任务，不外推十任务。各模型评分均来自当前版本相同测试任务的 pooled OOF 汇总。','',
        '## 六来源删组消融（XGBoost，四任务）','',
        '实验登记为四任务×六来源×五折，共 120 任务折；源组映射按冻结 taxonomy 哈希 `b8141c13...`，变量数量分别为 HUM 6、MET 29、OCN 11、RHT 22、SPA 2、TMP 7。ΔR²=删组模型减完整模型；负值表示删除后性能下降。该证据不等于单变量 SHAP 归因或环境因果效应。','',
        '| 任务 | 删除后性能下降的来源（ΔR²<0） | 删除后性能提高的来源（ΔR²>0） |','|---|---|---|']
    for task,g in a.groupby('task',sort=False):
        neg=', '.join(f"{x.source_group_removed} {f(x.delta_R2)}" for _,x in g[g.delta_R2<0].sort_values('delta_R2').iterrows())
        pos=', '.join(f"{x.source_group_removed} +{f(x.delta_R2)}" for _,x in g[g.delta_R2>0].sort_values('delta_R2',ascending=False).iterrows())
        lines.append(f'| {task} | {neg or "无"} | {pos or "无"} |')
    lines+=['','空间静态组在 Chl-a 表层和 Secchi 删组后 R² 分别下降 0.072 与 0.113；而在 TN 表层、TP 中层并未出现同方向同量级效应。TP 中层删去 RHT、HUM、SPA 后反而改善，说明特征组合在该任务下的泛化效用有限，不能推断原始来源具有负面环境作用。单来源删组无法识别来源间的交互或替代关系，互补性仅可作为待检验假说。','',
        '## 历史78特征分区实验（独立协议）','',
        '| 任务 | pooled ΔR²（分区－全域） | 改善折数/5 |','|---|---:|---:|']
    for _,x in z.iterrows():lines.append(f'| {x.target}_{x.depth} | {f(x.delta_r2_pooled)} | {x.positive_folds}/5 |')
    lines+=['','Chl-a 三层及历史 Secchi 三层的 pooled ΔR² 均为正；TN 中层、表层和 TP 三层为负（TN 底层略正）。分区收益存在明显目标依赖，尤其 TP 中层、表层退化。该历史实验采用78特征、包含 `water_zone_code`、12个目标×水层组合和60个任务折；历史 Secchi 三层不能改称正式77特征的三项 Secchi 任务。分区模型的全域回退决策也不等于每个海区都独立训练。','',
        '## 当前77项输入观测覆盖','',
        '基础冻结输入共 30,719 行、94 个站、2014-01-02 至 2024-12-30；以下为填补前的原始有效覆盖，不代表遥感每个像元的真实观测或发布时点可得性。','',
        '| 来源组 | 特征数 | 平均非缺失率 | 最低非缺失率 | 最少有效站点数 |','|---|---:|---:|---:|---:|']
    for _,x in cov.iterrows():lines.append(f'| {x.source_group} | {int(x.n_features)} | {f(100*x.mean_coverage,1)}% | {f(100*x.minimum_coverage,1)}% | {int(x.minimum_stations)} |')
    lines+=['','全部77特征的最低非缺失率为 96.1%（`ntl_avg_rad_delta1`），不能表述为无缺失。`四站当前特征覆盖.csv` 精确列出 VT11、SM18、SM19、PT3 的每特征覆盖；跨水层目标的诊断须按各任务实际样本子集重算，不能直接把全表站点覆盖称作中层或底层覆盖。`rain_month` 为采样自然月整月累计，气象、夜光、潮汐存在同期信息，本研究结论限于历史同期预测与留站点泛化。','',
        '## 可复现性和限制','',
        '- 冻结逐折预测在配对前校验了 `obs_id`、站点、日期、折次、真实值和 manifest SHA256；10任务×2模型×5折均为当前版本。每站点的记录数不等，重抽样后指标按所抽站点的全部记录 pooled 计算。',
        '- Bootstrap 的 95% 区间是站点抽样不确定性描述，不是因果检验；端点是否跨零见 CSV。',
        '- 本轮未重新训练、预测、计算 SHAP/ALE 或修改冻结数据。']
    (OUT/'当前版本实验结果综合分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
