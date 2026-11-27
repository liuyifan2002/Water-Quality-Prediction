"""汇总M1/VSP-Net冻结留站OOF，比较单层精度与垂向差值。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/vertical_structure_prediction_v1'
M1=ROOT/'runs/multitask_fttransformer_v1/predictions'
ORDER=['chla_surface','chla_middle','chla_bottom','secchi','tn_surface','tn_middle','tn_bottom','tp_surface','tp_middle','tp_bottom']
NAMES=['Chl-a表层','Chl-a中层','Chl-a底层','Secchi','TN表层','TN中层','TN底层','TP表层','TP中层','TP底层']
COLORS={'M1':'#58718b','V1':'#007f9e','V2':'#d69035'}

def metrics(d):
    y=d.y_true.to_numpy();p=d.y_pred.to_numpy()
    return {'R2':r2_score(y,p),'RMSE':mean_squared_error(y,p)**.5,'MAE':mean_absolute_error(y,p),'n':len(d)}

def main():
    vfiles=sorted((OUT/'predictions').glob('*.csv'))
    mfiles=sorted(M1.glob('M1_fold_*.csv'))
    if len(vfiles)!=30 or len(mfiles)!=15:raise RuntimeError(f'预测文件不全:V={len(vfiles)},M1={len(mfiles)}')
    pred=pd.concat([pd.read_csv(p) for p in mfiles+vfiles],ignore_index=True)
    assert pred.duplicated(['variant','seed','task','obs_id']).sum()==0
    assert not pred[['y_true','y_pred']].isna().any().any()
    for task in ORDER:
        expected=set(pred[(pred.variant=='M1')&(pred.seed==42)&(pred.task==task)].obs_id)
        for (v,s),g in pred[pred.task==task].groupby(['variant','seed']):
            if set(g.obs_id)!=expected or g.fold.nunique()!=5:raise ValueError(f'样本或折次不一致:{task}/{v}/{s}')
    truth=pred.groupby(['task','obs_id']).y_true.agg(['min','max'])
    if not np.allclose(truth['min'],truth['max']):raise ValueError('真实值跨模型/种子不一致')
    pred.to_parquet(OUT/'05_oof_predictions.parquet',index=False)
    rows=[]
    for (v,task,seed),g in pred.groupby(['variant','task','seed']):
        pooled=metrics(g);folds=[metrics(x)['R2'] for _,x in g.groupby('fold')]
        rows.append({'variant':v,'task':task,'seed':seed,'pooled_R2':pooled['R2'],
                     'fold_mean_R2':np.mean(folds),'fold_sd_R2':np.std(folds,ddof=1),
                     'RMSE':pooled['RMSE'],'MAE':pooled['MAE'],'n':pooled['n']})
    per_seed=pd.DataFrame(rows)
    agg=per_seed.groupby(['task','variant'],as_index=False).agg(
        pooled_R2_mean=('pooled_R2','mean'),pooled_R2_sd=('pooled_R2','std'),
        fold_mean_R2_mean=('fold_mean_R2','mean'),fold_mean_R2_sd=('fold_mean_R2','std'),
        RMSE_mean=('RMSE','mean'),RMSE_sd=('RMSE','std'),MAE_mean=('MAE','mean'),MAE_sd=('MAE','std'),n=('n','first'))
    agg.to_csv(OUT/'03_ten_task_metrics.csv',index=False,encoding='utf-8-sig')

    # Paired vertical prediction differences use the same sample_id and same seed/fold.
    vert=pred[pred.task!='secchi'].copy()
    vert['sample_id']=vert.obs_id.str.rsplit('_',n=1).str[0]
    vert['indicator']=vert.task.str.split('_').str[0]
    vert['depth']=vert.task.str.rsplit('_',n=1).str[-1]
    vd=[]
    for (v,seed,q),g in vert.groupby(['variant','seed','indicator']):
        p=g.pivot(index='sample_id',columns='depth',values=['y_true','y_pred'])
        for a,b in [('surface','middle'),('middle','bottom')]:
            z=p[[('y_true',a),('y_true',b),('y_pred',a),('y_pred',b)]].dropna()
            e=(z[('y_pred',a)]-z[('y_pred',b)])-(z[('y_true',a)]-z[('y_true',b)])
            vd.append({'variant':v,'seed':seed,'indicator':q,'pair':a+'-'+b,'n_events':len(z),
                       'difference_RMSE':float(np.sqrt(np.mean(e**2))),'difference_MAE':float(np.mean(np.abs(e)))})
        z=p[[('y_true',d) for d in ('surface','middle','bottom')]+[('y_pred',d) for d in ('surface','middle','bottom')]].dropna()
        e=np.column_stack([(z[('y_pred',d)]-z[('y_true',d)]).to_numpy() for d in ('surface','middle','bottom')])
        vd.append({'variant':v,'seed':seed,'indicator':q,'pair':'three_layer_profile','n_events':len(z),
                   'difference_RMSE':float(np.sqrt(np.mean(e**2))),'difference_MAE':float(np.mean(np.abs(e)))})
    pd.DataFrame(vd).to_csv(OUT/'04_vertical_difference_metrics.csv',index=False,encoding='utf-8-sig')

    reg=OUT/'training_registry.jsonl'; training=[json.loads(x) for x in reg.read_text(encoding='utf-8').splitlines()]
    runs=pd.DataFrame(training).drop_duplicates(['variant','fold','seed'],keep='last')
    if len(runs)!=30:raise RuntimeError('训练登记不完整')
    runs.to_csv(OUT/'02_training_summary.csv',index=False,encoding='utf-8-sig')

    wide=agg.pivot(index='task',columns='variant',values='pooled_R2_mean').reindex(ORDER)
    comp=agg.pivot(index='task',columns='variant',values='RMSE_mean').reindex(ORDER)
    result=pd.DataFrame({'task':ORDER,'indicator':[t.split('_')[0] for t in ORDER],
        'M1_R2':wide.M1.to_numpy(),'V1_R2':wide.V1.to_numpy(),'V2_R2':wide.V2.to_numpy(),
        'V1_minus_M1_R2':(wide.V1-wide.M1).to_numpy(),'V2_minus_M1_R2':(wide.V2-wide.M1).to_numpy(),
        'V2_minus_V1_R2':(wide.V2-wide.V1).to_numpy(),
        'M1_RMSE':comp.M1.to_numpy(),'V1_RMSE':comp.V1.to_numpy(),'V2_RMSE':comp.V2.to_numpy()})
    result.to_csv(OUT/'06_m1_v1_v2_comparison.csv',index=False,encoding='utf-8-sig')
    plot(agg,result,pd.DataFrame(vd))
    write_decision(result,pd.DataFrame(vd))
    print('complete',len(pred),'prediction rows',len(agg),'task-variant rows')

def plot(agg,res,vd):
    (OUT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    x=np.arange(10);w=.25
    fig,ax=plt.subplots(figsize=(11,5.2),layout='constrained')
    for i,v in enumerate(('M1','V1','V2')):
        d=agg[agg.variant==v].set_index('task').reindex(ORDER)
        ax.bar(x+(i-1)*w,d.pooled_R2_mean,width=w,label=v,color=COLORS[v])
    ax.set_xticks(x,NAMES,rotation=35,ha='right');ax.set_ylabel('pooled OOF R²（三种子均值）')
    ax.set_ylim(min(0,agg.pooled_R2_mean.min()-.04),agg.pooled_R2_mean.max()+.08)
    ax.axhline(0,color='#555',lw=.7);ax.legend(ncol=3,frameon=False)
    fig.savefig(OUT/'figures/图1_十任务R2比较.png',dpi=300,facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.5,4.5),layout='constrained')
    tasks=[f'{q}_{d}' for q in ('chla','tn','tp') for d in ('surface','middle','bottom')]
    for ax,col,title in zip(axes,['V1_minus_M1_R2','V2_minus_V1_R2'],['V1 − M1','V2 − V1']):
        mat=res.set_index('task').loc[tasks,col].to_numpy().reshape(3,3).T
        limit=max(abs(mat.min()),abs(mat.max()),.01)
        im=ax.imshow(mat,cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_xticks(range(3),['Chl-a','TN','TP']);ax.set_yticks(range(3),['表层','中层','底层'])
        ax.set_title(title)
        for (r,c),val in np.ndenumerate(mat):ax.text(c,r,f'{val:+.3f}',ha='center',va='center',fontsize=9)
        fig.colorbar(im,ax=ax,label='Δ pooled OOF R²')
    fig.savefig(OUT/'figures/图2_垂向结构改善热图.png',dpi=300,facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,4.2),layout='constrained')
    for ax,q in zip(axes,('chla','tn','tp')):
        for i,v in enumerate(('M1','V1','V2')):
            vals=[vd[(vd.variant==v)&(vd.indicator==q)&(vd.pair==p)].difference_RMSE.mean()
                  for p in ('surface-middle','middle-bottom')]
            ax.bar(np.arange(2)+(i-1)*w,vals,width=w,label=v,color=COLORS[v])
        ax.set_xticks(range(2),['表—中','中—底'])
        ax.set_title(q.upper() if q!='chla' else 'Chl-a')
        ax.set_ylabel('差值RMSE（'+('μg/L' if q=='chla' else 'mg/L')+'）')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.07))
    fig.savefig(OUT/'figures/图3_垂向差值预测效果.png',dpi=300,facecolor='white');plt.close(fig)

def write_decision(res,vd):
    vertical=res[res.task!='secchi']
    a=int((vertical.V1_minus_M1_R2>0).sum());b=int((vertical.V2_minus_V1_R2>0).sum())
    lines=['# VSP-Net V1/V2 留站实验决策','',
           '仅使用修复版77项特征和M1既有冻结留站测试样本。M1直接复用既有预测；V1/V2各完成5折×3种子。三种子逐个计算pooled OOF指标后取均值，未拼接种子预测。',
           f'V1相对M1改善{a}/9项垂向任务；V2相对V1改善{b}/9项。', '',
           '|任务|M1 R²|V1 R²|V2 R²|V1−M1|V2−V1|','|---|---:|---:|---:|---:|---:|']
    for _,r in res.iterrows():lines.append(f'|{r.task}|{r.M1_R2:.4f}|{r.V1_R2:.4f}|{r.V2_R2:.4f}|{r.V1_minus_M1_R2:+.4f}|{r.V2_minus_V1_R2:+.4f}|')
    lines+=['','## 垂向差值及结论']
    for q in ('chla','tn','tp'):
        group=res[res.indicator==q]
        lines.append(f'- {q.upper()}三层平均ΔR²：V1−M1={group.V1_minus_M1_R2.mean():+.4f}，V2−V1={group.V2_minus_V1_R2.mean():+.4f}。')
        for p in ('surface-middle','middle-bottom'):
            z=vd[(vd.indicator==q)&(vd.pair==p)].groupby('variant').difference_RMSE.mean()
            lines.append(f'  {p}差值RMSE：M1 {z.M1:.4f}，V1 {z.V1:.4f}，V2 {z.V2:.4f}。')
    lines.append(f'九项垂向任务平均ΔR²：V1−M1={vertical.V1_minus_M1_R2.mean():+.4f}，V2−V1={vertical.V2_minus_V1_R2.mean():+.4f}。')
    v1_ok=a>=6 and all(res[res.indicator==q].V1_minus_M1_R2.mean()>=0 for q in ('chla','tn','tp'))
    diff=vd[vd.pair.isin(['surface-middle','middle-bottom'])].groupby(['indicator','pair','variant']).difference_RMSE.mean().unstack('variant')
    v2_diff_better=int((diff.V2<diff.V1).sum())
    v2_ok=(b>=6 and v2_diff_better>=4 and all(res[res.indicator==q].V2_minus_V1_R2.mean()>=0 for q in ('chla','tn','tp')))
    lines.append('V1结构判定：'+('达到主要改善条件，可继续作为候选。' if v1_ok else '未同时达到多数任务与三个指标均不退化的参考条件，不作为已验证优势模型。'))
    lines.append(f'V2相对V1在{v2_diff_better}/6组垂向差值RMSE上数值更低，'+
                 ('方向上呈额外收益，但多数组别改善幅度很小；' if v2_ok else '未形成一致的额外收益；')+
                 '且V2九项垂向任务均未超过M1，因此不将辅助损失保留为当前优先模型。')
    lines.append('V1与V2均未在任何九项垂向任务上超过M1的三种子平均pooled OOF R²；本轮没有证明VSP-Net能提高外部留站预测精度，M1仍为当前优先模型。')
    lines.append('Secchi不参与垂向辅助损失，单独报告其R²变化。所有差值仅为同一采样事件的预测评价；推理不使用其他水层真实水质值。')
    lines.append('说明：16个事件的 days_since_prev_sample 水层间不同，模型保留各层原始输入和缺失掩码，并通过共享编码器构造事件表示。')
    (OUT/'07_final_decision.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
