"""图5-4：冻结77特征填补前输入非缺失率，来源、站点与年度。"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
OUT=Path(__file__).resolve().parents[1]/'正式论文图片'
OUT.mkdir(exist_ok=True)
TAX=ROOT/'runs/multisource_ablation_v1/00_feature_source_taxonomy.csv'
assert hashlib.sha256(TAX.read_bytes()).hexdigest()=='b8141c13a0d376986000de8fc0f5d9bac9f33c9545820cf8b4bc1eee243de9bc'
GROUPS=[('METEOROLOGY','气象 MET'),('RAINFALL_HYDRO_TIDE','降雨/水文/潮汐 RHT'),('OCEAN_PHYSICS','海洋物理 OCN'),('HUMAN_ACTIVITY','人类活动代理 HUM'),('TEMPORAL_PROCESS','时间过程 TMP'),('SPATIAL_STATIC','空间静态 SPA')]
COLORS=['#4477AA','#EE9944','#228833','#AA3377','#777777','#44AA99']

def main():
    base=pd.read_csv(SRC/'02_冻结候选/基础输入_77特征.csv')
    meta=pd.read_csv(TAX)
    assert len(meta)==77 and set(meta.feature_name).issubset(base.columns)
    base['year']=pd.to_datetime(base.date).dt.year
    fonts={x.name for x in font_manager.fontManager.ttflist}
    cn=next((x for x in ['Microsoft YaHei','Noto Sans CJK SC','SimHei'] if x in fonts),'DejaVu Sans')
    mpl.rcParams.update({'font.family':cn,'font.size':8.5,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(6.1,6.3),facecolor='white',layout='constrained')
    gs=fig.add_gridspec(3,1,height_ratios=[1.15,1.05,1.15],hspace=.14)
    ax0,ax1,ax2=[fig.add_subplot(gs[i]) for i in range(3)]
    labels=[x[1] for x in GROUPS]; yr=sorted(base.year.unique())
    annual=[]; station=[]
    for g,(code,_) in enumerate(GROUPS):
        fs=meta.loc[meta.source_group==code,'feature_name'].tolist()
        vals=np.array([base[f].notna().mean()*100 for f in fs]);
        ax0.scatter(np.full(len(vals),g)+np.linspace(-.18,.18,len(vals)),vals,s=12,color=COLORS[g],alpha=.67,linewidth=0)
        ax0.plot([g-.22,g+.22],[np.median(vals)]*2,color='black',lw=1.2)
        annual.append([base.loc[base.year==y,fs].notna().to_numpy().mean()*100 for y in yr])
        station.append([sub[fs].notna().to_numpy().mean()*100 for _,sub in base.groupby('station',sort=True)])
    ax0.set_xticks(range(6),['MET','RHT','OCN','HUM','TMP','SPA']);ax0.set_ylim(94,101);ax0.set_ylabel('非缺失率（%）');ax0.set_title('A  各来源特征的输入覆盖（点=单项特征；黑线=中位数）',loc='left',fontweight='bold')
    ax0.grid(axis='y',alpha=.2)
    im=ax1.imshow(np.array(annual),vmin=94,vmax=100,cmap='Blues',aspect='auto')
    ax1.set_yticks(range(6),labels);ax1.set_xticks(range(len(yr)),yr,rotation=0)
    ax1.set_title('B  各来源在不同年份的输入覆盖',loc='left',fontweight='bold')
    cb=fig.colorbar(im,ax=ax1,pad=.01,shrink=.85);cb.set_label('非缺失率（%）')
    for j,row in enumerate(annual):
        for k,val in enumerate(row): ax1.text(k,j,f'{val:.1f}',ha='center',va='center',fontsize=6.2,color='#202020' if val<97 else 'white')
    for g,vals in enumerate(station):
        vals=np.array(vals)
        ax2.boxplot(vals,positions=[g],widths=.43,showfliers=False,patch_artist=True,boxprops={'facecolor':COLORS[g],'alpha':.45,'linewidth':.6},medianprops={'color':'#222222','linewidth':1.3},whiskerprops={'linewidth':.7},capprops={'linewidth':.7})
    ax2.set_xticks(range(6),['MET','RHT','OCN','HUM','TMP','SPA']);ax2.set_xlim(-.5,5.5);ax2.set_ylim(88,101);ax2.set_ylabel('非缺失率（%）');ax2.set_title('C  94站的来源输入覆盖分布',loc='left',fontweight='bold');ax2.grid(axis='y',alpha=.2)
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'图5-4_修复版77特征真实观测覆盖.{ext}',dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('figure done',len(base),base.station.nunique(),len(yr))

if __name__=='__main__':main()
