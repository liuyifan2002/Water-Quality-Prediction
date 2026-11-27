"""核验正式特征之外的历史候选数据、排除证据及已有获取方式。"""
from pathlib import Path
import ast,json,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115';E=OUT/'证据'
OLD=ROOT/'codex/worktrees/9ff7/水质预测'
D=ROOT/'数据/外部数据/试验项目接管_20260521/data_processed_current'

def save(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def main():
    E.mkdir(parents=True,exist_ok=True)
    water=pd.read_csv(ROOT/'数据/处理后数据/正式主表/water_quality_master_2014_2024.csv',usecols=['obs_id','sample_id','station','date','depth'])
    samples=water.drop_duplicates('sample_id');stations=set(water.station);rows=[];detail=[]
    candidates=sorted(set(list(D.glob('*features*.csv'))+list(D.glob('*station_validation.csv'))+list((ROOT/'数据/外部数据/试验项目接管_20260521/data_model_input_new_sources').glob('*.csv'))+list((ROOT/'数据/处理后数据/衍生版本').glob('*.csv'))))
    for p in candidates:
        d=pd.read_csv(p,low_memory=False);station=next((c for c in ['station','Station','station_id'] if c in d),None)
        r={'数据文件':str(p),'记录数':len(d),'字段数':len(d.columns),'实际站点':d[station].nunique() if station else '区域日期代理','期待站点':94,'缺少站点':';'.join(sorted(stations-set(d[station].dropna()))) if station else '不适用：日期级代理须查日期覆盖','正式77以外也核查':True}
        if 'obs_id' in d:base=water;keys=['obs_id']
        elif 'sample_id' in d:base=samples;keys=['sample_id']
        elif station and 'date' in d:base=samples;keys=['station','date'];d=d.rename(columns={station:'station'});station='station'
        elif station and 'year' in d and 'month' in d:base=samples.assign(year=pd.to_datetime(samples.date).dt.year,month=pd.to_datetime(samples.date).dt.month);keys=['station','year','month'];d=d.rename(columns={station:'station'});station='station'
        elif station:base=water[['station']].drop_duplicates();keys=['station'];d=d.rename(columns={station:'station'});station='station'
        elif 'date' in d:base=samples[['date']].drop_duplicates();keys=['date']
        else:base=None;keys=[]
        if keys:
            r['连接键']='+'.join(keys);r['键重复']=int(d.duplicated(keys).sum())
            if 'date' in keys:
                base=base.copy();d=d.copy();base['date']=pd.to_datetime(base.date);d['date']=pd.to_datetime(d.date,errors='coerce')
            hit=base[keys].merge(d[keys].drop_duplicates(),on=keys,how='left',indicator=True,validate='many_to_one')
            r['实际需要键数']=len(base);r['未连接键数']=int(hit._merge.eq('left_only').sum())
        numeric=[c for c in d.select_dtypes(include='number') if c not in ['year','month','lon','lat','request_id']]
        r['全部缺失字段']=';'.join(c for c in d if d[c].isna().all())
        r['数值信号字段数']=len(numeric);r['数值字段缺失率中位数']=float(d[numeric].isna().mean().median()) if numeric else np.nan
        name=p.name
        if 'long_history' in name:r['本轮判断']='需按实际字段辨识海水/泳滩数据；不能仅凭历史年份或文件名视为同源海水历史样本'
        elif 'lagged_cross_water' in name:r['本轮判断']='历史同期水质滞后协变量；必须按站点水层、真实可用时刻核验，不能自动进入正式新站77特征'
        elif 'sentinel3' in name or 'ocean_color_kd' in name:r['本轮判断']='核查是否真实数值而非仅支持清单/全NA占位；按画像与原栅格确认，不能凭表存在宣称有有效输入'
        elif any(s in name for s in ['copernicus','strong_mechanism']):r['本轮判断']='数据链需重审：原CMEMS日窗实际为采样记录窗，末层代理及原下载范围问题已专项核实；不得直接承接旧无效果结论'
        elif 'globathy' in name:r['本轮判断']='空表/场景不适用：不能作为海洋真实水深或已完成实验'
        elif 'nasa' in name:r['本轮判断']='核心CHL/KD有94站132月原件，逐值/滞后另核；存在近岸回退和未支持字段，不保证有效提升'
        elif 'depth' in name:r['本轮判断']='94站近似总水深有原PDF；人为层深比例非实测；原实验未提升不等于源数据缺失'
        elif 'coastal' in name or 'mouth' in name:r['本轮判断']='真实静态代理，有原矢量/提取脚本；精度及河口系统定义独立限制，不能当流量'
        elif 'patch' in name:r['本轮判断']='有真实样本键和影像ID，缺失需分开传感器上线前/窗口无影像/QA无有效像元；不把插补当实测'
        elif 'tide' in name:r['本轮判断']='日潮位代理可追溯，缺精确采样时刻时不能称实测潮相'
        elif 'model_' in name:r['本轮判断']='历史派生建模表；检查全站范围、目标历史项可用性及环境输入缺口传播，不能混入正式77协议'
        else:r['本轮判断']='画像与连接已核，语义以来源脚本和本轮专项证据为准；未证明无效也未证明增益'
        rows.append(r)
        if 'patch' in name and station:
            for sensor in ['s2','landsat']:
                col=f'{sensor}_image_date'
                if col in d and 'date' in d:
                    image=pd.to_datetime(d[col],errors='coerce');dt=pd.to_datetime(d.date,errors='coerce');delta=(image-dt).dt.total_seconds()/86400
                    for yr,g in d.groupby(dt.dt.year):
                        sub=delta.loc[g.index];image_valid=image.loc[g.index].notna()
                        detail.append({'文件':str(p),'传感器':sensor,'年份':yr,'采样数':len(g),'有效影像日期':int(image_valid.sum()),'影像日期缺失率':float((~image_valid).mean()),'影像在采样之后':int(sub.gt(0).sum()),'绝对日期偏差最大天':float(sub.abs().max()) if sub.notna().any() else np.nan,'未来影像边界':'双侧窗口支持历史同期估计，不等于实时提前预测'})
    save('02_历史候选及排除数据逐表复核.csv',rows);pd.DataFrame(detail).to_csv(E/'22_高分辨率影像逐年及双侧匹配.csv',index=False,encoding='utf-8-sig')
    reports=[];decisions=[]
    for folder in [OLD/'experiments/reports/current',ROOT/'文档/主线接管_20260521/英文原始报告_归档/源项目_reports_current',ROOT/'文档/主线接管_20260521']:
        for p in sorted(folder.glob('*.md')):
            text=p.read_text(encoding='utf-8-sig',errors='replace')
            selected=any(t in p.name for t in ['no_promote','failed','失败','rejected','缺口','no_repeat','do_not_repeat','validation','source','数据源','增益'])
            if not selected:continue
            reports.append({'说明文件':str(p),'标题':text.splitlines()[0] if text.splitlines() else '空说明','审查方式':'保留旧结论作历史记录，与当前真实数据质量分开；乱码以CSV证据复核','文字损坏风险':'是' if text.count('?')>max(30,len(text)*.03) else '未明显发现'})
            for i,line in enumerate(text.splitlines(),1):
                if any(s.lower() in line.lower() for s in ['未 promote','未promote','没有进入','delta_r2_nonpositive','低收益','未突破','不再','reject','needs_user_api','未完成','缺少','not_promote']):
                    decisions.append({'原说明':str(p),'行号':i,'历史原文':line.strip(),'本轮处理':'保留冻结效果记录；先查来源/覆盖/连接/定义，再决定是否需重审，不直接将历史未增益当无效数据'})
    save('03_历史排除决定原文登记.csv',decisions);save('04_历史数据说明与有效性复核范围.csv',reports)
    catalog=[]
    scriptroot=ROOT/'脚本/历史版本/试验项目接管_20260521'
    for p in sorted(set(list((ROOT/'脚本/01_数据准备').glob('*.py'))+list((ROOT/'脚本/02_特征构建').glob('*.py'))+list((ROOT/'脚本/03_GEE遥感导出').glob('*.js'))+list(scriptroot.rglob('*.py'))+list((OLD/'scripts').rglob('*.py')))):
        text=p.read_text(encoding='utf-8-sig',errors='replace')
        if not any(x in p.name.lower() for x in ['download','fetch','source','build','prepare','extract','validate','refine','merge']):continue
        # 只记录公开数据入口，丢弃查询串与认证信息，不读取.env内容。
        urls=sorted(set(re.findall(r'https?://[A-Za-z0-9./_\-]+',text)))
        ids=sorted(set(re.findall(r'[\"\']((?:COPERNICUS|LANDSAT|NOAA|NASA|ECMWF)/[^\"\']+|cmems_[^\"\']+)[\"\']',text)))
        catalog.append({'获取或构建脚本':str(p),'公开入口':';'.join(urls),'产品标识':';'.join(ids),'调用状态':'只读审查，未直接执行旧脚本（旧脚本可能覆盖源表或联动训练）','追溯范围':'用户此前已有获取方式；当前补取脚本独立另存'})
    save('05_全部既有数据获取与构建方式.csv',catalog)
    print('历史候选表',len(rows),'排除原文',len(decisions),'说明',len(reports),'获取脚本',len(catalog),flush=True)

if __name__=='__main__':main()
