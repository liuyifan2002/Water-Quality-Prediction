"""只读新旧冻结预测与输入，分解版本性能变化；不导入或调用模型训练/预测。"""
from pathlib import Path
import hashlib,json,warnings
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115'
OUT=ROOT/'结果/诊断结果/正式77特征新旧性能归因_v1_20261115'
NAMES={'chla_surface':'Chl-a表层','chla_middle':'Chl-a中层','chla_bottom':'Chl-a底层','secchi':'Secchi','tn_surface':'TN表层','tn_middle':'TN中层','tn_bottom':'TN底层','tp_surface':'TP表层','tp_middle':'TP中层','tp_bottom':'TP底层'}
TARGET={'chla':'chla_ug_l','tn':'tn_mg_l','tp':'tp_mg_l','secchi':'secchi_depth_m'}
FILES={};CHECKS=[]
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,**kw):
    p=Path(p);FILES[str(p)]={'文件':str(p),'字节':p.stat().st_size,'SHA256':sha(p)}
    return pd.read_csv(p,**kw)
def js(p):
    p=Path(p);FILES[str(p)]={'文件':str(p),'字节':p.stat().st_size,'SHA256':sha(p)}
    return json.loads(p.read_text(encoding='utf-8-sig'))
def save(d,name):
    pd.DataFrame(d).to_csv(OUT/name,index=False,encoding='utf-8-sig',na_rep='NA')
def ck(name,ok,detail=''):
    CHECKS.append({'检查项':name,'通过':bool(ok),'说明':str(detail)})
def metric(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);sse=float(np.square(y-p).sum());sst=float(np.square(y-y.mean()).sum())
    return {'R2':1-sse/sst if sst>0 else np.nan,'RMSE':np.sqrt(sse/len(y)),'SSE':sse,'SST':sst}
def norm(d):
    return d.assign(model=d.model.replace({'xgb':'xgboost','tabicl':'tabiclv2'}))
def merge_pred(old,new,keys=['obs_id','fold','station_id']):
    assert not old.duplicated(keys).any() and not new.duplicated(keys).any()
    d=old.merge(new,on=keys,suffixes=('_old','_new'),how='outer',validate='one_to_one',indicator=True)
    assert d['_merge'].eq('both').all(),d['_merge'].value_counts()
    assert np.allclose(d.y_true_old,d.y_true_new,rtol=1e-8,atol=1e-9)
    return d.drop(columns='_merge').rename(columns={'y_true_old':'y_true'})
def corr(a,b):
    x=pd.concat([a.rename('a'),b.rename('b')],axis=1).dropna()
    return x.a.corr(x.b,method='spearman') if len(x)>3 and x.a.nunique()>1 and x.b.nunique()>1 else np.nan
def group(c):
    if '_x_' in c:return '交互特征'
    if c.startswith(('ntl','viirs')):return '夜间灯光'
    if c.startswith('tidal'):return '潮汐'
    if c.startswith('rain'):return '降雨'
    if c.startswith(('sst','ssrd')):return 'ERA5及SST'
    return '其他环境特征'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reg=read(RUN/'03_正式十任务重算/运行登记.csv');changes=read(RUN/'02_48项差异核查及验收决定.csv')
    existing=read(RUN/'08_重算结果核验/正式十任务新旧候选指标差异.csv')
    read(RUN/'08_重算结果核验/新候选_pooled与逐折指标.csv')
    priorchecks=read(RUN/'08_重算结果核验/逐项预测与指标复核.csv')
    ck('前阶段独立核验没有失败',priorchecks['结果'].eq('PASS').all(),len(priorchecks))
    policy=js(RUN/'02_冻结候选/特征与时间协议.json');features=policy['feature_order']
    freeze=js(RUN/'02_冻结候选/冻结清单.json')
    raw=read(ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv',low_memory=False).set_index('obs_id',drop=False)
    newraw=read(RUN/'02_冻结候选/基础输入_77特征.csv',low_memory=False).set_index('obs_id',drop=False)
    ck('旧新观测顺序一致',raw.index.equals(newraw.index),len(raw))
    ck('冻结新输入哈希',sha(RUN/'02_冻结候选/基础输入_77特征.csv')==freeze['input_sha256'])
    ck('77项且不含水区编码',len(features)==77 and 'water_zone_code' not in features)
    cfg=js(RUN/'03_正式十任务重算/模型固定配置.json')
    oldcfg=js(ROOT/'结果/扩展实验/候选模型稳健性验证_20260914/portable_package/config.json')
    ck('XGBoost显式参数一致',cfg['model_parameters']==oldcfg['model_parameters']['xgboost'],cfg['model_parameters'])
    sourcecfg=js(RUN/'04_来源消融重算/固定配置.json')
    ck('消融与完整模型显式参数一致',all(sourcecfg['params'][k]==v for k,v in cfg['model_parameters'].items()))
    summaries=[];folds=[];sites=[];decomp=[];obsdiag=[];allpairs={};manifests=[];boot=[];imputation=[]
    oldsec=norm(read(ROOT/'runs/tabiclv2_pilot/09_secchi_predictions.csv'));oldsec=oldsec[oldsec.validation.eq('station')]
    rng=np.random.default_rng(20261115)
    for (task,model),rr in reg.groupby(['task','model'],sort=False):
        target=TARGET[task.split('_')[0]];oldlist=[];newlist=[]
        for _,r in rr.iterrows():
            n=read(r.prediction_file);assert sha(r.prediction_file)==r.prediction_sha256
            if task=='secchi':o=oldsec[oldsec.model.eq(model)&oldsec.fold.eq(r.fold)].copy()
            else:o=read(ROOT/f'runs/{task.split("_")[0]}_family_v2/predictions/{task}_{model}_{r.fold}.csv')
            d=merge_pred(o,n);assert np.isfinite(d[['y_pred_old','y_pred_new']]).all().all()
            assert np.allclose(raw.loc[d.obs_id,target],d.y_true)
            oldlist.append(o);newlist.append(n)
            mp=Path(r.manifest_path);m=read(mp);mh=sha(mp);mf=m[m.fold.eq(r.fold)]
            assert mh==r.manifest_sha256 and set(mf.loc[mf.role.eq('test'),'obs_id'])==set(d.obs_id)
            ck(task+model+r.fold+'可比性',True,'目标/obs_id/站点/折一致，预测有限，77项；完整原测试批')
            manifests.append({'task':task,'model':model,'fold':r.fold,'manifest':str(mp),'manifest_SHA256':mh,'n_test':len(d),'n_train':int(r.n_train),'新有效输入SHA256':r.effective_input_sha256,'新参数SHA256':r.model_config_sha256})
            om=metric(d.y_true,d.y_pred_old);nm=metric(d.y_true,d.y_pred_new)
            folds.append({'task':task,'任务':NAMES[task],'model':model,'fold':r.fold,'n':len(d),'旧R2':om['R2'],'新R2':nm['R2'],'ΔR2':nm['R2']-om['R2'],'旧RMSE':om['RMSE'],'新RMSE':nm['RMSE'],'ΔRMSE':nm['RMSE']-om['RMSE'],'ΔSSE':nm['SSE']-om['SSE'],'目标SST':om['SST']})
        d=merge_pred(pd.concat(oldlist),pd.concat(newlist));allpairs[task,model]=d
        d['旧平方误差']=(d.y_pred_old-d.y_true)**2;d['新平方误差']=(d.y_pred_new-d.y_true)**2;d['ΔSSE']=d['新平方误差']-d['旧平方误差']
        om=metric(d.y_true,d.y_pred_old);nm=metric(d.y_true,d.y_pred_new)
        q=existing[existing.task.eq(task)&existing.model.eq(model)].iloc[0]
        ck(task+model+'已有新旧指标复核',np.allclose([q.R2,q.pooled_R2],[om['R2'],nm['R2']],atol=1e-8))
        sg=[]
        for st,g in d.groupby('station_id'):
            a=metric(g.y_true,g.y_pred_old);b=metric(g.y_true,g.y_pred_new)
            row={'task':task,'任务':NAMES[task],'model':model,'station':st,'fold':g.fold.iloc[0],'n':len(g),'旧RMSE':a['RMSE'],'新RMSE':b['RMSE'],'ΔRMSE':b['RMSE']-a['RMSE'],'旧SSE':a['SSE'],'新SSE':b['SSE'],'ΔSSE':b['SSE']-a['SSE'],'对总体ΔR2贡献':-(b['SSE']-a['SSE'])/om['SST'],'站内目标标准差':g.y_true.std(ddof=0),'目标均值':g.y_true.mean(),'新预测最大':g.y_pred_new.max(),'旧预测最大':g.y_pred_old.max()}
            sites.append(row);sg.append(row)
        s=pd.DataFrame(sg);f=pd.DataFrame([x for x in folds if x['task']==task and x['model']==model]);delta=nm['R2']-om['R2']
        # 分析性阈值，不把微小数值变化称为显著差异。
        category='改善' if delta>.005 else '下降' if delta<-.005 else '基本不变（|ΔR²|≤0.005）'
        top=s.nlargest(5,'ΔSSE');largest=f.sort_values('ΔSSE',ascending=False).iloc[0]
        summaries.append({'task':task,'任务':NAMES[task],'model':model,'n':len(d),'站点数':len(s),'特征数':77,'验证协议':'原五折留站点OOF','旧pooled_R2':om['R2'],'新pooled_R2':nm['R2'],'ΔR2':delta,'旧RMSE':om['RMSE'],'新RMSE':nm['RMSE'],'ΔRMSE':nm['RMSE']-om['RMSE'],'旧fold_R2_mean':f['旧R2'].mean(),'新fold_R2_mean':f['新R2'].mean(),'改善站点比例':s['ΔRMSE'].lt(0).mean(),'改善折数':int(f['ΔR2'].gt(0).sum()),'分类':category,'最差变化折':largest['fold'],'该折对总体ΔR2贡献':-largest['ΔSSE']/om['SST'],'退化最大5站':','.join(top.station),'该5站对总体ΔR2贡献':-top['ΔSSE'].sum()/om['SST'],'旧目标SST':om['SST'],'新目标SST':nm['SST'],'新预测超目标最大值数':int(d.y_pred_new.gt(d.y_true.max()).sum()),'新预测非有限数':int((~np.isfinite(d.y_pred_new)).sum())})
        # 站点簇配对bootstrap只重采样冻结损失，不拟合模型、不代表训练随机性。
        g=d.groupby('station_id').agg(n=('y_true','size'),sy=('y_true','sum'),sy2=('y_true',lambda x:(x*x).sum()),de=('ΔSSE','sum'))
        a=g.to_numpy();ix=rng.integers(0,len(a),size=(1000,len(a)));z=a[ix].sum(axis=1);den=z[:,2]-z[:,1]**2/z[:,0];bs=-z[:,3]/den
        boot.append({'task':task,'model':model,'ΔR2':delta,'配对站点簇95%下限':np.quantile(bs,.025),'配对站点簇95%上限':np.quantile(bs,.975),'重采样次数':1000,'说明':'条件于现有OOF预测；不含重新训练不确定性；探索性且未多重校正'})
        total=om['SST'];within=float(d.groupby('station_id').y_true.apply(lambda v:((v-v.mean())**2).sum()).sum())
        for version,pcol in [('旧','y_pred_old'),('新','y_pred_new')]:
            sse=metric(d.y_true,d[pcol])['SSE'];err=d.y_true-d[pcol];bias=float(d.assign(err=err).groupby('station_id').err.apply(lambda x:len(x)*x.mean()**2).sum())
            for label,mask in [('全部',pd.Series(True,index=d.index)),('目标最高5%（任务内）',d.y_true.ge(d.y_true.quantile(.95))),('其余95%',d.y_true.lt(d.y_true.quantile(.95)))]:
                decomp.append({'task':task,'model':model,'版本':version,'分组':label,'n':int(mask.sum()),'SSE':float(np.square(err[mask]).sum()),'占总SSE比例':float(np.square(err[mask]).sum()/sse),'总体SST':total,'站内SST':within,'站间SST':total-within,'站点均值偏差占SSE比例':bias/sse})
        for _,v in d.nlargest(10,'ΔSSE').iterrows():obsdiag.append({'task':task,'model':model,'obs_id':v.obs_id,'station':v.station_id,'fold':v.fold,'y_true':v.y_true,'旧预测':v.y_pred_old,'新预测':v.y_pred_new,'ΔSSE':v['ΔSSE'],'对总体ΔR2贡献':-v['ΔSSE']/total})
        for _,r in rr.iterrows():
            if model!='xgboost':continue
            m=read(r.manifest_path);trainids=m.loc[m.fold.eq(r.fold)&m.role.eq('train'),'obs_id'];testids=m.loc[m.fold.eq(r.fold)&m.role.eq('test'),'obs_id']
            a=raw.loc[trainids,features];b=read(RUN/f'03_正式十任务重算/inputs/{task}_{r.fold}_训练未填补有效特征.csv').set_index('obs_id')[features]
            bt=read(RUN/f'03_正式十任务重算/inputs/{task}_{r.fold}_测试未填补有效特征.csv').set_index('obs_id')[features]
            for c in changes.feature:
                am=a[c].median();bm=b[c].median();imputation.append({'task':task,'fold':r.fold,'feature':c,'类别':group(c),'旧训练缺失率':a[c].isna().mean(),'新训练缺失率':b[c].isna().mean(),'旧测试缺失率':raw.loc[testids,c].isna().mean(),'新测试缺失率':bt[c].isna().mean(),'旧训练中位数':am,'新训练中位数':bm,'旧全空训练列':a[c].notna().sum()==0,'新全空训练列':b[c].notna().sum()==0,'新测试超训练范围比例':float(((bt[c]<b[c].min())|(bt[c]>b[c].max())).mean()),'说明':'未填补值画像；NA不当作观测'})
    summary=pd.DataFrame(summaries);foldtab=pd.DataFrame(folds);sitetab=pd.DataFrame(sites)
    save(summary,'新旧版本十任务性能变化总表.csv');save(foldtab,'五折性能变化明细.csv');save(sitetab,'逐站点性能变化诊断.csv');save(manifests,'样本划分及哈希可比性核验.csv');save(boot,'性能变化_配对站点簇重采样.csv');save(decomp,'目标方差与误差分解.csv');save(obsdiag,'平方误差变化最大观测.csv');save(imputation,'48特征逐任务折缺失与训练填补.csv')
    print('CORE_DONE',len(summary),len(foldtab),len(sitetab),flush=True)
    # 消融严格以XGBoost同版本完整模型比较，旧完整基线单独从旧消融文件恢复。
    ar=read(RUN/'04_来源消融重算/消融运行登记.csv');taxonomy=read(ROOT/'runs/multisource_ablation_v1/00_feature_source_taxonomy.csv');read(RUN/'04_来源消融重算/四任务六来源_新版本完整对照.csv')
    read(ROOT/'runs/multisource_ablation_v1/multisource_ablation_metrics.csv');ab=[];abfold=[]
    for (task,source),g in ar.groupby(['task','source_group_removed']):
        newp=pd.concat([read(p) for p in g.prediction_file]);full=allpairs[task,'xgboost'];d=merge_pred(full[['obs_id','fold','station_id','y_true','y_pred_new']].rename(columns={'y_pred_new':'y_pred'}),newp)
        assert np.allclose(d.y_true,d.y_true_new)
        a=metric(d.y_true,d.y_pred_old);b=metric(d.y_true,d.y_pred_new)
        source_features=taxonomy.loc[taxonomy.source_group.eq(source),'feature_name'].tolist()
        od=pd.concat([read(ROOT/f'runs/multisource_ablation_v1/predictions/{task}_{source}_fold_{i}.csv') for i in range(1,6)])
        olda=metric(od.y_true,od.full_pred);oldb=metric(od.y_true,od.y_pred)
        match=set(od.obs_id)==set(newp.obs_id) and od.obs_id.is_unique
        if match:match=np.allclose(od.set_index('obs_id').loc[newp.obs_id,'y_true'],newp.y_true)
        for fold,q in d.groupby('fold'):
            am=metric(q.y_true,q.y_pred_old);bm=metric(q.y_true,q.y_pred_new)
            abfold.append({'task':task,'source':source,'fold':fold,'删组ΔR2':bm['R2']-am['R2'],'删组ΔRMSE':bm['RMSE']-am['RMSE']})
        af=pd.DataFrame([v for v in abfold if v['task']==task and v['source']==source]);npositive=int(af['删组ΔR2'].lt(0).sum())
        stationwin=[]
        for _,q in d.groupby('station_id'):stationwin.append(metric(q.y_true,q.y_pred_old)['RMSE']<metric(q.y_true,q.y_pred_new)['RMSE'])
        ab.append({'task':task,'任务':NAMES[task],'模型':'XGBoost','来源组':source,'删除特征数':len(source_features),'改变特征数':len(set(source_features)&set(changes.feature)),'新完整R2':a['R2'],'新删组R2':b['R2'],'新删组ΔR2':b['R2']-a['R2'],'新完整RMSE':a['RMSE'],'新删组RMSE':b['RMSE'],'新删组ΔRMSE':b['RMSE']-a['RMSE'],'来源有益折数':npositive,'完整优于删组站点比例':np.mean(stationwin),'旧完整R2':olda['R2'],'旧删组R2':oldb['R2'],'旧删组ΔR2':oldb['R2']-olda['R2'],'旧新消融样本可比':match,'删组ΔR2版本变化':(b['R2']-a['R2'])-(oldb['R2']-olda['R2']) if match else np.nan,'证据分类':'本任务方向一致的条件贡献' if npositive==5 and b['R2']<a['R2'] else '删组总体改善（可能冗余或噪声）' if b['R2']>a['R2'] else '折间混合的条件贡献','组合效应':'NA：只有单来源删除，无双删组/因子交互实验；不得相加归因','适用边界':'XGBoost四代表任务；不能直接解释TabICLv2 TP中层下降'})
    ab=pd.DataFrame(ab);save(ab,'来源消融性能归因表.csv');save(abfold,'来源消融逐折条件效应.csv')
    print('ABLATION_DONE',len(ab),flush=True)
    # 分布统计仅对未填补原始有效值；所有相关性均为描述性。
    dist=[];stationdist=[];targetcorr=[];pairs=[]
    oc=raw[features].corr(method='spearman');nc=newraw[features].corr(method='spearman')
    for c in changes.feature:
        a=raw[c];b=newraw[c];entry={'feature':c,'类别':group(c),'注册来源':taxonomy.set_index('feature_name').loc[c,'source_group'],'统计粒度':'obs_id三水层行；环境值可能跨层重复','n':len(a),'改变行数':int((~np.isclose(a,b,equal_nan=True,rtol=1e-7,atol=1e-8)).sum())}
        for label,x in [('旧',a),('新',b)]:
            entry.update({label+'有效观测数':x.notna().sum(),label+'缺失率':x.isna().mean(),label+'唯一值数':x.nunique(),label+'均值':x.mean(),label+'标准差':x.std(),label+'最小':x.min(),label+'P01':x.quantile(.01),label+'P50':x.median(),label+'P99':x.quantile(.99),label+'最大':x.max()})
        dif=(nc[c]-oc[c]).drop(c).abs().dropna().sort_values(ascending=False)
        entry['相关性变化最大对应特征']=dif.index[0] if len(dif) else 'NA';entry['该对旧Spearman']=oc.loc[c,dif.index[0]] if len(dif) else np.nan;entry['该对新Spearman']=nc.loc[c,dif.index[0]] if len(dif) else np.nan
        entry['单位与实际拟合限制']='SST异常4项基础参考2014—2020；实际按折气候态见逐折表' if c.startswith('sst_anom') else '原单位保留；未填补NA不算真值'
        for label,x in [('旧',a),('新',b)]:
            st=x.groupby(raw.station).mean();entry[label+'站点均值标准差']=st.std();entry[label+'有真实支持站点数']=x.groupby(raw.station).count().gt(0).sum()
        # 独立sample_id去除三水层复制，避免把重复环境值误认为新增观测。
        sel=~raw.sample_id.duplicated();entry['独立采样数']=int(sel.sum());entry['旧独立采样缺失率']=a[sel].isna().mean();entry['新独立采样缺失率']=b[sel].isna().mean()
        dist.append(entry)
        for st,ids in raw.groupby('station').groups.items():
            aa=a.loc[ids];bb=b.loc[ids];stationdist.append({'feature':c,'station':st,'n':len(aa),'旧缺失率':aa.isna().mean(),'新缺失率':bb.isna().mean(),'旧均值':aa.mean(),'新均值':bb.mean(),'旧唯一值':aa.nunique(),'新唯一值':bb.nunique()})
        for task in NAMES:
            ids=allpairs[task,'xgboost'].obs_id;target=TARGET[task.split('_')[0]];aa=a.loc[ids];bb=b.loc[ids];y=raw.loc[ids,target]
            targetcorr.append({'task':task,'feature':c,'旧有效对数':int((aa.notna()&y.notna()).sum()),'新有效对数':int((bb.notna()&y.notna()).sum()),'旧Spearman':corr(aa,y),'新Spearman':corr(bb,y),'共同非空旧Spearman':corr(aa[bb.notna()],y[bb.notna()]),'共同非空新Spearman':corr(bb[aa.notna()],y[aa.notna()]),'说明':'缺失导致支持集合不同；共同支持另列；相关不代表模型收益/因果'})
        for f in features:
            if f!=c:pairs.append({'feature':c,'另一特征':f,'旧Spearman':oc.loc[c,f],'新Spearman':nc.loc[c,f],'ΔSpearman':nc.loc[c,f]-oc.loc[c,f]})
    save(dist,'48项变化特征分布对比.csv');save(stationdist,'48特征逐站点分布.csv');save(targetcorr,'48特征与十任务目标相关性.csv');save(pairs,'48特征与全部输入相关性变化.csv')
    # 夜光覆盖改善与损失变化的描述性分层，不能识别只改夜光的反事实。
    ntl=[]
    for (task,model),d in allpairs.items():
        before=d.obs_id.map(raw.ntl_avg_rad.isna());after=d.obs_id.map(newraw.ntl_avg_rad.isna())
        categories=np.select([before&~after,~before&~after,before&after],['原缺失→真实覆盖','两版均有真实值','两版均缺失'],default='原有值→新缺失')
        for label,g in d.groupby(categories):
            aa=metric(g.y_true,g.y_pred_old);bb=metric(g.y_true,g.y_pred_new);ntl.append({'task':task,'model':model,'夜光支持组':label,'n':len(g),'旧RMSE':aa['RMSE'],'新RMSE':bb['RMSE'],'ΔSSE':bb['SSE']-aa['SSE'],'总体ΔR2贡献':-(bb['SSE']-aa['SSE'])/metric(d.y_true,d.y_pred_old)['SST'],'因果归因':'不可识别：其他47项同期改变，群组地理分布不同'})
    save(ntl,'夜光真实覆盖分组预测变化.csv')
    print('FEATURES_DONE',len(dist),flush=True)
    # 历史高分独立验证口径，不与正式留站点数据合并作一条变化曲线。
    high=ROOT/'结果/扩展实验/质量感知遥感门控融合_20260905'
    hm=read(high/'质量感知遥感融合_指标.csv');hp=read(high/'质量感知遥感融合_预测.csv');hr=read(RUN/'11_历史高分随机门控重算/运行登记.csv');hrows=[]
    for _,r in hr.iterrows():
        npred=read(r.prediction_file);oldp=hp[hp.target.eq(r.target)&hp.depth.eq(r.depth)]
        assert set(npred.obs_id)==set(oldp.obs_id) and np.allclose(oldp.set_index('obs_id').loc[npred.obs_id,'observed'],npred.observed)
        mets=read(RUN/f'11_历史高分随机门控重算/metrics/{r.task}.csv')
        for _,q in mets[mets.quality_group.eq('all')].iterrows():
            a=hm[hm.target.eq(r.target)&hm.depth.eq(r.depth)&hm.model_variant.eq(q.model_variant)&hm.quality_group.eq('all')].iloc[0]
            hrows.append({'task':r.task,'模型变体':q.model_variant,'验证':'历史随机80/20；与正式留站点不可混比','旧R2':a.r2,'新R2':q.r2,'ΔR2':q.r2-a.r2,'旧RMSE':a.rmse,'新RMSE':q.rmse,'测试数':q.rows,'特征数':a.feature_count,'限制':'旧覆盖参数快照未找到；含water_zone_code/目标历史/双侧影像；不作数据修复单因果比较'})
    hrows=pd.DataFrame(hrows);save(hrows,'历史随机高分_独立口径新旧对照.csv')
    read(RUN/'11_历史高分随机门控重算/运行登记.csv')
    # 参数、运行环境的证据保留；旧运行并非完整二进制环境快照。
    for p in [ROOT/'脚本/04_模型训练/36_run_tn_family_v2_models.py',ROOT/'脚本/04_模型训练/44_run_chla_family_v2_models.py',ROOT/'脚本/04_模型训练/41_run_tp_family_v2_models.py',ROOT/'脚本/04_模型训练/51_recompute_accepted_candidate_v1.py',ROOT/'models/tabiclv2_adapter.py',ROOT/'runs/tabiclv2_pilot/00_environment_check.md']:
        FILES[str(p)]={'文件':str(p),'字节':p.stat().st_size,'SHA256':sha(p)}
    env=js(RUN/'08_重算结果核验/核验环境.json')
    report(summary,foldtab,sitetab,ab,pd.DataFrame(dist),hrows,pd.DataFrame(boot),pd.DataFrame(decomp))
    supplement(summary,foldtab,sitetab,allpairs,raw,newraw,features)
    for p,v in FILES.items():assert sha(p)==v['SHA256'],f'分析期间输入变化：{p}'
    save(CHECKS,'归因分析核验记录.csv');save(FILES.values(),'分析输入文件与SHA256.csv')
    assert all(v['通过'] for v in CHECKS)
    outputfiles=[{'文件':str(p.relative_to(OUT)),'SHA256':sha(p),'字节':p.stat().st_size} for p in OUT.iterdir() if p.is_file() and p.name!='交付SHA256.csv']
    save(outputfiles,'交付SHA256.csv')
    print(json.dumps({'输出':str(OUT),'核验项':len(CHECKS),'失败':sum(not x['通过'] for x in CHECKS),'文件':len(outputfiles)},ensure_ascii=False),flush=True)

def table(d):
    def fmt(v):
        if pd.isna(v):return 'NA'
        return f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(fmt(x) for x in row)+' |' for row in d.itertuples(index=False,name=None)])

def supplement(s,f,st,allpairs,raw,newraw,features):
    """少数站点、折4与历史原始评分口径的补充证据。"""
    detail=[];featurediag=[];hist=[]
    changed=read(RUN/'02_48项差异核查及验收决定.csv').feature.tolist()
    for task in ['tp_middle','tn_middle','tp_surface']:
        d=allpairs[task,'tabiclv2'];total=metric(d.y_true,d.y_pred_old)['SST']
        for maskname,mask in [('第4折',d.fold.eq('fold_4')),('其他四折',~d.fold.eq('fold_4')),('VM12及VT11',d.station_id.isin(['VM12','VT11'])),('其余站点',~d.station_id.isin(['VM12','VT11']))]:
            q=d[mask];a=metric(q.y_true,q.y_pred_old);b=metric(q.y_true,q.y_pred_new)
            detail.append({'task':task,'分组':maskname,'n':len(q),'旧R2_仅本组':a['R2'],'新R2_仅本组':b['R2'],'旧RMSE':a['RMSE'],'新RMSE':b['RMSE'],'对全任务ΔR2贡献':-(b['SSE']-a['SSE'])/total,'解释边界':'只作损失算术分解；不能删去退化站后宣称正式提升'})
        train=read(RUN/f'03_正式十任务重算/inputs/{task}_fold_4_训练未填补有效特征.csv').set_index('obs_id')
        test=read(RUN/f'03_正式十任务重算/inputs/{task}_fold_4_测试未填补有效特征.csv').set_index('obs_id')
        for station in ['VM12','VT11']:
            ids=d.loc[d.station_id.eq(station),'obs_id']
            for c in changed:
                a=raw.loc[ids,c];b=test.loc[ids,c];tr=train[c];featurediag.append({'task':task,'station':station,'fold':'fold_4','feature':c,'旧原始缺失率':a.isna().mean(),'新有效缺失率':b.isna().mean(),'旧原始中位数':a.median(),'新有效中位数':b.median(),'新训练P01':tr.quantile(.01),'新训练P99':tr.quantile(.99),'新站点超训练P01P99比例':((b<tr.quantile(.01))|(b>tr.quantile(.99))).mean(),'说明':'缺失和范围诊断非独立致因证据，SST采用本折有效输入'})
    save(detail,'第4折与关键站点误差贡献.csv');save(featurediag,'关键退化站48项输入支持.csv')
    for file in ['09_secchi_predictions.csv','14_tn_bottom_predictions.csv','17_tp_middle_predictions.csv']:
        p=ROOT/'runs/tabiclv2_pilot'/file;d=norm(read(p))
        for (validation,model),g in d.groupby(['validation','model']):
            a=metric(g.y_true,g.y_pred);hist.append({'源文件':str(p),'task':g.target.iloc[0],'model':model,'验证':validation,'n':len(g),'折数':g.fold.nunique(),'pooled_R2':a['R2'],'fold_R2_mean':np.mean([metric(q.y_true,q.y_pred)['R2'] for _,q in g.groupby('fold')]),'特征':'X_strict_77','与新正式比较':'仅Secchi station为相同注册任务；其他需独立obs_id/manifest核验，禁止直接替代Family V2'})
    save(hist,'历史Pilot不同验证评分_独立审计.csv')
    p=OUT/'预测精度下降原因综合报告.md'
    text=p.read_text(encoding='utf-8')
    appendix=['','## 补充：共同退化折与主要站点的定量定位','',table(pd.DataFrame(detail)), '',
      'TP中层VM12、VT11合计对总体ΔR²贡献约−0.0980，其他站合计抵消约+0.0046；第4折贡献约−0.1009，其他四折合计约+0.0075。TN中层、TP表层同样由第4折抵消其他折增益。因此“全域性数据补充导致所有区域恶化”不符合这些冻结预测；更准确的是特定留出站与折的上下文敏感性，具体来源仍待受控验证。',
      '剔除这些站点仅用于误差组成诊断，未生成或推荐删站后的正式性能。补充表列出VM12/VT11在第4折的48项输入支持和训练P01/P99范围；跨范围只是线索，不可替代反事实。','',
      '### 现有证据能确定与不能确定的部分','',
      '- 已确定：同一测试真值/SST下的SSE变化；不是测试集缩放、目标方差漂移或指标改名造成。',
      '- 已确定：XGBoost气象/海洋物理/空间来源在Chl-a表层五折删组均变差；Secchi气象与空间来源亦为五折一致。TP中层雨潮来源五折删组均改善，但该证据仅适用于该XGBoost配置。',
      '- 尚未确定：夜光补齐或其他单源对TabICLv2退化的独立份额；现有单组XGBoost删除不能回答。',
      '- 尚未确定：旧新软件运行默认值和GPU重复幅度是否解释一部分变化；必须先做同有效输入重复。',
      '- 历史高分不仅有随机划分，也可能是同一留站点实验的某一较好折，或fold_R2_mean；不能把这些高分平均推广到pooled全任务。','']
    p.write_text(text+'\n'.join(appendix),encoding='utf-8')
    hpath=OUT/'历史高分与正式结果可比性审计.md'
    hpath.write_text(hpath.read_text(encoding='utf-8')+'\n\n## 历史Pilot三种验证评分原件补查\n\n'+table(pd.DataFrame(hist))+'\n\n这些分数按验证口径分别统计，未绘成同一性能变化曲线。旧TN底层/TP中层Pilot的样本范围可能不同于后续Family V2，不能用其高分直接声称77特征数据修复下降。另注意TP中层正式旧第2折R²约0.642，新第2折约0.653，但pooled仍为0.378；单折高分不是全任务pooled。\n',encoding='utf-8')
def report(s,f,st,ab,dist,h,b,dec):
    t=s[s.model.eq('tabiclv2')].copy();x=s[s.model.eq('xgboost')];worst=t.sort_values('ΔR2').iloc[0]
    lines=['# 正式77特征新旧预测性能变化归因分析','',
      '## 结论先行','',
      f'没有发生十任务整体精度下降：XGBoost 10/10项pooled R²提高；TabICLv2有{int(t["ΔR2"].gt(0).sum())}项提高、{int(t["ΔR2"].lt(0).sum())}项数值下降。最明显下降为{worst["任务"]}，{worst["旧pooled_R2"]:.5f}→{worst["新pooled_R2"]:.5f}，ΔR²={worst["ΔR2"]:+.5f}。',
      '“仍低于0.5”与“修复导致从0.7跌到0.5”是两个问题。旧正式留站点TabICLv2本就有多项低于0.5；历史0.6/0.7分数必须按具体任务、随机/留站点、pooled/折均值辨别。数据补齐不增加独立目标观测，也不保证跨站点泛化提高。','',
      '## 1. 可比性和指标定义','',
      '逐个核对100个新旧任务×模型×折的obs_id、目标值、测试站点、fold、manifest SHA256及77列。Family V2九分层任务保持84共同站，Secchi保持94站。测试真值和目标SST完全不变；没有按分数删站、删样本或重划分。旧值使用冻结预测重新统计，与既有08汇总对照。',
      'R²=1−SSE/SST；RMSE=√(SSE/n)。pooled_R2在全部OOF观测上计算；fold_R2_mean是五个独立折R²算术平均，两者不能互换。逐站点版本改善率=RMSE新<RMSE旧的站点数/全部站点数，区别于新版本TabICLv2对XGBoost胜率。',
      '全部CSV中的Δ均为新−旧；来源消融Δ为删组−同版本完整模型。以|ΔR²|≤0.005标为“基本不变”仅是描述性筛选，不是统计显著性标准。另做1000次配对站点簇重采样，条件于现有预测；区间不包含重训随机性，也未多重检验校正。','',
      '## 2. 十任务版本变化','',table(s[['任务','model','旧pooled_R2','新pooled_R2','ΔR2','旧RMSE','新RMSE','改善站点比例','改善折数','分类']]),'',
      f'TabICLv2新版本低于0.5共{int(t["新pooled_R2"].lt(.5).sum())}项，旧版本共{int(t["旧pooled_R2"].lt(.5).sum())}项。不是所有低分由本次修复新造成。XGBoost最大的增益为{x.sort_values("ΔR2",ascending=False).iloc[0]["任务"]}。','',
      '## 3. 下降由哪些折和站点驱动','']
    for _,r in t.sort_values('ΔR2').iterrows():
        if r['ΔR2']>=0:continue
        q=f[f.task.eq(r.task)&f.model.eq('tabiclv2')];p=st[st.task.eq(r.task)&st.model.eq('tabiclv2')].sort_values('ΔSSE',ascending=False);ci=b[b.task.eq(r.task)&b.model.eq('tabiclv2')].iloc[0]
        lines += [f'### {r["任务"]}', '',f'ΔR²={r["ΔR2"]:+.5f}，RMSE {r["旧RMSE"]:.6f}→{r["新RMSE"]:.6f}；{int(r["改善折数"])}/5折改善，{r["改善站点比例"]:.1%}站改善。最差变化折{r["最差变化折"]}对总体ΔR²贡献{r["该折对总体ΔR2贡献"]:+.5f}。退化最大五站{r["退化最大5站"]}合计贡献{r["该5站对总体ΔR2贡献"]:+.5f}，是总净变化的组成项，可被其他站抵消，不能当成占比。配对站点簇95%区间[{ci["配对站点簇95%下限"]:+.5f}, {ci["配对站点簇95%上限"]:+.5f}]。', '',table(q[['fold','旧R2','新R2','ΔR2','旧RMSE','新RMSE','ΔSSE']]),'',table(p.head(5)[['station','n','旧RMSE','新RMSE','ΔRMSE','对总体ΔR2贡献']]),'']
    lines += ['SSE分解为严格可加的算术贡献，不是环境因果贡献。即使改善站点占多数，少数高误差或样本多的站点仍可推高pooled误差。站点均值差异构成总体SST，pooled R²也可能高于以站内中心化为基准的指标；不能把它等同于每个站的时序预测能力。见目标方差与误差分解、最大误差观测表。','',
      '## 4. 来源消融证据','',
      '120个消融任务为四代表任务×六来源×五折，模型全部为XGBoost。消融和完整模型使用同版本、同输入中位数、同77列母集、同测试观测和显式参数。负删组ΔR²表示该来源在这些条件下有预测收益；正值说明删除该组后模型更好，可能是冗余、噪声、固定参数适配或特征交互，不能推导为环境物理负作用。','',
      table(ab[['任务','来源组','新删组ΔR2','新删组ΔRMSE','来源有益折数','完整优于删组站点比例','证据分类']]),'',
      '空间静态在Chl-a表层和Secchi上有较大条件贡献；不能因此断言其对所有任务稳定有益。人类活动代理（六项夜光相关字段）在Chl-a表层、Secchi、TP中层删组后pooled R²提高，在TN表层则降低。气象在Chl-a表层、Secchi、TP中层删组后下降，在TN表层则提高。水文降雨潮汐在TN表层、TP中层删组后提高，在Chl-a表层、Secchi下降。具体折间一致性见表，不能凭pooled方向声称五折稳定。',
      '六组是注册taxonomy的主归属，跨来源交互项只属于一个注册组，不能任意重新归类。没有双来源删除或完整因子实验，不能识别组合交互、不能把六个删组效应加总成全部贡献。旧新消融基线逐项独立恢复；样本范围不等价的比较标NA。',
      '**关键限制：这些是XGBoost实验，无法直接证明TabICLv2的TP中层下降由夜光、潮汐或气象单独造成。未计算TabICLv2同组替换或删除反事实；不能把XGBoost删组收益套到TabICLv2。**','',
      '## 5. 48项变化特征','',table(dist[['feature','类别','旧缺失率','新缺失率','旧唯一值数','新唯一值数','旧P50','新P50']]),'',
      '分布、唯一值、极值和相关性全部基于未填补有效值；逐任务折表另列训练中位数、全空列及测试超训练范围。主表三层环境值可能重复，另列sample_id去重后的缺失率；不把三层复制视为三份独立环境测量。SST异常四项基础表2014—2020参考仅供分布审查，实际训练使用本折训练期气候态，实际缺失和中位数取保存的有效折输入。',
      '夜光从原8站低覆盖补至94站同源覆盖，同时保留无晴空质量缺失。夜光真实覆盖组预测变化表可以显示损失变化发生在哪类样本，但其他47项同期变化、覆盖组具有空间差异，故不能据此估计“补夜光”的净因果增益。新旧共同非空支持的目标相关性与各自有效支持相关性分开输出，不把填补常数当观测。','',
      '## 6. 模型、评估和数据变化的边界','',
      '源码和配置证明XGBoost显式参数、seed=42、log1p/expm1及非负截断一致；TabICLv2沿用同一适配器、seed=42、float32、官方预训练权重和完整原测试集批调用，并非P1单查询。训练上下文观测ID不变，但48项输入与训练中位数改变，所以上下文数值已改变；该变化不能称为模型算法改变。',
      '新运行保存了按折未填补输入、训练中位数及有效NPZ，前阶段2661项核验已检验训练目标、输入哈希和保存模型恢复；本轮再次核对预测对齐、有限值和指标。旧训练二进制环境/输入NPZ快照不完整，源码同参不等于完全相同运行时；旧新软件默认行为、GPU数值与种子重复幅度无法彻底排除。旧协议记录sklearn 1.6.1，新核验环境1.9.1，不得将所有差异无条件全归数据。',
      '目前可确认的是“同任务同测试集上的版本流水线差异”，而非48项各自独立效应。XGBoost全任务提高、TabICLv2部分退化，更支持模型对数据表征变化的响应不同；不支持“真实数据越完整越差”的普遍论断。潮汐支持、气候态重构、ERA5/降雨补齐、夜光补齐同时发生，必须做最小来源替换才能定量区分。',
      '同期整月输入保留既定定义，月末可能晚于采样日，因此仅解释历史同期/空间留站点泛化；不是提前预报。补充的Landsat、CMEMS海色和掩膜未新增进正式77项，不能归因为它们使本轮十任务变差。','',
      '## 7. 历史高分核对','',
      '见独立历史审计和历史随机对照CSV。不得把随机80/20高分与正式留站点pooled R²画在同一版本变化序列中。高分不是虚假结果，但对应更容易共享站点背景的任务、额外water_zone_code/目标历史、双侧影像或不同目标范围，回答的问题不同。也不能反向声称0.7全部来自泄漏；需要查明具体数据行和协议。','',
      '## 8. 最优先投入','',
      '先做TP中层TabICLv2最差变化折上的同数据原样重复与逐来源旧/新替换；上下文顺序、测试批、seed、权重和环境全部锁定，确认随机波动底噪。第二优先TN中层/TP表层；仅在筛查显示特定来源效应后补全该来源五折。不为提分删除真值，不用观察到的最高测试分直接选最终方案。详见下一步最小补充实验清单。','',
      '本轮无模型训练、推理或解释计算；所有输出为已有预测的确定性统计及固定种子统计重采样。原始预测、数据、划分、论文均未修改。']
    (OUT/'预测精度下降原因综合报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit=['# 历史高分与正式结果可比性审计','',
      '## 同一正式口径','',
      '旧Family V2九项任务及旧Secchi station预测与本次十任务逐obs_id完全配对；同一目标、站点折和测试真值，77项特征名顺序保留。旧指标从预测复算，禁止用历史随机、年份、不同层位或不同样本任务替代。',
      '可比性成立是评估集合相同，不意味着所有训练环境因素受控：输入值、缺失、中位数和SST气候态有意改变；旧独立环境/模型参数覆盖快照不完整。','',
      '## 历史随机高分（单列，不与留站点混合）','',
      table(h[['task','模型变体','旧R2','新R2','ΔR2','测试数','特征数']]),'',
      '原随机门控实验2014—2024、12目标×层组合，固定80/20随机测试ID；样本间可能共享站点。基础82项包含water_zone_code与目标历史，另有遥感/质量扩展，双侧匹配可能使用采样后影像；不是正式77列单模型空间外推。Secchi历史三水层亦不等于正式单一Secchi任务。新高分重放恢复原训练/测试ID但旧运行覆盖参数快照未找到，采用原源码可追溯默认值，不能将旧新差值全归数据。',
      '0.6/0.7必须对应具体task/model_variant/quality_group。主对照只取quality_group=all；晴空或高质量子集分数不代表全部样本。全部原180行及本次12任务各分组文件均读取并留哈希，不将子集最佳值包装成总体性能。','',
      '## pooled与折均值','',
      table(s[['任务','model','旧pooled_R2','旧fold_R2_mean','新pooled_R2','新fold_R2_mean']]),'',
      'pooled包含站点间和站内目标方差；五折均值是不同目标分布下的五个比值平均。不能把均值、pooled和随机单次得分视为同一指标。旧正式最优任务本来就在0.6左右，不能把其他历史实验0.7作为所有任务旧基线。']
    (OUT/'历史高分与正式结果可比性审计.md').write_text('\n'.join(audit)+'\n',encoding='utf-8')
    experiments=['# 下一步最小补充实验清单','',
      '本清单仅提出后续实验，本轮未执行。全部在新扩展目录另存，不改正式77项定义、原manifest或真值。探索筛查不能直接当最终测试选模。','',
      '1. **TP中层TabICLv2同输入重复底噪（最高优先）**：在最差ΔSSE折，固定新NPZ、训练ID顺序、完整测试批、环境和权重，seed42原样重复2次，再按预登记种子重复2次。4次推理；只重建预训练上下文，不微调预训练权重。记录数值差与现有ΔR²，先判断下降是否超过运行波动。','',
      '2. **同折来源旧/新替换**：保留77列，锁定同一处理器与环境，在最差折比较全旧、全新、仅夜光新、仅ERA5/SST新、仅雨潮新五套输入；全新现值可作基准，另需4次拟合上下文与预测。交互项按注册公式从所选父特征确定性重建，所有跨来源交互预先写明版本，不拼接物理不一致输入。旧错误值仅作为诊断反事实，不重新推荐为正式数据。补数据与SST训练期气候态修复应分开登记。','',
      '3. **TN中层与TP表层定向复核**：仅对其各自最差变化折重做上述筛查；TP底层|ΔR²|约0.002暂低优先。若下降只是少数站点驱动，同时报告站点簇不确定性、目标尾部损失与训练输入覆盖，不删站提分。','',
      '4. **只扩展被筛中的来源到五折**：一项来源替换×一个任务最多5折，先TP中层，再视证据决定是否TN中层。用同站点ID配对SSE，不比较独立抽样的两组分数。不要一开始重跑全部十任务×两模型。','',
      '5. **组合效应仅在单组证据明确后**：对至多两个来源预注册2×2旧/新因子设计（00/10/01/11），先一折再五折；用交互对比Δ11−Δ10−Δ01+Δ00，不能从现有单组删除实验相加推算。','',
      '6. **版本冻结和环境控制**：实验前保存旧新参数全展开、包版本、预训练权重哈希、上下文顺序/填补统计、有效输入与预测哈希；若无法恢复旧环境，报告为新环境内受控反事实，不冒充旧运行精确复现。严格提前预测属独立问题，不混入本次留站点归因。','',
      '计算资源首选TP中层TabICLv2，不优先重算SHAP/ALE：解释图不能替代预测反事实和同输入重复，现有XGBoost消融无法识别TabICLv2退化原因。']
    (OUT/'下一步最小补充实验清单.md').write_text('\n'.join(experiments)+'\n',encoding='utf-8')

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    main()
