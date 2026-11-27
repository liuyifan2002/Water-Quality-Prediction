"""只读审查回传SHAP/ALE：哈希、上下文、可加性、固定探针和SQLite算术，不调用模型。"""
from pathlib import Path,PurePosixPath
import ast,hashlib,json,zipfile,io,sqlite3,warnings,difflib
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[2]
LOCAL=ROOT/'交付/修复77特征解释计算迁移_20261116_v3/TabICLv2_SHAP_ALE_PORTABLE'
ZIP=LOCAL/'02_data/TabICLv2_WQ77_EXPLANATION_RESULTS_20261120.zip'
OUT=ROOT/'结果/诊断结果/修复77特征SHAP回传核验_20261120_v2'
CHECKS=[];SOURCES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ha(value):
    a=np.ascontiguousarray(np.asarray(value));return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def hi(v):return hashlib.sha256('\n'.join(map(str,v)).encode()).hexdigest()
def ck(name,ok,detail='',kind='HARD'):
    CHECKS.append({'检查项':name,'结果':'PASS' if bool(ok) else 'FAIL','类别':kind,'依据':str(detail)})
def source(p):
    p=Path(p);SOURCES[str(p)]={'文件':str(p),'SHA256':sha(p),'字节':p.stat().st_size};return p
def save(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig',na_rep='NA')
def md(d):
    def fmt(x):return 'NA' if pd.isna(x) else f'{x:.7g}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])
def main():
    if OUT.exists():raise FileExistsError('不覆盖既有核验目录')
    OUT.mkdir(parents=True)
    source(ZIP);expected=ZIP.with_suffix('.zip.sha256').read_text(encoding='utf-8-sig').split()[0]
    assert sha(ZIP)==expected,'ZIP SHA256不一致，停止接受回传'
    ck('ZIP SHA256与旁路一致',True,expected)
    with zipfile.ZipFile(ZIP) as z:
        names=z.namelist();assert len(names)==len(set(names));assert z.testzip() is None
        ck('ZIP CRC及唯一成员',True,len(names))
        for n in names:
            p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts and '\\' not in n
            assert ((z.getinfo(n).external_attr>>16)&0o170000)!=0o120000,'不接受符号链接'
        z.extractall(OUT/'原始回传解包')
    R=OUT/'原始回传解包'
    manifest=pd.read_csv(R/'PACKAGE_MANIFEST.csv');ck('回传清单覆盖全部数据代码（附说明另计）',set(manifest.path)==set(names)-{'PACKAGE_MANIFEST.csv','RESULTS_README.txt'})
    CHECKS.append({'检查项':'附说明未列内部清单','结果':'WARN','类别':'PACKAGING','依据':'RESULTS_README.txt未列PACKAGE_MANIFEST，但由外层ZIP SHA256覆盖；全部科学数据和代码均已列入并逐项验证'})
    for row in manifest.itertuples():ck('回传内部哈希 '+row.path,(R/row.path).stat().st_size==row.bytes and sha(R/row.path)==row.sha256)
    cfg=json.loads((R/'04_explanation_config/portable_config.json').read_text(encoding='utf-8-sig'));lc=json.loads(source(LOCAL/'04_explanation_config/portable_config.json').read_text(encoding='utf-8-sig'))
    cs=sha(R/'04_explanation_config/portable_config.json');fs=cfg['features'];tasks=cfg['tasks']
    ck('正式版本',cfg['dataset_version']=='WQ77_REPAIRED_20261115_V1')
    ck('配置与原迁移配置完全相同',cfg==lc and cs==sha(LOCAL/'04_explanation_config/portable_config.json'))
    samples=pd.read_csv(R/'04_explanation_config/shap/core_samples.csv');orig=pd.read_csv(source(LOCAL/'04_explanation_config/shap/core_samples.csv'))
    ck('30预登记核心ID及顺序未改变',samples.to_json()==orig.to_json() and len(samples)==30 and samples.groupby('task').size().eq(3).all())
    shapfiles=list((R/'outputs/shap').glob('*/*_seed*.csv.gz'));ck('仅30核心SHAP文件',len(shapfiles)==30)
    scriptdiff=[]
    for p in (R/'05_scripts').glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'));old=LOCAL/'05_scripts'/p.name
        equal=old.exists() and ast.dump(ast.parse(old.read_text(encoding='utf-8-sig')))==ast.dump(ast.parse(p.read_text(encoding='utf-8-sig')))
        scriptdiff.append({'脚本':p.name,'AST与原迁移版相同':equal,'原SHA256':sha(old) if old.exists() else 'NA','回传SHA256':sha(p)})
        if not equal and old.exists():(OUT/('代码差异_'+p.stem+'.txt')).write_text(''.join(difflib.unified_diff(old.read_text(encoding='utf-8-sig').splitlines(True),p.read_text(encoding='utf-8-sig').splitlines(True))),encoding='utf-8')
    for n in ['portable_context.py','reused_shap_engine.py','reused_background_rule.py','tabiclv2_adapter.py','run_preflight.py','run_prediction_audit.py','run_shap.py','reused_ale_engine.py']:
        ck('核心算法源码 '+n,next(v['AST与原迁移版相同'] for v in scriptdiff if v['脚本']==n))
    results=[];audits=[];importance=[];p1s=[];backgrounds=[];expected_context={};arrays={}
    for task in tasks:
        c=next(v for v in cfg['contexts'] if v['task']==task and v['fold']=='fold_1')
        p=source(LOCAL/c['npz']);ck(task+'本地新有效输入哈希',sha(p)==c['effective_input_sha256']);a=np.load(p);arrays[task]=a
        y=a['y_train'];yf=np.log1p(y) if y.min()>=0 else y;ctx=ha(a['x_train'])+':'+ha(yf.astype('float32'));expected_context[task]=ctx
        impp=source(LOCAL/c['imputer']);imp=joblib.load(impp) # 已有可信本地SimpleImputer，只读取statistics_，不fit/transform/predict。
        raw=pd.read_csv(source(LOCAL/c['raw_test'])).set_index('obs_id')
        audit=json.loads((R/f'outputs/audit/{task}.json').read_text(encoding='utf-8-sig'))
        ck(task+'审查上下文身份',audit['context_hash']==ctx and audit['effective_input_sha256']==sha(p) and audit['configuration_sha256']==cs and audit['imputer_statistics_hash']==ha(imp.statistics_) and audit['checkpoint_sha256']==cfg['checkpoint_sha256'])
        auditdf=pd.read_csv(R/f'outputs/audit/{task}_fixed_samples.csv');selected=samples[samples.task.eq(task)];ref=pd.read_csv(source(LOCAL/c['prediction'])).set_index('obs_id')
        ck(task+'固定探针覆盖原核心',len(auditdf)==9 and set(auditdf.obs_id)==set(selected.obs_id) and auditdf.groupby('obs_id').size().eq(3).all())
        ck(task+'探针原批量参考值真实',np.allclose(auditdf.frozen_batch_prediction,ref.loc[auditdf.obs_id,'y_pred'],rtol=1e-12,atol=1e-12))
        ranges=auditdf.groupby('obs_id').p1_prediction.agg(lambda x:x.max()-x.min())
        ck(task+'P1重复范围记录一致',np.isclose(ranges.max(),audit['p1_repeat_max_range'],rtol=0,atol=1e-12))
        bg=pd.read_csv(R/f'outputs/shap/{task}/background_manifest.csv');ids=bg.obs_id.tolist();index={v:i for i,v in enumerate(a['train_ids'])};bgarray=a['x_train'][[index[i] for i in ids]]
        ck(task+'背景来自本折训练集且16条',len(bg)==16 and bg.obs_id.is_unique and set(ids)<=set(a['train_ids']))
        ck(task+'背景矩阵哈希',bg.background_hash.eq(ha(bgarray)).all())
        backgrounds.append({'task':task,'n':len(bg),'独立站点':len(set(i.split('_')[0] for i in ids)),'新背景SHA256':ha(bgarray),'背景IDs顺序SHA256':hi(ids)})
        index={v:i for i,v in enumerate(a['test_ids'])};p1dif=[]
        for sample in selected.itertuples():
            path=R/f'outputs/shap/{task}/{sample.obs_id}_seed42.csv.gz';d=pd.read_csv(path)
            ck(task+sample.obs_id+'77维顺序',len(d)==77 and d.feature.tolist()==fs and d.feature_order.tolist()==list(range(1,78)))
            ck(task+sample.obs_id+'身份/协议',d.dataset_version.eq(cfg['dataset_version']).all() and d.fold.eq('fold_1').all() and d.obs_id.eq(sample.obs_id).all() and d.protocol.eq('TABICLV2_POINTWISE_EXPLANATION_V1').all() and d.selection_phase.eq('CORE_3').all())
            ck(task+sample.obs_id+'新上下文/预处理血缘',d.context_hash.eq(ctx).all() and d.effective_input_sha256.eq(sha(p)).all() and d.imputer_statistics_hash.eq(ha(imp.statistics_)).all() and d.configuration_sha256.eq(cs).all() and d.checkpoint_sha256.eq(cfg['checkpoint_sha256']).all())
            ck(task+sample.obs_id+'训练测试顺序哈希',d.train_obs_order_hash.eq(hi(a['train_ids'])).all() and d.test_obs_order_hash.eq(hi(a['test_ids'])).all() and d.feature_order_hash.eq(hi(fs)).all())
            query=a['x_test'][index[sample.obs_id]:index[sample.obs_id]+1]
            ck(task+sample.obs_id+'真实查询/特征值',np.array_equal(d.feature_value.to_numpy(dtype='float32'),query[0]) and d.query_hash.eq(ha(query)).all())
            ck(task+sample.obs_id+'背景血缘',d.background_hash.eq(ha(bgarray)).all() and d.background_obs_order_hash.eq(hi(ids)).all())
            ck(task+sample.obs_id+'原SHAP预算',d.shap_seed.eq(42).all() and d.shap_max_evals.eq(155).all() and d.background_n.eq(16).all() and d.output_scale.eq('original_target').all())
            ck(task+sample.obs_id+'SHAP代码哈希',d.shap_script_sha256.eq(sha(R/'05_scripts/reused_shap_engine.py')).all())
            finite=np.isfinite(d[['shap_value','base_value','model_prediction','feature_value']].to_numpy()).all();error=abs(float(d.base_value.iloc[0])+d.shap_value.sum()-float(d.model_prediction.iloc[0]))
            ck(task+sample.obs_id+'有限及可加性',finite and error<=.02 and d.status.eq('COMPUTED_PASS').all())
            ck(task+sample.obs_id+'保存可加误差与复算一致',np.allclose(d.additivity_abs_error,error,rtol=0,atol=1e-10))
            p1audit=float(auditdf[auditdf.obs_id.eq(sample.obs_id)].p1_prediction.iloc[0]);p1=float(d.model_prediction.iloc[0]);delta=p1-p1audit;p1dif.append(abs(delta))
            ck(task+sample.obs_id+'SHAP预测与独立固定P1探针',abs(delta)<=1e-6,delta,'NUMERICAL')
            item={'task':task,'obs_id':sample.obs_id,'特征数':len(d),'独立背景数':len(bg),'可加性复算误差':error,'SHAP保存误差':d.additivity_abs_error.iloc[0],'SHAP基线':d.base_value.iloc[0],'SHAP模型预测':p1,'审查P1预测':p1audit,'SHAP减审查P1':delta,'正式批量预测':ref.loc[sample.obs_id,'y_pred'],'P1减正式批量':p1-float(ref.loc[sample.obs_id,'y_pred']),'实际模型调用数':int(d.actual_model_calls.iloc[0]),'计算秒':float(d.elapsed_seconds.iloc[0]),'非零特征数':int(d.shap_value.ne(0).sum())}
            results.append(item);importance.append(d[['task','feature','shap_value']].assign(abs_shap=d.shap_value.abs()))
            p1s.append({'task':task,'obs_id':sample.obs_id,'三次探针范围':float(ranges.loc[sample.obs_id]),'SHAP与审查P1差':delta})
        audits.append({'task':task,'严格批量复现状态':audit['batch_reproduction'],'最大绝对误差':audit['max_abs_error'],'平均绝对误差':audit['mean_abs_error'],'固定P1三次范围':float(ranges.max()),'SHAP与审查P1最大差':max(p1dif),'数值阈值_最大':1e-6,'数值阈值_均值':1e-7,'批量样本数':audit['batch_n'],'核心数':3})
    importance=pd.concat(importance);rank=importance.groupby(['task','feature'],sort=False).abs_shap.mean().reset_index(name='mean_abs_shap');rank['rank']=rank.groupby('task').mean_abs_shap.rank(method='min',ascending=False)
    summaryfile=next((R/'outputs/summary').glob('30*77*.csv'));returned=pd.read_csv(summaryfile)
    cmp=rank.merge(returned,on=['task','feature'],suffixes=('_review','_returned'),validate='one_to_one');ck('770项汇总和排名独立复算',len(cmp)==770 and np.allclose(cmp.mean_abs_shap_review,cmp.mean_abs_shap_returned,rtol=1e-10,atol=1e-12) and cmp.rank_review.eq(cmp.rank_returned).all())
    localdb=R/'outputs/checkpoints/ale_1d_local_effects.sqlite';dbhash=sha(localdb);con=sqlite3.connect('file:'+localdb.as_posix()+'?mode=ro&immutable=1',uri=True);ck('局部差值SQLite完整性',con.execute('PRAGMA integrity_check').fetchone()[0]=='ok');local=pd.read_sql_query('select * from local_effect',con);con.close();ck('只读SQLite未改变',sha(localdb)==dbhash)
    ale=[];revisions=[]
    for c in cfg['ale_curves']:
        task=c['task'];feat=c['feature'];key=f'{task}_{feat}';q=pd.read_csv(R/f'outputs/ale_1d/{key}_query_manifest.csv');oq=pd.read_csv(LOCAL/f'04_explanation_config/ale/{key}_query_manifest.csv');sup=pd.read_csv(R/f'outputs/ale_1d/{key}_bin_support.csv',float_precision='round_trip');osup=pd.read_csv(LOCAL/f'04_explanation_config/ale/{key}_bin_support.csv',float_precision='round_trip');curve=pd.read_csv(R/f'outputs/ale_1d/{key}_curve.csv');rep=pd.read_csv(R/f'outputs/ale_1d/{key}_repeatability.csv');v=local[local.task.eq(task)&local.feature.eq(feat)].sort_values('obs_id')
        ck(key+'原查询ID/边界保留',q.obs_id.tolist()==oq.obs_id.tolist() and np.array_equal(sup[['lower','upper']],osup[['lower','upper']]))
        a=arrays[task];cr=next(x for x in cfg['contexts'] if x['task']==task and x['fold']=='fold_1');raw=pd.read_csv(LOCAL/cr['raw_test']).set_index('obs_id');values=raw.loc[q.obs_id,feat].to_numpy();edges=np.array(sup.lower.tolist()+[float(sup.upper.iloc[-1])]);membership=np.clip(np.digitize(values,edges[1:-1],right=True),0,len(sup)-1)+1
        ck(key+'修复真值与分箱归属',np.allclose(q.feature_value,values) and np.array_equal(q.bin,membership) and np.isfinite(values).all() and (values>=edges[0]).all() and (values<=edges[-1]).all())
        changed=q.bin.ne(oq.bin)
        if changed.any():
            records=q.loc[changed,['obs_id','bin']].copy();records['旧bin']=oq.loc[changed,'bin'].values;records['feature']=feat;revisions.extend(records.to_dict('records'))
            ck(key+'唯一登记修订',task=='secchi' and feat=='ssrd_sum_3d' and set(records.obs_id)=={'NT1_20161107_surface','NM6_20161107_surface'} and records['旧bin'].eq(3).all() and records.bin.eq(4).all())
        ck(key+'SQLite样本完整/有限',len(v)==len(q) and v.obs_id.is_unique and set(v.obs_id)==set(q.obs_id) and np.isfinite(v[['prediction_lower','prediction_upper','local_effect']]).all().all())
        ck(key+'上下界预测差复算',np.allclose(v.prediction_upper-v.prediction_lower,v.local_effect,rtol=0,atol=1e-12))
        ck(key+'SQLite上下文',v.context_hash.eq(expected_context[task]).all())
        idx={obs:i for i,obs in enumerate(a['test_ids'])};expectedq=[ha(a['x_test'][idx[obs]]) for obs in v.obs_id];ck(key+'SQLite查询哈希',v.query_hash.tolist()==expectedq)
        bins=v.groupby('bin').local_effect.mean().reindex(curve.bin);n=v.groupby('bin').size().reindex(curve.bin);s=v.groupby('bin').station_id.nunique().reindex(curve.bin);cumulative=bins.cumsum();computed=cumulative-np.average(cumulative,weights=n)
        ck(key+'ALE均值/累积/中心化/n/s复算',np.allclose(curve.local_effect,bins,rtol=1e-10,atol=1e-12) and np.allclose(curve.ale,computed,rtol=1e-10,atol=1e-12) and np.array_equal(curve.n_samples,n) and np.array_equal(curve.n_stations,s))
        ck(key+'ALE新版本身份及脚本',curve.dataset_version.eq(cfg['dataset_version']).all() and curve.context_hash.eq(expected_context[task]).all() and curve.configuration_sha256.eq(cs).all() and curve.ale_script_sha256.eq(sha(R/'05_scripts/reused_ale_engine.py')).all())
        ck(key+'实际支持无空箱',n.ge(30).all() and s.ge(5).all() and len(curve)==c['bins'])
        rr=float(rep.local_effect.max()-rep.local_effect.min());ck(key+'重复差值记录',np.allclose(rep.prediction_upper-rep.prediction_lower,rep.local_effect,atol=1e-12) and np.allclose(curve.repeatability_max_local_delta,rr,atol=1e-12))
        ale.append({'task':task,'feature':feat,'分箱数':len(curve),'查询数':len(q),'最小bin_n':int(n.min()),'最小bin_站点':int(s.min()),'局部差值复算最大误差':float(np.max(abs(curve.local_effect.to_numpy()-bins.to_numpy()))),'ALE复算最大误差':float(np.max(abs(curve.ale.to_numpy()-computed.to_numpy()))),'重复局部差值范围':rr,'登记改变行数':int(changed.sum()),'数值状态':curve.status.iloc[0]})
    checks=pd.DataFrame(CHECKS);res=pd.DataFrame(results);au=pd.DataFrame(audits);al=pd.DataFrame(ale)
    save('01_逐项核验.csv',checks);save('02_30核心SHAP核验.csv',res);save('03_十任务预测复现与P1审查.csv',au);save('04_20条ALE核验.csv',al);save('05_770项SHAP汇总独立复算.csv',rank);save('06_代码版本差异.csv',scriptdiff);save('07_背景样本核验.csv',backgrounds);save('08_ALE登记修订明细.csv',revisions);save('09_本地依据SHA256.csv',SOURCES.values())
    pre=source(ROOT/'runs/tabiclv2_pilot/环境/.venv/Lib/site-packages/tabicl/_sklearn/preprocessing.py')
    code=pre.read_text(encoding='utf-8');ck('预处理集合顺序风险的源码证据','used_methods = list(set([config[1] for config in shuffle_norm_configs]))' in code,pre,'DIAGNOSTIC')
    report(res,au,al,rank,checks,R)
    for p,v in SOURCES.items():assert sha(p)==v['SHA256'],'核验期间原件改变'
    ck('回传ZIP未改变',sha(ZIP)==expected)
    status={'ZIP_SHA256':expected,'SHAP核心数':len(res),'任务数':res.task.nunique(),'SHAP特征行':int(len(res)*77),'汇总行':len(rank),'ALE曲线':len(al),'ALE实际bin':int(al['分箱数'].sum()),'核验项':len(CHECKS),'硬性失败':int(((checks['类别']=='HARD')&checks['结果'].eq('FAIL')).sum()),'数值比对失败':int(((checks['类别']=='NUMERICAL')&checks['结果'].eq('FAIL')).sum()),'批量严格复现通过':int(au['严格批量复现状态'].eq('PASS').sum()),'本机模型推理或解释重算':0}
    (OUT/'验收状态.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8');save('01_逐项核验.csv',CHECKS)
    files=[{'相对路径':p.relative_to(OUT).as_posix(),'SHA256':sha(p)} for p in OUT.iterdir() if p.is_file() and p.name!='核验交付哈希.csv'];save('核验交付哈希.csv',files)
    print(json.dumps(status,ensure_ascii=True),flush=True)

def report(res,au,al,rank,checks,R):
    hard=checks[(checks['类别']=='HARD')&checks['结果'].eq('FAIL')];num=checks[(checks['类别']=='NUMERICAL')&checks['结果'].eq('FAIL')]
    top=[]
    for task,g in rank.groupby('task',sort=False):top.append({'task':task,'Top5':'；'.join(g.nlargest(5,'mean_abs_shap').feature)})
    lines=['# 修复77特征SHAP回传核验报告','',
      '核验对象：TabICLv2_WQ77_EXPLANATION_RESULTS_20261120.zip。附件中的脚本和说明只作为审查证据，没有执行；独立计算均来自已有数值。原ZIP、迁移包及正式实验文件不改写。','',
      '## 审查结论','',
      f'回传覆盖十任务30个预登记核心样本×77特征，共{len(res)*77}个局部归因值；另含20条一维ALE。硬性血缘/算术失败{len(hard)}项，SHAP预测与独立P1探针的1e-6数值比对失败{len(num)}项。',
      '**本结果必须以P1单查询、3核心样本任务内解释使用；不能称为对正式完整测试批预测的严格等价解释，也不能称为十任务总体稳定性验证通过。**十项批量复现审查均未达到原最大1e-6/均值1e-7阈值，不能被run_status.exit_code=0或quality_audit的50 PASS覆盖。', '',
      '## 1. 文件与版本','',
      'ZIP SHA256与旁路文件相同，CRC、成员路径及PACKAGE_MANIFEST逐项哈希已检查。全部科学数据和代码均在清单；RESULTS_README.txt未列内部清单，是包装记录小缺口，外层ZIP哈希仍覆盖它。portable_config与本地原迁移版相同；30核心ID/顺序不变，未把种子重复算作独立样本。逐任务核对本地修复版NPZ、训练/测试ID顺序、77列顺序、官方checkpoint、新imputer统计哈希、每个SHAP查询float32值/哈希以及16训练背景矩阵哈希。',
      '目的机未回传独立pip freeze、GPU型号/驱动完整快照；配置锁版本和入口逻辑可以检查，但不据此推断具体设备或把数值差全部归为GPU。','',
      '## 2. SHAP数值完整性','',
      f'最大独立复算可加误差={res["可加性复算误差"].max():.12g}。每文件77列排序、特征值、原目标尺度、seed42、155评估预算、16背景、有限值及保存的可加性残差逐一核验。平均绝对SHAP按每任务3核心聚合，全部770项与回传汇总及任务内名次对照。',
      f'30个文件记录的实际模型调用合计{res["实际模型调用数"].sum():,}次；保存的计算时间合计{res["计算秒"].sum()/3600:.2f}小时（只是各文件记录的合计，不等于独立验证了GPU占用或墙钟日历时间）。', '',
      md(res[['task','obs_id','可加性复算误差','SHAP减审查P1','P1减正式批量','非零特征数']]),'',
      '可加性接近零只能证明当前SHAP分解与当前预测数值相加一致，不能证明解释稳定、代表总体或有因果含义。仅单次155预算Permutation归因，不是多次排列一致性结论。当前未见seed43结果，故未完成该种子敏感性审查，不补造重复。','',
      '## 3. 正式批量预测与P1边界','',md(au),'',
      '表中最大/平均批量误差为目的机已有审查记录，回传未含全批新复现预测向量，因此本机不能独立重算这两个全批统计量。已独立核对固定3样本的原冻结参考、三次P1范围及SHAP保存预测；对应明细见02/03号表。',
      '三次P1范围为0只覆盖30个固定核心输入的该次执行，不能外推到所有扰动样本、所有站点或历史失败事件。需要区分：输入矩阵/处理器/预训练权重血缘一致；全批严格数值复现失败；同一实例固定点P1重复范围为0；归因可加性通过。这四件事可以同时成立。',
      '**跨重建不一致是本轮核心遗留**：SHAP另一次上下文重建保存的预测与独立审查P1有18/30个样本超1e-6，涉及Chl-a表/中层、TN表/底层、TP表/底层六任务；另四任务三样本均一致。最大差为Chl-a表层SM13_20150515_surface约0.0143162，不能据同实例三次为0声称跨进程完全稳定。差值不等于SHAP贡献误差大小；不能由预测端点差直接推断排名一定改变或一定不变。',
      '源码风险线索：TabICL预处理_generate_ensemble用`used_methods = list(set(...))`生成视图顺序；回传run_all未显式向子进程固定PYTHONHASHSEED，context_hash仅覆盖输入矩阵和目标，不包括实际ensemble_configs_/视图顺序/完整内部预处理状态。random_state=42不自动锁定Python哈希种子。这是可见的不确定性来源，但本轮没有运行模型或取得各次实际视图顺序，不能断言它是全部数值差的已证实原因；AMP/GPU等也未被受控排除。',
      '如果使用了原迁移包的此入口，需要说明：该迁移入口只恢复正式保存的矩阵/处理器，没有把旧P1隔离patch和内部视图序固定机制作为新模型身份的一部分；不能用相同context_hash宣称完整预测函数已严格相同。', '',
      '## 4. 本批前五特征（描述性）','',md(pd.DataFrame(top)),'',
      'Top5来自本批3核心平均绝对SHAP，不用来重新挑选解释样本。不同水质任务原始SHAP量纲不同，不能横向比较绝对贡献；排名仅任务内部相对次序。','',
      '## 5. 随包ALE核验','',md(al),'',
      'SQLite用只读immutable模式打开；逐曲线检查保存的局部上下界预测差、查询ID/哈希、新上下文、bin内均值、累积与按样本数加权中心化、n与独立站点数，未调用模型或重新计算ALE预测。表中ALE复算只是对已有局部差值作算术核验。',
      'Secchi ssrd_sum_3d已在目的机以secchi_boundary_v2修订两条登记：NT1_20161107_surface、NM6_20161107_surface由第3箱移至第4箱；查询ID与六个边界保留。这是可追溯的登记修订，不能写成原登记毫无变化，也不能把原DATA_DRIFT记录删除或改称从未失败。其余19曲线边界精度修复记录已与源边界比对。',
      'Secchi曲线应在图注/数据说明中注明登记v2；来源配置中的原登记状态仍是需核对，最终结果与修订JSON要一同引用。','',
      '## 6. 代码差异和是否可用','',
      'SHAP核心算法、背景规则、上下文适配器与预测审查代码均按AST与原迁移版比对。ALE入口新增round_trip精度读写和显式Secchi两条边界修订；质量审查相应增加修订校验。原SHAP核心未被替换成TreeSHAP、旧模型或历史82/111特征计算。',
      '数据完整性和既有数值内部算术通过，建议保留为本版本P1解释候选，不应现在无条件标为正式解释质量全部通过。回传50 PASS的实际范围是30份SHAP的有限值/加和与20份ALE支持等内部检查，并未包含这里发现的跨重建P1一致性。因此它与本报告的待核验结论不矛盾。严格批量等价、跨重建一致性、种子敏感性与总体代表性仍不具备通过结论。',
      '最小下一步：在另一台计算机选最大差SM13_20150515_surface及一个当前无差的对照，跨进程各重建2—3次，仅做少量固定P1预测，同时记录实际norm视图顺序、ensemble配置、PYTHONHASHSEED、AMP/torch/tabicl/GPU版本及内部上下文哈希。此步骤无需重跑SHAP。先确定差异来源；如需锁定顺序，必须核对与正式预测函数的定义是否相同，不可默默修改模型再混用旧SHAP。只有证据表明原SHAP所解释函数受影响，再决定是否只重算受影响任务，避免全部重算。该建议本轮未执行。', '',
      '## 7. 本轮未执行事项','',
      '未运行回传的任何Python/PowerShell脚本；未加载TabICLv2模型；未训练、推理或计算SHAP/ALE。仅加载本地可信SimpleImputer序列化对象读取statistics_，未调用fit/transform/predict；原数据、预测和论文均不修改。']
    if len(hard):lines+=['','## 必须先处理的硬性问题','',md(hard)]
    if len(num):lines+=['','## SHAP对固定P1探针的额外数值不一致','',md(num)]
    (OUT/'SHAP回传结果核验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    main()
