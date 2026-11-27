"""对本轮新包作静态终检修订，另存v2并重新压缩解压；不运行任何计算入口。"""
from pathlib import Path
import json,shutil,hashlib,ast,zipfile,subprocess
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'交付/修复77特征解释计算迁移_20261116_v1'
DEST=ROOT/'交付/修复77特征解释计算迁移_20261116_v3'
NAME='TabICLv2_SHAP_ALE_PORTABLE_WQ77_REPAIRED_20261115_V1.zip'
PY=ROOT/'runs/tabiclv2_pilot/环境/.venv/Scripts/python.exe'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not DEST.exists(),'终检版已经存在，拒绝覆盖'
    bundle=DEST/'TabICLv2_SHAP_ALE_PORTABLE';shutil.copytree(BASE/bundle.name,bundle,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    templates=Path(__file__).parent/'wq77_portable_templates'
    for name in ['run_prediction_audit.py','run_quality_audit.py']:shutil.copy2(templates/name,bundle/'05_scripts'/name)
    p=bundle/'05_scripts/reused_ale_engine.py';s=p.read_text(encoding='utf-8');needle='    curve["context_hash"] = meta["context_hash"]'
    assert s.count(needle)==1
    s=s.replace(needle,needle+'\n    curve["dataset_version"] = meta["dataset_version"]\n    curve["configuration_sha256"] = meta["configuration_sha256"]');p.write_text(s,encoding='utf-8')
    # 两条边界冲突的精确来源证据；不重新分箱、不改变原登记。
    run=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115';q=pd.read_csv(bundle/'04_explanation_config/ale/secchi_ssrd_sum_3d_query_manifest.csv')
    new=pd.read_csv(run/'03_正式十任务重算/inputs/secchi_fold_1_测试未填补有效特征.csv').set_index('obs_id')
    records=[]
    for obs in ['NT1_20161107_surface','NM6_20161107_surface']:
        old=q[q.obs_id.eq(obs)].iloc[0];v=float(new.loc[obs,'ssrd_sum_3d']);assert old.feature_value==50345940 and v==50345942
        records.append({'task':'secchi','feature':'ssrd_sum_3d','obs_id':obs,'原登记值':float(old.feature_value),'修复有效原值':v,'原登记bin':int(old.bin),'原箱上边界':50345940,'差值_J_m2':v-float(old.feature_value),'相对差':(v-float(old.feature_value))/float(old.feature_value),'处理':'保留原ID/边界，迁移端提示登记冲突；不自动改变箱号'})
    pd.DataFrame(records).to_csv(bundle/'04_explanation_config/ale/secchi_boundary_precision_evidence.csv',index=False,encoding='utf-8-sig')
    appendix='\n\n## 终检补充：边界差精确量级\n\nSecchi的两条记录为NT1_20161107_surface、NM6_20161107_surface；原登记值/第3箱上边界50,345,940 J/m²，新有效原值50,345,942 J/m²，差2 J/m²（约4×10⁻⁸相对差），落到相邻第4箱。该证据只证明数值边界不一致，不能宣称数据大幅漂移或样本不足，也不自动决定舍入规则；迁移程序保持原登记并提示核对。证据见04_explanation_config/ale/secchi_boundary_precision_evidence.csv。\n\n终检另增强：目的端预测审查按任务捕获失败后继续其他任务；ALE输出显式附新版本/配置哈希，质量审查逐曲线核对新上下文；缺项或质量失败返回非零，避免将脚本退出当作全量通过。均为静态代码检查，本机未运行这些推理分支。\n'
    for name in ['03_missing_assets.md','MISSING_ASSETS.md','02_asset_provenance_audit.md','PACKAGING_REPORT.md']:
        p=bundle/'08_reports'/name;p.write_text(p.read_text(encoding='utf-8')+appendix,encoding='utf-8')
    # 已保存的旧背景ID作为历史登记证据单独归档；从不作为新上下文数值。
    old=ROOT/'runs/tabicl_v2_slim_explainability_v2_all10';background=[]
    config=json.loads((bundle/'04_explanation_config/portable_config.json').read_text(encoding='utf-8'))
    extra=[]
    for task in config['tasks']:
        p=old/'shap'/task/'background_manifest.csv';background.append({'task':task,'原背景ID清单是否存在':p.exists(),'新任务背景执行方式':'原16背景规则在新训练上下文上执行；目的端保存新IDs，不改30核心ID'})
        if p.is_file():
            dest=bundle/'07_reference/historical_registration/background_ids'/f'{task}.csv';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(p)==sha(dest)
            extra.append({'源路径':str(p),'包内路径':dest.relative_to(bundle).as_posix(),'用途':'旧背景ID仅为历史证据，不作为新背景值','数据版本':'LEGACY_PLAN_ONLY','字节':p.stat().st_size,'SHA256':sha(p),'状态':'COPIED_HASH_VERIFIED'})
    pd.DataFrame(background).to_csv(bundle/'08_reports/背景登记完整性.csv',index=False,encoding='utf-8-sig')
    inventory=bundle/'08_reports/01_asset_inventory.csv';inv=pd.read_csv(inventory)
    for name in ['run_prediction_audit.py','run_quality_audit.py']:
        p=bundle/'05_scripts'/name;ix=inv['包内路径'].eq('05_scripts/'+name);inv.loc[ix,'SHA256']=sha(p);inv.loc[ix,'字节']=p.stat().st_size
    pd.concat([inv,pd.DataFrame(extra)],ignore_index=True).to_csv(inventory,index=False,encoding='utf-8-sig')
    for name in ['124_build_repaired_tabicl_explanation_portable_v1.py','125_finalize_repaired_explanation_portable_review_v1.py']:
        p=Path(__file__).parent/name;shutil.copy2(p,bundle/'07_reference/method_source'/(name+'.txt'))
    for p in (bundle/'05_scripts').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    for p in bundle.rglob('*.json'):json.loads(p.read_text(encoding='utf-8-sig'))
    entries=[]
    for p in sorted(bundle.rglob('*')):
        if p.is_file() and p.name not in ['MANIFEST.csv','SHA256SUMS.txt']:entries.append({'relative_path':p.relative_to(bundle).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    pd.DataFrame(entries).to_csv(bundle/'08_reports/MANIFEST.csv',index=False,encoding='utf-8-sig')
    (bundle/'08_reports/SHA256SUMS.txt').write_text('\n'.join(f'{v["sha256"]}  {v["relative_path"]}' for v in entries)+'\n',encoding='utf-8')
    result=subprocess.run([str(PY),'-B',str(bundle/'05_scripts/run_preflight.py'),'--static-only'],cwd=DEST,capture_output=True,text=True,encoding='utf-8',check=True)
    (DEST/'本机静态预检.json').write_text(result.stdout,encoding='utf-8')
    zipfile_path=DEST/NAME
    with zipfile.ZipFile(zipfile_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,arcname=bundle.name+'/'+p.relative_to(bundle).as_posix())
    zippedsha=sha(zipfile_path);(DEST/(NAME+'.sha256')).write_text(zippedsha+'  '+NAME+'\n',encoding='utf-8')
    with zipfile.ZipFile(zipfile_path) as z:
        assert z.testzip() is None
        z.extractall(DEST/'解压核验')
    ex=DEST/'解压核验'/bundle.name
    for v in entries:assert sha(ex/v['relative_path'])==v['sha256'] and (ex/v['relative_path']).stat().st_size==v['bytes']
    for name in ['MANIFEST.csv','SHA256SUMS.txt']:assert sha(ex/'08_reports'/name)==sha(bundle/'08_reports'/name)
    result=subprocess.run([str(PY),'-B',str(ex/'05_scripts/run_preflight.py'),'--static-only'],cwd=DEST/'解压核验',capture_output=True,text=True,encoding='utf-8',check=True)
    (DEST/'解压后静态预检.json').write_text(result.stdout,encoding='utf-8')
    # 拷贝溯源清单和原件再次只读验证；来源模板版本变更已显式登记。
    for v in pd.read_csv(inventory).to_dict('records'):assert sha(v['源路径'])==v['SHA256'],v['源路径']
    state={'状态':'静态迁移包终检通过；1条ALE原登记边界精度需核对','数据版本':config['dataset_version'],'文件清单数':len(entries),'TabICLv2新上下文折':50,'新imputer':50,'原100任务模型折预测文件':100,'核心SHAP样本':30,'ALE原登记':20,'边界需核对曲线':'Secchi ssrd_sum_3d：2记录、2J/m²边界差','ZIP字节':zipfile_path.stat().st_size,'ZIP_SHA256':zippedsha,'所有清单哈希/解压/语法/相对路径':'PASS','本机模型加载/训练/推理/SHAP/ALE次数':0,'目的端数值资格':'未运行，不宣称通过'}
    (DEST/'最终迁移包验收.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(state,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
