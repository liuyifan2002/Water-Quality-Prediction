"""核验全量数据审查交付、修复候选和原件哈希；不更改正式数据。"""
from pathlib import Path
import hashlib,json,py_compile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115';E=OUT/'证据'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    hashes=json.loads((E/'全数据原件SHA256.json').read_text(encoding='utf-8'))
    bad=[]
    for i,(path,h) in enumerate(hashes.items()):
        p=Path(path)
        if not p.exists() or digest(p)!=h:bad.append(path)
        if i%1000==0:print('原件SHA256已检查',i+1,'/',len(hashes),flush=True)
    assert not bad,bad
    names=[f'{i:02d}_{n}' for i,n in [(1,'全量数据复核报告.md'),(2,'历史候选及排除数据逐表复核.csv'),(3,'历史排除决定原文登记.csv'),(4,'历史数据说明与有效性复核范围.csv'),(5,'全部既有数据获取与构建方式.csv'),(6,'77项正式特征逐项核验.csv'),(7,'全部问题与处理状态.csv'),(8,'真实补取与修复记录.csv'),(9,'受影响实验重审与重算清单.csv'),(10,'历史无效果结论复核说明.md'),(11,'未闭环问题与后续执行顺序.md')]]
    for n in names:assert (OUT/n).stat().st_size>100,n
    original=pd.read_csv(ROOT/'数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv',low_memory=False)
    candidate=pd.read_csv(OUT/'修复候选/正式77特征_全环境覆盖修复候选_未用于实验.csv',low_memory=False)
    strict=pd.read_csv(ROOT/'runs/tabiclv2_pilot/01_strict_feature_manifest.csv');features=strict.loc[strict.included_in_pilot.astype(str).str.lower().eq('true'),'feature'].tolist()
    assert len(features)==77 and candidate.columns[-77:].tolist()==features
    for c in ['obs_id','sample_id','station','depth','chla_ug_l','secchi_depth_m','tn_mg_l','tp_mg_l']:assert original[c].equals(candidate[c]),c
    assert pd.to_datetime(original.date).equals(pd.to_datetime(candidate.date))
    for n in ['05_原始水质全部字段溯源核对.csv','06_坐标原始矢量核对.csv','07_272列主表逐项重建核对.csv']:
        d=pd.read_csv(E/n);assert d.不一致行数.eq(0).all(),n
    assert pd.read_csv(E/'17_NASA两核心产品24816格原始复核.csv').一致.all()
    assert pd.read_csv(E/'23_CMEMS扩边重叠原始数值核验.csv').差异.eq(0).all()
    assert pd.read_csv(E/'25_气象小时原件与月日汇总核验.csv').差异.eq(0).all()
    assert pd.read_csv(E/'26_正式冻结预测真值与测试ID.csv')['真值与原始水质不一致'].eq(0).all()
    met=pd.read_csv(E/'27_正式FamilyV2冻结预测指标一致性.csv');assert met[['登记R2一致','RMSE一致','MAE一致']].all().all()
    assert pd.read_csv(E/'28_真实高分影像导出与处理逐字段核对.csv').导出与处理差异.eq(0).all()
    assert pd.read_csv(E/'31_ERA5前史补取重叠验证.csv').差异.eq(0).all()
    scripts=[ROOT/'脚本/05_诊断分析'/n for n in ['109_inventory_all_data_quality_v1.py','110_validate_full_data_lineage_v1.py','111_review_excluded_data_and_sources_v1.py','112_verify_historical_raw_sources_v1.py','113_build_full_data_review_reports_v1.py','114_check_source_identity_and_frozen_predictions_v1.py','115_validate_full_data_review_delivery_v1.py']]
    scripts += [ROOT/'脚本/02_特征构建'/n for n in ['28_rebuild_full_station_environment_candidate_v1.py','29_fetch_cmems_full_station_extent_v1.py','30_rebuild_cmems_candidate_from_daily_raw_v1.py','32_fetch_era5_pre2014_support_v1.py']]
    scripts += [ROOT/'脚本/01_数据准备/03_fetch_verified_epd_marine_history_v1.py']
    for p in scripts:py_compile.compile(str(p),doraise=True)
    # 补取与输出另列哈希，不把新增原件误记为旧文件。
    outputs={str(p.relative_to(ROOT)):digest(p) for p in scripts}
    for folder in ['修复候选','真实补取']:
        for p in (OUT/folder).rglob('*'):
            if p.is_file():outputs[str(p.relative_to(ROOT))]=digest(p)
    for n in names:outputs[str((OUT/n).relative_to(ROOT))]=digest(OUT/n)
    (E/'修复输出与脚本SHA256.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8')
    expected={'水质预测硕士学位论文_最终定点排版修订版.docx':'94ede7bb400ca71854a7f4fccea887973a00d96de045247bb94d5414de8cf668','水质预测硕士学位论文_最终定点排版修订版.pdf':'153ff79fa140c8e66f09c4d13c15b3f51989318c87ad08a7ff97412cca953aa1'}
    for n,h in expected.items():assert digest(ROOT/'结果/论文最终定点排版修订_v1_20261115'/n)==h
    epd=pd.read_csv(OUT/'真实补取/EPD真实海水年度_1986_2024/真实海水年度下载及内容验收.csv')
    assert len(epd)==39 and epd.状态.str.startswith('已补取').all()
    check={'受保护原件':len(hashes),'原件变更':bad,'12份代码语法检查':True,'77顺序/目标/站点/日期/水层/观测顺序保留':True,'论文DOCX及PDF未改':True,'真实EPD年度':len(epd),'真实EPD原始记录':int(epd.记录数.sum()),'科学模型训练/推理/SHAP/ALE':False,'候选可直接与旧预测混用':False,'全部科学问题已经解决':False,'未闭环清单':'11_未闭环问题与后续执行顺序.md'}
    (OUT/'12_原件保护与交付验收.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(check,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
