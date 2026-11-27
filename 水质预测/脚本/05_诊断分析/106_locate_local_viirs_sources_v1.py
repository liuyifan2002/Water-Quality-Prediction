"""只读检索本地相关研究项目：CSV表头和夜光栅格名称；不读出凭证或令牌。"""
from pathlib import Path
import os,csv,json,re

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/图片数据完整性整改_v1_20261115/证据'
roots=[ROOT/'数据',ROOT/'transfer',ROOT/'codex/worktrees',Path('E:/研究生/codex_chla_optimization'),Path('E:/研究生/workbuddy'),Path('E:/研究生/cloude code'),Path('E:/研究生/gis'),Path('C:/Users/刘逸凡/Desktop/水质预测'),Path('C:/Users/刘逸凡/Desktop/开题'),Path('E:/刘逸凡/Documents'),Path('E:/迅雷下载'),Path('E:/迅雷云盘'),Path('E:/备份/BaiduNetdiskDownload')]
skip={'.git','.venv','venv','node_modules','site-packages','__pycache__','.cache','.next','nasa_oceancolor_monthly_9km','tabicl_explanation_portable_bundle_v2','tabicl_v2_slim_explainability_v2_all10_portable_bundle_v1','tabicl_v2_slim_explainability_v2_all10_portable_bundle_v1_prevalidation','_portable_bundle_audit_tmp'}
roots.append(Path('D:/Users/刘逸凡/Downloads'))
hits=[];scanned=[];seen=set()
for base in roots:
    nc=0;nf=0
    if not base.is_dir():continue
    for folder,dirs,files in os.walk(base):
        dirs[:]=[d for d in dirs if d not in skip]
        for name in files:
            p=Path(folder)/name
            if str(p).casefold() in seen:continue
            seen.add(str(p).casefold());nf+=1
            if p.suffix.lower()=='.csv':
                nc+=1
                try:
                    with p.open('rb') as f:header=f.readline(16384).decode('utf-8-sig',errors='replace')
                    cols=[x.strip('"\r\n') for x in header.split(',')]
                    target=[c for c in cols if 'ntl_avg_rad' in c.lower() or c.lower() in ['avg_rad','avg_rade9h','cf_cvg']]
                    if target:hits.append({'路径':str(p),'类型':'CSV含夜光字段','字节数':p.stat().st_size,'夜光字段':';'.join(target)})
                except (OSError,PermissionError):pass
            elif re.search(r'ntl|night.?light|vcmsl|vnl|夜光',name,re.I) and p.suffix.lower() in ['.tif','.tiff','.gz','.tgz','.zip','.xlsx','.xls','.nc','.csv']:
                hits.append({'路径':str(p),'类型':'夜光名称候选','字节数':p.stat().st_size,'夜光字段':'待核对'})
    scanned.append({'目录':str(base),'检查文件数':nf,'检查CSV表头数':nc})
for name,rows in [('本地夜光候选来源清单.csv',hits),('本地查找范围.csv',scanned)]:
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['路径']);w.writeheader();w.writerows(rows)
print(json.dumps({'检查范围':scanned,'夜光候选数':len(hits),'前30个候选':hits[:30]},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':pass
