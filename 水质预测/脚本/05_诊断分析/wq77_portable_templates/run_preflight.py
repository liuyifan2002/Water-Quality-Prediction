"""默认只做静态文件检查；--dependencies供迁移机验证依赖/GPU，不做推理。"""
from pathlib import Path
import csv,json,ast,sys,importlib.metadata
ROOT=Path(__file__).resolve().parents[1]
def preflight(dependencies=False):
    from portable_context import file_sha,config
    cfg=config();issues=[];count=0
    with (ROOT/'08_reports/MANIFEST.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            p=(ROOT/r['relative_path']).resolve()
            if not p.is_relative_to(ROOT.resolve()) or not p.is_file():issues.append('缺失/越界 '+r['relative_path']);continue
            count+=1
            if p.stat().st_size!=int(r['bytes']) or file_sha(p)!=r['sha256']:issues.append('哈希错误 '+r['relative_path'])
    for p in (ROOT/'05_scripts').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    for p in ROOT.rglob('*.json'):
        if 'outputs' not in p.parts:json.loads(p.read_text(encoding='utf-8-sig'))
    for r in cfg['contexts']:
        for k in ['npz','imputer','raw_train','raw_test','prediction','manifest']:
            p=Path(r[k])
            if p.is_absolute() or not (ROOT/p).is_file():issues.append('上下文路径错误 '+str(p))
    if len(cfg['features'])!=77 or len(set(cfg['features']))!=77 or len(cfg['contexts'])!=50:issues.append('77列/50折上下文不完整')
    if dependencies:
        if sys.version_info[:3]!=(3,11,9):issues.append('Python必须与源3.11.9一致')
        for k,v in cfg['runtime_versions'].items():
            try:
                if importlib.metadata.version(k)!=v:issues.append(f'{k}版本不同，需要{v}')
            except importlib.metadata.PackageNotFoundError:issues.append('缺依赖 '+k)
        if not issues:
            import torch
            if not torch.cuda.is_available():issues.append('CUDA GPU不可用，禁止静默CPU回退')
    result={'status':'FAIL' if issues else 'PASS','files_checked':count,'dependencies_checked':dependencies,'issues':issues,'inference_executed':False}
    if issues:raise RuntimeError(json.dumps(result,ensure_ascii=False))
    return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dependencies',action='store_true');p.add_argument('--static-only',action='store_true');a=p.parse_args()
    print(json.dumps(preflight(dependencies=a.dependencies),ensure_ascii=False,indent=2))
