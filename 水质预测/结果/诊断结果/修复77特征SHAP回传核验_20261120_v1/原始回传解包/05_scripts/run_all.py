"""迁移机串行入口；任一任务失败记录后继续其他单元，不会启动二维ALE。"""
import subprocess,sys,os,json
from portable_context import *
def main():
    from run_preflight import preflight
    preflight(dependencies=True);logs=OUT/'logs';logs.mkdir(parents=True,exist_ok=True);statuses=[]
    for script in ['run_prediction_audit.py','run_shap.py','run_ale.py','run_quality_audit.py']:
        with (logs/(script+'.log')).open('a',encoding='utf-8') as f:
            p=subprocess.run([sys.executable,str(ROOT/'05_scripts'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        statuses.append({'script':script,'exit_code':p.returncode});atomic_json(OUT/'run_status.json',statuses)
    if any(v['exit_code'] for v in statuses):raise SystemExit('部分任务未完成，见outputs/logs和status；不可宣称全完成')
if __name__=='__main__':main()
