"""仅迁移机执行：调用原99号Permutation SHAP核心，使用新版本上下文。"""
import argparse,json,traceback
from portable_context import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS);p.add_argument('--sensitivity',action='store_true');a=p.parse_args();failed=[]
    with gpu_lock():
        for task in ([a.task] if a.task else TASKS):
            try:
                prerequisite(task)
                module=load_module(ROOT/'05_scripts/reused_shap_engine.py','reused_wq77_shap')
                module.main(task,'SENSITIVITY_1' if a.sensitivity else 'CORE_3')
            except Exception as e:
                failed.append(task);atomic_json(OUT/'status'/f'shap_{task}.json',{'status':'FAILED','error':str(e),'traceback':traceback.format_exc()})
            else:atomic_json(OUT/'status'/f'shap_{task}.json',{'status':'COMPUTED','phase':'SENSITIVITY_1' if a.sensitivity else 'CORE_3'})
    if failed:raise SystemExit('失败任务（其他任务已继续）：'+','.join(failed))
if __name__=='__main__':main()
