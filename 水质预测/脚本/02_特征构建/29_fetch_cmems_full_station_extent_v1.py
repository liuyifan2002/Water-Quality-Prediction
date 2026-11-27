"""按真实94站及网格边距补取CMEMS；原始产品单独存放，凭据不写入输出。"""
from pathlib import Path
import json,os
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'结果/诊断结果/全量数据来源与质量复核_v1_20261115'

def main():
    import copernicusmarine as cm
    folder=OUT/'真实补取/CMEMS全站扩边';folder.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    # 读取用户原有配置，不显示或另存任何凭据。
    path=ROOT/'codex/worktrees/9ff7/水质预测/.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k,v=line.split('=',1);env.setdefault(k.strip(),v.strip().strip('\"').strip("'"))
    meta=pd.read_csv(ROOT/'数据/处理后数据/正式主表/station_meta.csv')
    states=[]
    for family,id_,variables,margin in [('physics','cmems_mod_glo_phy_my_0.083deg_P1D-m',['thetao','so','uo','vo','zos','mlotst'],.09),('bgc','cmems_mod_glo_bgc_my_0.25deg_P1D-m',['chl','no3','po4','si','o2','nppv'],.26)]:
        request=dict(dataset_id=id_,dataset_version='202311' if family=='physics' else '202406',variables=variables,minimum_longitude=float(meta.lon.min()-margin),maximum_longitude=float(meta.lon.max()+margin),minimum_latitude=float(meta.lat.min()-margin),maximum_latitude=float(meta.lat.max()+margin),start_datetime='2013-12-15',end_datetime='2024-12-31',coordinates_selection_method='outside',output_filename=f'{family}_94站扩边_20131215_20241231.nc',output_directory=str(folder),disable_progress_bar=True,skip_existing=True)
        state={'来源':family,'请求':request,'状态':'开始实际补取','凭据已配置':bool(env.get('COPERNICUSMARINE_USERNAME') and env.get('COPERNICUSMARINE_PASSWORD'))}
        states.append(state);(folder/'同源扩边补取状态.json').write_text(json.dumps(states,ensure_ascii=False,indent=2),encoding='utf-8')
        try:
            cm.subset(**request,username=env.get('COPERNICUSMARINE_USERNAME'),password=env.get('COPERNICUSMARINE_PASSWORD'))
            state['状态']='已补取，待版本与重叠逐值验收';state['字节']=(folder/request['output_filename']).stat().st_size
        except Exception as ex:
            # 仅保存错误类型，防止第三方异常携带认证内容。
            state['状态']='补取未完成';state['错误类型']=type(ex).__name__
        (folder/'同源扩边补取状态.json').write_text(json.dumps(states,ensure_ascii=False,indent=2),encoding='utf-8')
        print(family,state['状态'],state.get('错误类型',''),flush=True)

if __name__=='__main__':main()
