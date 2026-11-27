"""读取原随机/年份/站点划分，核验时间边界及同期月合成适用范围；不训练预测。"""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'结果/扩展实验/候选数据验收与一致性重算_v1_20261115';OUT=RUN/'07_时间协议与空间核验'
def main():
    p=ROOT/'runs/tabiclv2_pilot/02_split_manifest.csv';s=pd.read_csv(p,parse_dates=['date']);daily=pd.read_csv(RUN/'02_冻结候选/ERA5逐日气候态来源.csv',parse_dates=['date']);rows=[]
    for (task,validation,fold),g in s.groupby(['task','validation','fold']):
        tr=g[g.split.eq('train')];te=g[g.split.eq('test')];end=tr.date.max();start=te.date.min();overlap=len(set(tr.station)&set(te.station));obs=len(set(tr.obs_id)&set(te.obs_id));ref=daily[daily.date.between('2014-01-01',end)]
        row={'task':task,'validation':validation,'fold':fold,'train_n':len(tr),'test_n':len(te),'train_min':tr.date.min(),'train_max':end,'test_min':start,'test_max':te.date.max(),'station_overlap':overlap,'obs_overlap':obs,'train_max_before_test_min':bool(end<start),'新气候态参考末日':ref.date.max(),'气候态参考日数':len(ref),'新预测是否执行':'否；本表仅检查既有协议，不能作为新性能结果','范围边界':'原TabICLv2 pilot注册，不是新FamilyV2十任务统一注册','原manifest_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
        assert obs==0
        if 'year' in validation:assert end<start and ref.date.max()<=end
        if 'station' in validation:assert overlap==0
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'原注册随机年份留站点协议边界.csv',index=False,encoding='utf-8-sig')
    d=pd.read_csv(RUN/'02_冻结候选/基础输入_77特征.csv',parse_dates=['date']);sample=d.drop_duplicates('sample_id').copy();sample['当月最后自然日']=sample.date+pd.offsets.MonthEnd(0);sample['月末晚于采样日天数']=(sample['当月最后自然日']-sample.date).dt.days
    sample[['sample_id','station','date','当月最后自然日','月末晚于采样日天数']].to_csv(OUT/'同期月合成可用时点保守下界.csv',index=False,encoding='utf-8-sig')
    stats={'样本层记录':len(sample),'月末晚于采样日记录':int(sample['月末晚于采样日天数'].gt(0).sum()),'比例':float(sample['月末晚于采样日天数'].gt(0).mean()),'中位晚于日数':float(sample['月末晚于采样日天数'].median()),'上界晚于日数':int(sample['月末晚于采样日天数'].max()),'说明':'月末是完整当月合成可用时刻的保守下界，未假设实际产品在月末即时发布；月末当天也不证明采样时刻前可用','本版定位':'历史同期/留站点泛化，不是实时提前预测','年份留出性能是否重算':False}
    (OUT/'同期信息与提前预测边界.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'时间协议补充审查.md').write_text('# 时间协议补充审查\n\n'+f'读取原已注册{len(rows)}组随机、年份、留站点划分；未生成或更改任何划分，没有新增这些协议下的模型性能结果。具体边界见CSV。\n\n'+
      f'10866个采样时点中，{stats["月末晚于采样日记录"]}个（{stats["比例"]:.2%}）的当月完整合成月末晚于采样日，中位相差{stats["中位晚于日数"]}日。这不是数据提取错误，而是当前同期解释/泛化协议的适用边界。气象/夜光/月雨的真实发布时间通常更晚，未凭空指定。\n\n'+
      '新SST气候态拟合规则在原年份切分训练截止日之前取参考日；测试期日值只用于该测试样本自身的外生输入，不用于拟合气候态。随机划分不是特征空间插值的保证，留站点与年份评价的泛化对象不同。\n\n'+
      '本候选正式执行的是原十任务留站点协议。保留完整当月信息时，年份留出只能解释为历史同期协变量条件下的年份泛化，不能写成实时提前预报。严格实时版本还需原始小时日界、采样时刻、发布时间与所有派生滞后重新冻结并重算，不与本版混用。\n',encoding='utf-8')
    print('TIME_BOUNDARIES_PASS',len(rows),stats,flush=True)
if __name__=='__main__':main()
