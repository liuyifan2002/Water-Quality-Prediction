"""绘制图1-1研究技术路线；仅展示论文已登记的工作环节。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':['Microsoft YaHei','Arial'], 'font.size':9,
                     'pdf.fonttype':42, 'axes.unicode_minus':False})
fig, ax = plt.subplots(figsize=(15.5/2.54, 10.2/2.54))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 15.5); ax.set_ylim(0, 10.2); ax.axis('off')

def box(x,y,w,h,title,lines,fill='#F4F7F9'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10,rounding_size=0.12',
                               fc=fill,ec='#9AAAB6',lw=.75))
    ax.text(x+w/2,y+h-.32,title,ha='center',va='top',fontsize=9.5,fontweight='bold',color='#23384A')
    ax.text(x+w/2,y+(h-.45)/2-.08,'\n'.join(lines),ha='center',va='center',fontsize=8.4,
            color='#324655',linespacing=1.55)
def arrow(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=.9,color='#687F90'))

box(.25,7.50,4.6,2.0,'研究对象与预测目标',['香港近岸94站，2014—2024年','Chl-a／TN／TP三水层','Secchi单任务'])
box(5.45,7.50,4.6,2.0,'实际输入数据',['监测值、站点与采样日','六类实际环境数据','按站点与时点匹配'])
box(10.65,7.50,4.6,2.0,'特征与任务登记',['77特征，十项正式任务','78特征历史分区另列','按地理海区分区'])
arrow((4.91,8.5),(5.36,8.5));arrow((10.11,8.5),(10.56,8.5))
box(.25,4.00,4.6,2.3,'统一实验条件',['冻结留站五折','同任务、同特征比较','随机／年份验证另列'],fill='#EEF4F8')
box(5.45,4.00,4.6,2.3,'模型与扩展比较',['XGBoost／TabICLv2：十任务','TabPFN／TabM：四代表任务','历史分区与正式消融'],fill='#EEF4F8')
box(10.65,4.00,4.6,2.3,'评价分析',['R²／RMSE／MAE','站点改善与空间分布','指标、水层与任务差异'],fill='#EEF4F8')
arrow((12.95,7.39),(2.55,6.43));arrow((4.91,5.15),(5.36,5.15));arrow((10.11,5.15),(10.56,5.15))
box(2.1,.45,5.35,2.2,'模型解释',['TabICLv2：SHAP特征归因','一维ALE局部响应','样本、站点及数值支持'],fill='#F5F7F3')
box(8.05,.45,5.35,2.2,'综合讨论',['未见站点泛化与局地误差','环境信息的任务差异','分区、消融与解释边界'],fill='#F5F7F3')
arrow((12.95,3.89),(10.72,2.78));arrow((5.9,3.89),(4.77,2.78));arrow((7.53,1.55),(7.96,1.55))
fig.tight_layout(pad=.15)
for suffix,kwargs in [('png',{'dpi':600}),('pdf',{})]:
    fig.savefig(OUT/f'图1-1_研究技术路线.{suffix}',facecolor='white',**kwargs)
plt.close(fig)
