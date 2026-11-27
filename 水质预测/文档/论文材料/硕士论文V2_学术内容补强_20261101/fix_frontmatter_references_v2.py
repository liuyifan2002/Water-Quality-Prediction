"""修正V1遗留的摘要/研究目标模型归属，并补齐真实文献的正文引用。"""
from pathlib import Path
from docx import Document

OUT = Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文V2_学术内容补强_20261101")
SOURCE = OUT / "水质预测硕士学位论文_学术内容补强V2_最终版.docx"
TARGET = OUT / "水质预测硕士学位论文_学术内容补强V2_交付版.docx"
assert SOURCE.is_file() and not TARGET.exists()
doc=Document(SOURCE)

def one(prefix):
    a=[p for p in doc.paragraphs if p.text.startswith(prefix) and "\t" not in p.text]
    if len(a)!=1: raise ValueError((prefix,len(a)))
    return a[0]

one("香港近岸海域受珠江径流").text = (
    "香港近岸水质具有空间、水层与时间差异，模型在未见站点的预测能力需要与同站点插值能力区分。本文以2014—2024年香港近岸监测和多源环境数据构建叶绿素a、透明度、总氮与总磷共10项任务，在统一77项严格特征及冻结站点划分下比较XGBoost与TabICLv2，并在四项代表任务中纳入TabPFN、TabM。TabICLv2在十项正式任务的pooled R²均高于XGBoost；叶绿素a表层、透明度、总氮表层和总磷中层分别由0.457、0.535、0.259、0.304提高至0.589、0.602、0.402、0.472。叶绿素a三水层、透明度和总氮表层的站点改善及配对站点bootstrap证据较强，部分TN、TP任务虽有总体增益，却未获得同等强度的稳定性支持。解释分析以TabICLv2为主线：历史四任务五折结果与十任务第1折P1单查询结果分层报告，后者包含20条一维ALE曲线。数值重复性与分箱样本、站点覆盖分别审查；三组二维候选仅完成联合支持审查，未计算二维ALE响应曲面。数值等价受控复现XGBoost模型的TreeSHAP/ALE作为独立基准解释列于附录。来源级消融显示，空间静态信息对叶绿素a表层和透明度预测较重要，气象信息在透明度、总氮表层及总磷中层具有任务依赖性。结果表明，表格基础模型的优势随指标、水层与站点而变，解释曲线提供的是模型响应而非环境因果效应。"
)
one("Hong Kong coastal waters are jointly").text = (
    "Coastal water quality varies across locations, depths, and time; performance at unseen stations therefore needs to be distinguished from within-station interpolation. Using Hong Kong monitoring and multi-source environmental data from 2014–2024, this thesis defines ten prediction tasks for chlorophyll-a, Secchi depth, total nitrogen, and total phosphorus. XGBoost and TabICLv2 are compared using the same 77 strict features and frozen station-held-out splits; TabPFN and TabM are included for four representative tasks. TabICLv2 yields higher pooled R-squared than XGBoost in all ten registered tasks. For surface chlorophyll-a, Secchi depth, surface total nitrogen, and middle-layer total phosphorus, pooled R-squared increases from 0.457, 0.535, 0.259, and 0.304 to 0.589, 0.602, 0.402, and 0.472, respectively. Station-level gains and paired station bootstrap intervals provide stronger support for the chlorophyll-a depths, Secchi depth, and surface total nitrogen than for several other nutrient tasks. TabICLv2 is the primary explanation target: historical five-fold results for four tasks are reported separately from the first-fold P1 single-query analysis across ten tasks, including 20 one-dimensional ALE curves. Numerical repeatability and observation/station support are audited separately. Three two-dimensional candidates underwent joint-support assessment only; no two-dimensional ALE response surface was computed. TreeSHAP and ALE from numerically equivalent, controlled XGBoost refits are retained as an independent baseline explanation in the appendix. Source-group ablation indicates task-specific roles of spatial-static and meteorological information. The observed gains vary by target, depth, and station, while SHAP and ALE describe model responses rather than environmental causation."
)
one("本文旨在构建面向香港近岸").text = (
    "本文旨在构建面向香港近岸多指标、多水层任务的可复现预测与解释框架。工作包括：建立2014—2024年多源环境特征表；在十项正式站点留出任务比较XGBoost与TabICLv2；在四项代表任务纳入TabPFN和TabM；结合逐站点误差与站点bootstrap讨论空间泛化；利用冻结来源消融和分区结果分析任务差异；以TabICLv2为主要对象开展模型无关SHAP和一维ALE解释，并对二维候选只做联合支持审查。XGBoost受控复现模型的TreeSHAP/ALE作为独立基准解释置于附录D。"
)
one("水质预测经历了从经验统计模型").text += " 随机森林[8]与梯度提升框架[9]提供了传统树集成的两条方法线索，本文选择XGBoost作为统一树模型基准。"
one("XGBoost通过逐轮拟合残差方向").text += " TabICLv2的正式模型思想与实现见其ICML论文[13]。"
one("SHAP基于合作博弈").text += " 对解释对象、特征相关性和数值稳定性的区分亦符合可解释机器学习的通用方法讨论[10]。"
one("水质观测时间范围为2014—2024年").text += " 监测网络与官方水质数据来源见香港环境保护署资料[11]。"
one("模型输入由气象、降雨—水文—潮汐").text += " 气象再分析输入的来源记录为ERA5单层小时数据[12]。"
one("综合来看，已有研究仍可能存在").text = (
    "现有研究的关键挑战是验证对象与应用目标不匹配：同站点随机划分容易掩盖新站点误差，跨指标统一分数又会忽略量纲和样本量差异。本文通过统一77特征、冻结站点划分、逐站点误差、来源级消融和覆盖约束下的SHAP/ALE分析，把性能、解释和证据范围连接起来。"
)
one("本文的贡献属于应用与验证层面").text = (
    "本文的贡献集中在应用验证：将TabICLv2用于香港近岸多指标、多水层预测，并在同一77特征和严格留站划分下与XGBoost比较；通过pooled OOF、逐站点误差和站点bootstrap同时评价总体精度与空间覆盖；结合分区建模、来源消融及TabICLv2解释结果，识别不同水质任务对环境输入的统计依赖。"
)
one("[12] EUROPEAN CENTRE").text = (
    "[12] EUROPEAN CENTRE FOR MEDIUM-RANGE WEATHER FORECASTS. ERA5 hourly data on single levels from 1940 to present[DB/OL]. Copernicus Climate Change Service Climate Data Store, 2018. DOI: 10.24381/cds.adbb2d47. https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels."
)
doc.save(TARGET)
print("FRONTMATTER_AND_CITATIONS_FIXED",TARGET)
