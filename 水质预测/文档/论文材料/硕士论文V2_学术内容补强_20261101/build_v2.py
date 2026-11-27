"""从V1唯一母本生成V2；只读取冻结实验材料，不调用模型。"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(r"E:\研究生\codex\水质预测")
OUT = ROOT / "文档/论文材料/硕士论文V2_学术内容补强_20261101"
SOURCE = ROOT / "文档/论文材料/硕士论文解释章节重构V1_20261101/水质预测硕士学位论文_解释章节重构V1.docx"
DOCX = OUT / "水质预测硕士学位论文_学术内容补强V2.docx"
RUN = ROOT / "runs"


def find(doc: Document, prefix: str):
    matches = [p for p in doc.paragraphs if p.text.startswith(prefix) and not p.style.name.startswith("toc")]
    if len(matches) != 1:
        raise ValueError((prefix, len(matches)))
    return matches[0]


def set_text(doc: Document, prefix: str, value: str):
    p = find(doc, prefix)
    p.text = value
    return p


def after(paragraph, text: str, style: str | None = None):
    p = paragraph.insert_paragraph_before(text, style=style)
    paragraph._p.addnext(p._p)
    return p


def before(paragraph, text: str, style: str | None = None):
    return paragraph.insert_paragraph_before(text, style=style)


def font_run(run, size=10.5):
    run.font.size = Pt(size)
    run.font.name = "宋体"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def set_cell(cell, value: str, size=8.5):
    cell.text = value
    for p in cell.paragraphs:
        p.paragraph_format.space_after = Pt(0)
        for run in p.runs:
            font_run(run, size)
    tcpr = cell._tc.get_or_add_tcPr()
    if tcpr.find(qn("w:noWrap")) is None:
        tcpr.append(OxmlElement("w:noWrap"))


def table_note_after(table, text: str):
    p = table._parent.add_paragraph(text)
    table._element.addnext(p._p)
    for r in p.runs:
        font_run(r, 8)
    return p


def make_corrected_distribution() -> Path:
    data = pd.read_csv(ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv",
                       usecols=["sample_id", "depth", "chla_ug_l", "secchi_depth_m", "tn_mg_l", "tp_mg_l"])
    sec = data[["sample_id", "secchi_depth_m"]].dropna().drop_duplicates("sample_id")
    assert sec.sample_id.is_unique
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.0), constrained_layout=True)
    items = [("chla_ug_l", "Chl-a (µg/L)"), ("secchi_depth_m", "Secchi depth (m)"),
             ("tn_mg_l", "TN (mg/L)"), ("tp_mg_l", "TP (mg/L)")]
    for ax, (col, label) in zip(axs.flat, items):
        if col == "secchi_depth_m":
            arrays = [sec.secchi_depth_m.to_numpy()]
            labels = ["Station–date"]
        else:
            arrays = [data.loc[data.depth.eq(dep), col].dropna().to_numpy() for dep in ["surface", "middle", "bottom"]]
            labels = ["Surface", "Middle", "Bottom"]
        ax.boxplot(arrays, labels=labels, showfliers=False, widths=0.55,
                   boxprops={"color": "#557a91"}, medianprops={"color": "#b0785c"},
                   whiskerprops={"color": "#557a91"}, capprops={"color": "#557a91"})
        ax.set_ylabel(label)
        if min(np.min(a) for a in arrays if len(a)) > 0:
            ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    path = OUT / "图2-4_水质指标分布_Secchi站点日期去重.png"
    fig.savefig(path, dpi=600, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    (OUT / "图2-4_重绘数据口径.txt").write_text(
        f"主表样本行数={len(data)}；Secchi站点—日期唯一记录={len(sec)}；旧图脚本按depth展示Secchi，V2按sample_id去重。\n", encoding="utf-8")
    return path


def replace_picture(paragraph, path: Path, width_cm: float):
    paragraph.clear()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Cm(width_cm))


def build():
    assert SOURCE.is_file() and not DOCX.exists(), "母本缺失或V2已存在；禁止覆盖"
    OUT.mkdir(parents=True, exist_ok=True)
    doc = Document(SOURCE)
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()

    # 第三章方法：文字仅来自已核实模型代码、环境manifest与冻结注册表。
    set_text(doc, "本节概述XGBoost", "XGBoost、TabM与TabPFN构成模型比较框架，其中XGBoost为十任务统一基准；TabM与TabPFN只参与四项代表任务的现代基线比较。TabICLv2承担本文主要预测模型与解释对象。XGBoost通过逐轮加入回归树拟合损失的梯度方向，并以树复杂度正则项控制模型容量[1]；该加法树结构对应附录D的TreeSHAP解释，不等同于TabICLv2的上下文预测机制。")
    set_text(doc, "本章以统一的站点划分", "本章在冻结站点划分与77项严格特征条件下，分别评价TabICLv2的预测表现和模型响应。SHAP用于报告特征贡献，ALE用于刻画有观测支持的一维局部响应。历史四任务五折解释与后续十任务第1折P1单查询解释采用不同预测协议，应分层报告。XGBoost受控复现模型的TreeSHAP与ALE作为独立树模型对照列于附录D；其TreeSHAP位于log1p尺度，而XGBoost ALE及TabICLv2解释位于原始目标尺度，不能直接比较绝对归因值。")
    set_text(doc, "XGBoost以加法方式", "TabICLv2是基于上下文学习的表格基础模型。预训练阶段使模型学习多种表格监督任务的统计规律；推理阶段由当前任务的带标签训练样本构成上下文，对未标记样本进行条件预测[13]。与逐任务独立拟合参数的传统监督模型不同，其任务适配主要发生在上下文输入中。本文保持77项特征及站点划分不变，将每项水质任务的训练折样本作为预测上下文，评价未见站点预测能力。")
    set_text(doc, "TabICLv2将带标签训练样本", "每项任务按照冻结manifest确定训练、测试obs_id。训练折先按固定77特征顺序组成矩阵并拟合中位数填补器；训练目标在非负条件下进行log1p变换，随后将训练特征和标签传入官方TabICL回归器。测试样本只提供特征，不提供目标值。解释运行中固定训练上下文，仅改变被解释的查询特征；因此解释的是给定上下文条件下的预测函数，而非脱离上下文的通用物理关系。")
    p = find(doc, "3.2.3 特征编码与集成推理")
    after(p, "本地适配器将输入转换为float32，并调用官方TabICLRegressor；环境登记记录默认8成员集成、none与power两种归一化视图及自动混合精度设置。解释脚本还核验特征顺序、训练上下文、填补器统计量与checkpoint哈希。上述配置说明实际运行条件，不推断预训练器内部未记录的编码细节。")
    set_text(doc, "本文固定每折训练样本", "推理固定随机种子42、预训练权重、每折训练样本顺序与77特征顺序；对应环境manifest记录8个集成成员及none、power视图。非负目标的预测先从log1p尺度经expm1回到原始水质单位，再按既有脚本执行非负截断。历史批查询与P1单查询条件预测函数分开记录，不用一次重复探针代替所有折的数值稳定性检验。")
    set_text(doc, "本文区分随机划分", "本文区分随机划分、年份留出和站点留出三种泛化问题。正式主结果采用冻结五折站点留出，同一站点在任一折只属于训练集或测试集的一侧；这使报告的OOF预测面向未见监测站点，而非同站点跨日期插值[14]。随机和年份方案仅在协议与数据版本可追溯时作为补充，不与正式站点留出混作配对试验。")
    set_text(doc, "正式实验配置、随机种子", "正式预测沿用各Family V2及Secchi冻结配置。解释运行使用官方TabICL回归器适配器、随机种子42、float32输入和预训练权重；日志记录模型文件哈希及训练上下文哈希。另行受控复现的20个XGBoost折模型在事前容差下逐样本预测达到20/20折数值等价，但它们不是找回的原始冻结二进制，仅供附录D的基准模型解释。")
    p = find(doc, "3.3.5 复现性控制与数据泄漏防范")
    after(p, "站点折由冻结station-fold map及逐样本manifest指定，先按obs_id取出训练和测试记录，再核验两侧站点集合不相交。每折中位数填补器仅在训练样本拟合，所得统计量原样用于测试样本；统一77特征在建模前冻结，解释候选变量只从该清单与训练样本覆盖中选择，不使用测试标签作筛选依据。对非负目标，目标变换参数和上下文标签只从训练折获得；模型预测经expm1逆变换及既定非负约束回到原始单位后，才与测试目标比较。随机种子、预训练权重哈希、特征顺序哈希、训练obs_id顺序哈希和上下文哈希共同约束复现；历史批查询与P1单查询的预测协议分别留痕。")
    set_text(doc, "分海区与来源级消融均读取", "分海区建模在留站验证下比较全域与区域模型，来源消融按冻结77特征分组逐次删除一个来源组；两者均沿用原实验的训练与评价口径。二者与TabICLv2解释图件不是同一模型上的同一种统计量，因此后文只讨论证据是否相互呼应，不把XGBoost消融差值表述为TabICLv2内部贡献。")
    set_text(doc, "决定系数R²衡量", "设测试样本真实值为yᵢ、预测值为ŷᵢ、测试均值为ȳ、样本数为n。R²=1−Σ(yᵢ−ŷᵢ)²/Σ(yᵢ−ȳ)²，RMSE=√[Σ(yᵢ−ŷᵢ)²/n]，MAE=Σ|yᵢ−ŷᵢ|/n。pooled OOF将各折互斥测试预测合并后计算一次指标；fold_R2_mean先逐折计算R²再求平均，因分母和样本量不同而不能互换。站点RMSE胜率为在站点级RMSE上TabICLv2低于XGBoost的站点数占比，MAE同理。配对bootstrap以站点为重采样单位，重复抽取站点时保留该站点全部观测及两模型配对预测，再计算模型间误差差值的2.5%与97.5%分位数；站点内时间相关性因整站保留而未被当作独立观测处理。")
    nse = find(doc, "3.4.2 Nash-Sutcliffe效率系数")
    nse._p.getparent().remove(nse._p)
    for old, new in [("3.4.3 pooled OOF与折间统计", "3.4.2 pooled OOF与折间统计"),
                     ("3.4.4 逐站点误差与改善比例", "3.4.3 逐站点误差与改善比例"),
                     ("3.4.5 站点重复抽样与稳定性评价", "3.4.4 站点聚类bootstrap与稳定性评价")]:
        set_text(doc, old, new)
    set_text(doc, "R²", "R²") if False else None
    # 不把R²以另一名称重复定义为空小节；公式段落与表3-2保留。

    # 结果分析与讨论：所有数值按V2 registry、站点级OOF及冻结消融/分区表核实。
    set_text(doc, "TabICLv2在10项任务上的pooled", "TabICLv2在十项任务的pooled R²均高于XGBoost，但增益分布不均。Chl-a表、中、底层分别提高0.132、0.106、0.136，Secchi提高0.067，TN表层提高0.143；这些任务的站点RMSE胜率为75.0%—88.1%，配对站点bootstrap的RMSE差值区间上限均低于零。TN中、底层及TP三水层虽有pooled R²增益，其RMSE区间上限接近或跨越零，故总体提升与跨站点稳健性应分开陈述。TP表层从−0.127升至0.247，但不能据此称为稳定通过。")
    set_text(doc, "TabICLv2的RMSE站点胜率", "十任务站点RMSE胜率为70.2%—88.1%，说明改善覆盖多数站点；但余下站点仍可能退化。站点级评价把总体误差改善拆解为空间上的受益与损失：例如Chl-a中层南部海区SM18站点126条记录的RMSE差值为+1.785，而TN表层维多利亚港VT11站点67条记录的差值为+0.174（正值表示TabICLv2误差更大）。二者不改变总体优势，却提示局地模型失配需要单独检查。")
    set_text(doc, "随机划分允许同一站点", "随机留出允许同站点不同时段记录分别进入训练和测试，回答监测网络内部插值问题；它与冻结站点留出的测试对象不同。项目中有历史随机实验文件，但尚未核实与十任务Family V2完全一致的特征、样本和模型协议，因此本节不把那些数值作为正式站点留出结果的配对比较。")
    set_text(doc, "年份留出要求模型迁移", "年份留出评价模型对未参与训练时段的外推，时间漂移与空间新站点是不同误差来源。现有主登记表未提供可与十任务站点留出逐项对齐的正式年份留出结果，故这里保留验证目标与适用边界，而不填入缺乏同口径核验的数字。")
    set_text(doc, "站点留出是本文正式主口径", "站点留出是本文正式主口径，十任务均有可追溯的冻结预测、pooled OOF指标和逐站点误差。它回答未见站点的预测问题；随机与年份留出若要作定量横向比较，还需统一样本、目标变换和模型配置，不能仅凭文件名相近拼接指标。")
    for old, new in [("4.3 随机、年份与站点留出验证比较", "4.3 不同验证口径与证据边界"),
                     ("4.3.1 随机划分结果", "4.3.1 随机划分的评价对象"),
                     ("4.3.2 年份留出结果", "4.3.2 年份留出的评价对象"),
                     ("4.3.3 站点留出结果", "4.3.3 正式站点留出结果")]:
        set_text(doc, old, new)
    after(find(doc, "4.4 空间泛化稳定性与高误差站点分析"), "逐站点OOF误差显示，Chl-a中层南部海区SM18站点（n=126）相对XGBoost的RMSE增加1.785；Chl-a表层Port Shelter海区PT3站点（n=65）增加0.818。营养盐任务中，维多利亚港VT11站点的TN表层（n=67）和TP表层（n=67）分别增加0.174 mg/L与0.072 mg/L。相比之下，TN表层南部海区SM19站点（n=120）RMSE下降1.011 mg/L。局地退化与改善均有实际记录，且水区分布并非单一集中；现有证据不足以把误差直接归因于岸距、污染源或某一环境特征。进一步联系空间特征分布时，应以站点坐标和原始环境值为依据，避免将经纬度代理变量当作机制解释。")
    after(find(doc, "5.6 不同任务环境依赖差异"), "不同任务对输入信息的使用并不相同。十任务P1单查询SHAP中，Chl-a表层较高的贡献来自3日太阳辐射总量、风速—降雨交互代理、3日海表温度及经度；Secchi的纬度、月份正弦、经度和年内日序位居前列。TN表层以经度与年份偏移为主，TP中层则以年份偏移、经纬度和月份编码居前。上述排名只在各自任务内成立，不能跨目标直接比较SHAP绝对值。")
    after(find(doc, "5.7 解释结果与高误差站点的对应关系"), "站点级OOF误差证实少数位置有退化，但现有解释产物主要是任务级或少量预登记样本的归因，不能逐站点地把某一SHAP特征与误差原因一一配对。南部海区SM18的Chl-a中层以及维多利亚港VT11的TN、TP表层均出现较大正RMSE差值；与此同时，南部海区SM19的TN表层差值为负，表明同一大海区内也存在相反表现。现阶段可靠的结论是模型增益具有站点异质性，而非已识别这些站点的具体环境成因。")
    after(find(doc, "5.8 环境解释、空间分区与泛化表现的综合讨论"), "预测精度、特征贡献和空间分区结果共同反映不同任务对环境信息的依赖差异。XGBoost来源消融中，删除空间静态特征使Chl-a表层和Secchi的pooled R²分别下降0.087和0.171；TabICLv2的SHAP也识别出空间坐标贡献，但两项分析属于不同模型，因而只能作为跨模型现象的对照。气象来源删除使Secchi、TN表层和TP中层的R²分别下降0.032、0.033和0.083。历史分海区五折验证中，Chl-a表层区域模型相对全域模型的pooled R²提高0.098，4/5折改善；这一补充结果与正式Family V2主比较不属于同一注册实验，不能替代十任务站点留出性能。TN与TP分区收益并不普遍，区域针对性与训练样本量存在权衡。由此，模型选择应综合总体精度、站点退化范围、来源组消融和区域样本覆盖，而不只看单一总体R²。")
    set_text(doc, "（4）TabICLv2四项代表任务", "（4）TabICLv2历史四项代表任务完成五折模型无关SHAP与一维ALE，后续十任务扩展完成第1折P1单查询解释；两者的预测协议与证据范围不同。P1重复探针仅覆盖九任务，历史部分折未通过严格逐样本复现检验。二维候选只完成联合支持审查，未计算ALE响应曲面。")
    set_text(doc, "本章综合来源消融", "本章将来源消融、分区验证与TabICLv2解释结果并列讨论：空间定位与气象信息的重要性具有任务差异，站点级误差进一步表明总体优势并不均匀分布。")

    # 引言综述增加研究问题与真实来源，不扩展到未检索的机制结论。
    after(find(doc, "1.2.1 近岸水质预测研究进展"), "近岸水质预测同时面临监测站点分布不均与环境信息异质性。已有海岸带水质机器学习研究将环境特征与时空验证结合，说明随机划分、留站验证和解释方法回答的问题并不相同[15]。本文据此把未见站点误差作为主要评价对象，而将解释结果限定为模型在给定训练数据和特征体系下的统计响应。")
    after(find(doc, "1.2.4 水质预测的空间泛化与时间外推"), "当同一站点的观测在训练与测试两侧重复出现时，随机验证可能高估面向新地点的外推表现。具有空间或时间依赖的数据需要与研究目标匹配的阻断验证策略[14]。本文以站点留出为正式主结果，年份留出与随机划分则用于界定不同预测场景，不把三类验证的分数视为可直接配对的同一指标。")

    # 表格压缩与防断词。
    task_labels = {"chla_surface": "Chl-a表层", "chla_middle": "Chl-a中层", "chla_bottom": "Chl-a底层", "secchi": "Secchi",
                   "tn_surface": "TN表层", "tn_middle": "TN中层", "tn_bottom": "TN底层", "tp_surface": "TP表层", "tp_middle": "TP中层", "tp_bottom": "TP底层"}
    for row in doc.tables[1].rows[1:]:
        raw = row.cells[0].text.strip()
        set_cell(row.cells[0], task_labels.get(raw, raw))
        ver = row.cells[5].text.strip()
        set_cell(row.cells[5], "C" if ver.startswith("CHLA") else "T" if ver.startswith("TN") else "P" if ver.startswith("TP") else "S")
    table_note_after(doc.tables[1], "注：C、T、P分别表示Chl-a、TN、TP Family V2统一站点五折划分；S表示Secchi冻结站点留出划分。")
    source_short = {"METEOROLOGY": "MET", "RAINFALL_HYDRO_TIDE": "RHT", "OCEAN_PHYSICS": "OCN", "HUMAN_ACTIVITY": "HUM", "TEMPORAL_PROCESS": "TMP", "SPATIAL_STATIC": "SPA"}
    class_cn = {"IMPORTANT": "重要", "SUPPORTIVE": "支持", "NEUTRAL": "中性", "MIXED": "混合"}
    for row in doc.tables[8].rows[1:]:
        set_cell(row.cells[0], task_labels.get(row.cells[0].text.strip(), row.cells[0].text.strip()))
        set_cell(row.cells[1], source_short.get(row.cells[1].text.strip(), row.cells[1].text.strip()))
        set_cell(row.cells[5], class_cn.get(row.cells[5].text.strip(), row.cells[5].text.strip()))
    table_note_after(doc.tables[8], "注：MET=气象，RHT=降雨/水文/潮汐，OCN=海洋物理，HUM=人类活动代理，TMP=时间过程，SPA=空间静态。ΔR²为删除来源组后相对完整模型的变化；负值表示删除后性能下降。")
    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True
        for row in table.rows:
            for cell in row.cells:
                tcpr = cell._tc.get_or_add_tcPr()
                if tcpr.find(qn("w:noWrap")) is None:
                    tcpr.append(OxmlElement("w:noWrap"))

    # 图2/图4按文中出现顺序编号；图2-4修复Secchi重复粒度。
    for old, new in [("图2-4 多源环境数据接入", "图2-3 多源环境数据接入"),
                     ("图2-3 四项水质指标", "图2-4 四项水质指标"),
                     ("图4-4 随机、年份", "图4-3 不同验证口径"),
                     ("图4-3 四个代表任务", "图4-4 四个代表任务")]:
        set_text(doc, old, find(doc, old).text.replace(old.split()[0], new.split()[0], 1).replace("随机、年份与站点验证下的模型差异", "不同验证口径下的模型比较范围"))
    corrected = make_corrected_distribution()
    distribution_caption = find(doc, "图2-4 四项水质指标")
    caption_index = next(i for i, item in enumerate(doc.paragraphs) if item._p is distribution_caption._p)
    replace_picture(doc.paragraphs[caption_index - 1], corrected, 15.2)
    distribution_caption.text = "图2-4 叶绿素a、透明度、总氮与总磷的观测分布"
    after(distribution_caption, "注：Chl-a、TN和TP按三水层绘制；透明度按唯一站点—日期记录绘制一次，不重复计入水层。箱线图不显示离群点，纵轴为对数刻度。")

    # 参考文献只追加经出版源验证的条目；修复[3]作者拼写。
    set_text(doc, "[3] MÜLLER", "[3] MÜLLER S, HOLLMANN N, PINEDA S, GRABOCKA J, HUTTER F. Transformers can do Bayesian inference[C]//International Conference on Learning Representations. 2022. https://openreview.net/forum?id=KSugKcbNf9.")
    ref_anchor = find(doc, "致 谢")
    for ref in [
        "[13] QU J, HOLZMÜLLER D, VAROQUAUX G, LE MORVAN M. TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model[C]//Proceedings of the 43rd International Conference on Machine Learning. PMLR, 2026, 306: 102477-102555. https://proceedings.mlr.press/v306/qu26a.html.",
        "[14] ROBERTS D R, BAHN V, CIUTI S, et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure[J]. Ecography, 2017, 40(8): 913-929. DOI: 10.1111/ecog.02881.",
        "[15] GRBČIĆ L, DRUŽETA S, MAUŠA G, et al. Coastal water quality prediction based on machine learning with feature interpretation and spatio-temporal analysis[J]. Environmental Modelling & Software, 2022, 155: 105458. DOI: 10.1016/j.envsoft.2022.105458.",
    ]:
        before(ref_anchor, ref)
    # 模板待核时不擅改结论章节编号和简历次序，明确留待作者确认。
    doc.save(DOCX)
    (OUT / "母本与数值来源.txt").write_text(
        f"唯一编辑母本：{SOURCE}\nSHA256：{source_hash}\n正式性能：runs/final_experiment_closure_audit_v1/final_core_results_table_v2.csv\n站点误差：runs/{{chla,tn,tp}}_family_v2/09_*_station_metrics.csv\n", encoding="utf-8")
    print("V2_DRAFT_SAVED", DOCX)


if __name__ == "__main__":
    build()
