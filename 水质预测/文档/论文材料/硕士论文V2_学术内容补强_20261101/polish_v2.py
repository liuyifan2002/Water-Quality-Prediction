"""V2末轮学术与版式修整；只读冻结来源，另存校样。"""
from pathlib import Path
import csv
import re

import pandas as pd
from docx import Document
from docx.shared import Cm, Pt

ROOT = Path(r"E:\研究生\codex\水质预测")
OUT = ROOT / "文档/论文材料/硕士论文V2_学术内容补强_20261101"
SOURCE = OUT / "水质预测硕士学位论文_学术内容补强V2_图目录校正版.docx"
TARGET = OUT / "水质预测硕士学位论文_学术内容补强V2_校样.docx"


def find(doc, prefix, style=None):
    a = [p for p in doc.paragraphs if p.text.startswith(prefix) and (style is None or p.style.name == style) and "\t" not in p.text]
    if len(a) != 1: raise ValueError((prefix, style, len(a)))
    return a[0]


def after(paragraph, text):
    p = paragraph.insert_paragraph_before(text)
    paragraph._p.addnext(p._p)
    return p


def station_evidence():
    manifest = pd.read_csv(ROOT / "runs/tabiclv2_pilot/01_strict_feature_manifest.csv")
    features = manifest.loc[manifest.included_in_pilot.eq(True), "feature"].tolist()
    assert len(features) == 77
    main = pd.read_csv(ROOT / "数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv",
                       usecols=["station", "sample_id", "depth", "lat", "lon"] + features)
    main = main.loc[main.depth.eq("surface")].drop_duplicates("sample_id")
    station = main.groupby("station").agg(n_station_dates=("sample_id", "nunique"), lat=("lat", "median"), lon=("lon", "median"))
    station["missing_77_share"] = main.groupby("station")[features].apply(lambda x: float(x.isna().mean().mean()))
    station["lat_percentile"] = station.lat.rank(pct=True)
    rows = []
    for family, task, sid in [("chla", "chla_middle", "SM18"), ("chla", "chla_surface", "PT3"),
                              ("tn", "tn_surface", "VT11"), ("tp", "tp_surface", "VT11"),
                              ("tn", "tn_surface", "SM19")]:
        d = pd.read_csv(ROOT / f"runs/{family}_family_v2/09_{family}_family_v2_station_metrics.csv")
        r = d.loc[d.task.eq(task) & d.station_id.eq(sid)].iloc[0]
        s = station.loc[sid]
        rows.append({"task": task, "station_id": sid, "n_oof": int(r.n_obs), "XGB_RMSE": float(r.XGB_RMSE),
                     "TabICL_RMSE": float(r.TabICL_RMSE), "delta_RMSE": float(r.delta_RMSE),
                     "station_date_n_in_main_surface": int(s.n_station_dates), "missing_77_share_in_main_surface": float(s.missing_77_share),
                     "lat": float(s.lat), "lon": float(s.lon), "latitude_station_percentile": float(s.lat_percentile)})
    pd.DataFrame(rows).to_csv(OUT / "站点误差与特征覆盖补充核对.csv", index=False, encoding="utf-8-sig")
    return station


def main():
    assert SOURCE.is_file() and not TARGET.exists()
    station = station_evidence()
    doc = Document(SOURCE)

    # 3.4仅保留实质内容，不把所有公式置于最后一个空标题下面。
    for prefix in ["3.4.1 决定系数", "3.4.2 pooled OOF", "3.4.3 逐站点误差", "3.4.4 站点聚类bootstrap"]:
        p = find(doc, prefix, "Heading 3")
        p._p.getparent().remove(p._p)
    old = find(doc, "设测试样本真实值为yᵢ")
    old._p.getparent().remove(old._p)
    anchor = find(doc, "表3-2 主要评价指标", "表题")
    anchor.insert_paragraph_before("设测试样本真实值为yᵢ、预测值为ŷᵢ、测试均值为ȳ、样本数为n。决定系数R²=1−Σ(yᵢ−ŷᵢ)²/Σ(yᵢ−ȳ)²；均方根误差RMSE=√[Σ(yᵢ−ŷᵢ)²/n]；平均绝对误差MAE=Σ|yᵢ−ŷᵢ|/n。R²越高、误差指标越低表示同任务内预测更好。本文不另设与R²定义相同而未独立用于正式结果的NSE小节。")
    anchor.insert_paragraph_before("pooled OOF指标将各折互斥测试预测合并后计算一次，fold_R2_mean则先逐折计算R²再求平均。两者分母、样本权重不同，不相互替代。逐站点RMSE胜率定义为TabICLv2站点RMSE低于XGBoost的站点占比；MAE胜率同理，使用同一任务、同一站点集合上的配对比较。")
    anchor.insert_paragraph_before("不确定性描述采用冻结脚本中的1000次站点聚类bootstrap：每次有放回抽取站点，并连同该站全部观测及两模型的成对预测一起重采样，再计算ΔRMSE与ΔMAE分布的2.5%和97.5%分位数。该做法尊重同站点观测相关性，但区间仍只概括既定站点总体内的重采样波动，不应外推为独立海域的泛化保证。")

    # 十任务SHAP正文按两组各解释一次，增强图件与论证衔接。
    c3 = find(doc, "图3-3 TabICLv2十项任务", "图题")
    after(c3, "前五项任务的P1解释呈现明显差异：Chl-a三水层均有三日太阳辐射总量居前，但表层还显示风速—降雨代理与海表温度贡献；Secchi以纬度、月份正弦和经度靠前，TN表层则以经度与年份偏移靠前。这里的平均绝对SHAP表示各任务当前查询样本中的模型归因，不应把Chl-a与营养盐任务的原始数值放在同一尺度比较。")
    c4 = find(doc, "图3-4 TabICLv2十项任务", "图题")
    after(c4, "TN中、底层与TP三水层的前列特征更集中于经纬度、年份编码及少量气象或人类活动代理。TP表层和中层的平均绝对SHAP量级较小，与目标自身量纲和模型输出尺度有关；其排序只说明同任务内部相对重要性。三核心样本的覆盖范围有限，因此图3-3、图3-4作为第1折P1证据，不能代替历史五折的折间一致性检验。")

    # 历史验证图与正式Family V2区分，避免图面数值被误读为十任务配对结果。
    cap = find(doc, "图4-3 不同验证口径", "图题")
    after(cap, "注：本图保留历史代表任务的不同划分实验，协议与十任务Family V2及Secchi正式站点留出并不完全一致；图中差值用于说明验证口径的差异，不纳入正式十任务配对提升判定。")

    find(doc, "逐站点OOF误差显示").text = (
        "逐站点OOF误差显示局地退化与总体改善并存：Chl-a中层南部海区SM18站点（n=126）的RMSE由XGBoost的2.462增至TabICLv2的4.247 µg/L；Chl-a表层Port Shelter海区PT3站点（n=65）由4.143增至4.961 µg/L。营养盐任务中，维多利亚港VT11站点的TN表层RMSE由0.345增至0.519 mg/L，TP表层由0.062增至0.134 mg/L，均为67条OOF记录。相对地，TN表层南部海区SM19站点（n=120）的RMSE由1.197降至0.186 mg/L。各站点的样本量和海区均不相同，不能用单一总体平均掩盖这些方向相反的变化。"
    )
    find(doc, "站点级OOF误差证实").text = (
        "站点误差与原始特征覆盖存在值得进一步检验的对应。VT11表层主表仅71个站点—日期记录，77项严格特征的平均缺失比例为62.7%，明显高于94站点中位数8.8%；该站TN、TP表层的OOF RMSE均退化。SM18的纬度位于站点分布约第2百分位，Chl-a中层RMSE退化，但纬度相近的SM19在TN表层却明显改善。这说明稀疏输入和空间边缘可作为诊断线索，不能被直接定为误差成因；任务级SHAP也不足以替代该站点的局部归因检验。"
    )

    # 全文结构与模型主线一致。
    structure = find(doc, "全文共六章。")
    structure.text = "全文共六章。第1章提出近岸水质预测的空间泛化问题；第2章介绍研究区、监测数据和77项严格特征；第3章说明TabICLv2及基准模型、冻结站点划分和TabICLv2解释方法，XGBoost解释对照置于附录D；第4章分析十任务预测精度与站点异质性；第5章讨论区域建模、来源消融与任务环境依赖；第6章总结主要结论、限制和后续工作。"
    find(doc, "结论与展望", "Heading 1").text = "第6章 结论与展望"
    for old, new in [("主要研究结论", "6.1 主要研究结论"), ("研究创新与实际意义", "6.2 研究创新与实际意义"),
                     ("研究不足与展望", "6.3 研究不足与展望")]:
        find(doc, old, "Heading 2").text = new

    # 新增表3-3优化列宽，避免任务名和来源缩写拥挤断行。
    tables = [t for t in doc.tables if len(t.rows) > 0 and "主要贡献变量" in t.rows[0].cells[1].text]
    assert len(tables) == 1
    t = tables[0]
    widths = [Cm(2.6), Cm(7.5), Cm(2.3), Cm(2.6)]
    t.autofit = False
    for col, width in zip(t.columns, widths): col.width = width
    for row in t.rows:
        row.cells[3].text = row.cells[3].text.replace("P1；n=3", "P1/3样本")
        for cell, width in zip(row.cells, widths):
            cell.width = width
            for p in cell.paragraphs:
                for run in p.runs: run.font.size = Pt(7.8)

    doc.save(TARGET)
    print("V2_POLISHED", TARGET, "VT11 missing", round(float(station.loc["VT11", "missing_77_share"]), 4))


if __name__ == "__main__": main()
