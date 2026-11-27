"""只读取冻结回传审查结果，形成正式解释资格矩阵。"""
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
AUDIT = PROJECT / "结果/诊断结果/修复77特征SHAP回传核验_20261120_v2"

shap = pd.read_csv(AUDIT / "02_30核心SHAP核验.csv")
batch = pd.read_csv(AUDIT / "03_十任务预测复现与P1审查.csv")
ale = pd.read_csv(AUDIT / "04_20条ALE核验.csv")
rows = []
for _, b in batch.iterrows():
    subset = shap.loc[shap.task.eq(b.task)]
    rows.append({
        "任务": b.task,
        "核心样本数": len(subset),
        "SHAP端点与独立P1超1e-6数": int((subset["SHAP减审查P1"].abs() > 1e-6).sum()),
        "SHAP端点最大差": float(subset["SHAP减审查P1"].abs().max()),
        "整批严格复现状态": b["严格批量复现状态"],
        "整批最大误差": float(b["最大绝对误差"]),
        "ALE曲线数": int(ale.task.eq(b.task).sum()),
        "seed43已回传": "否",
        "正式论文SHAP资格": "未通过预测函数一致性门槛",
        "正式论文ALE资格": "未通过预测函数一致性门槛",
    })
result = pd.DataFrame(rows)
assert len(result) == 10 and result["核心样本数"].sum() == 30
assert result["ALE曲线数"].sum() == 20
result.to_csv(OUT / "十任务正式解释资格矩阵.csv", index=False, encoding="utf-8-sig")
print(f"10任务、30核心样本、20条ALE；超阈值样本={result['SHAP端点与独立P1超1e-6数'].sum()}")
