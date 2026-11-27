"""恢复TabICLv2解释实验；跳过已有检查点并继续未完成阶段。"""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUNNER=ROOT/"脚本/05_诊断分析/79_run_all_tabicl_explanation_experiments.py"
PATCH=ROOT/"runs/tabicl_v2_deterministic_repair_v1/isolated_patch"
os.environ.update({"PYTHONHASHSEED":"42","CUBLAS_WORKSPACE_CONFIG":":4096:8","TABICL_FIXED_NORM_ORDER":"none,power"})
os.environ["PYTHONPATH"]=str(PATCH)+os.pathsep+os.environ.get("PYTHONPATH","")
os.execv(sys.executable,[sys.executable,str(RUNNER),"--run-all"])
