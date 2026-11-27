"""只读显示十任务精简解释实验的实时落盘进度。"""

from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tabicl_v2_slim_explainability_v1"
TASKS = (
    "chla_surface", "chla_middle", "chla_bottom", "secchi",
    "tn_surface", "tn_middle", "tn_bottom", "tp_surface", "tp_middle", "tp_bottom",
)


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def shap_record(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            data = list(csv.DictReader(handle))
        if len(data) != 77 or len({row["feature"] for row in data}) != 77:
            return None
        if any(row.get("status") != "COMPUTED_PASS" for row in data):
            return None
        return data[0]
    except (OSError, ValueError, KeyError, UnicodeError):
        return None


def main() -> None:
    manifest = rows(OUT / "01_preregistered_shap_samples.csv")
    if not manifest:
        raise SystemExit(f"找不到样本清单：{OUT}")
    samples_by_task = {task: [row for row in manifest if row["task"] == task] for task in TASKS}
    exact_samples: list[tuple[str, str, dict[str, str], datetime]] = []
    pass_tasks: list[str] = []
    blocked_tasks: list[str] = []
    task_rows = []
    for task in TASKS:
        gate = rows(OUT / "gates" / f"{task}_p1_decision.csv")
        passed = bool(gate and gate[0].get("status") == "P1_GATE_PASS")
        if passed:
            pass_tasks.append(task)
        else:
            blocked_tasks.append(task)
        core_done = optional_done = 0
        for sample in samples_by_task[task]:
            path = OUT / "shap" / task / f"{sample['obs_id']}_seed42.csv.gz"
            record = shap_record(path)
            if record is None:
                continue
            exact_samples.append((task, sample["obs_id"], record,
                                  datetime.fromtimestamp(path.stat().st_mtime)))
            if sample["phase"] == "CORE_3":
                core_done += 1
            else:
                optional_done += 1
        sensitivity = sum(shap_record(path) is not None for path in
                          (OUT / "shap" / task).glob("*_seed43.csv.gz")) if passed else 0
        curves = list((OUT / "ale_1d").glob(f"{task}_*_curve.csv")) if passed else []
        valid_curves = 0
        for curve in curves:
            result = rows(curve)
            if result and result[0].get("status") in (
                "COMPUTED_PASS", "COMPUTED_WITH_NUMERICAL_LIMITATION"
            ):
                valid_curves += 1
        task_rows.append((task, "通过" if passed else "阻断", core_done, optional_done,
                          sensitivity, valid_curves))
    print("TabICLv2 精简解释实验实时进度（只读；以已落盘且结构合格的文件为准）")
    print(f"检查时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"任务资格：通过 {len(pass_tasks)}/10；阻断 {len(blocked_tasks)}/10")
    print(f"核心SHAP：{sum(x[2] for x in task_rows)}/{3 * len(pass_tasks)} 个合格任务预登记样本")
    print(f"可选SHAP：{sum(x[3] for x in task_rows)} 个；种子敏感性：{sum(x[4] for x in task_rows)} 个")
    print(f"一维ALE：{sum(x[5] for x in task_rows)} 条已落盘曲线；每任务最多2条，须先通过覆盖审查")
    print("\n任务                    资格   核心SHAP  可选SHAP  敏感性  一维ALE")
    for task, gate, core, optional, sensitivity, ale in task_rows:
        print(f"{task:<23} {gate:<4}   {core}/3        {optional}/2        {sensitivity}         {ale}/2")
    if exact_samples:
        task, obs_id, record, when = max(exact_samples, key=lambda item: item[3])
        print(f"\n最近完成：{task} / {obs_id}，{when:%Y-%m-%d %H:%M:%S}")
        print(f"这一样本真实模型调用：{record.get('actual_model_calls', '未记录')} 次；"
              f"用时：{float(record.get('elapsed_seconds', 0)) / 3600:.2f} 小时")
        print(f"已完成预登记SHAP真实调用合计："
              f"{sum(int(float(item[2].get('actual_model_calls', 0))) for item in exact_samples)} 次")
    if pass_tasks:
        completed_core = [float(record.get("elapsed_seconds", 0)) for task, obs, record, _ in exact_samples
                          if task in pass_tasks and any(row["obs_id"] == obs and row["phase"] == "CORE_3"
                                                      for row in samples_by_task[task])]
        if completed_core:
            remaining = 3 * len(pass_tasks) - sum(x[2] for x in task_rows)
            mean_hours = sum(completed_core) / len(completed_core) / 3600
            print(f"核心SHAP粗估剩余：{remaining * mean_hours:.1f} GPU小时"
                  "（仅按已完成样本速度；不含可选SHAP、ALE和其他任务差异）")
    logs = [path for path in (OUT / "logs").glob("*.log") if path.is_file()]
    if logs:
        newest = max(logs, key=lambda path: path.stat().st_mtime)
        print(f"最新单元日志：{newest}（{datetime.fromtimestamp(newest.stat().st_mtime):%H:%M:%S}）")
    print("说明：进程是否仍在运行由外层PowerShell显示；已有文件数不包含正在内存中计算的样本。")


if __name__ == "__main__":
    main()
