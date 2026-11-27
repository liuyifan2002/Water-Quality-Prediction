"""Read-only localhost progress page for the WQ77 explanation run."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from collections import deque
import csv
from datetime import datetime
import gzip
import json
import sqlite3
import statistics
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CONFIG = json.loads((ROOT / "04_explanation_config" / "portable_config.json").read_text(encoding="utf-8"))
TASKS = CONFIG["tasks"]
CURVES = CONFIG["ale_curves"]
OLD_RUNTIME = ROOT.parent / "tabicl_explanation_portable_bundle_v2" / "workspace" / "runs" / "tabicl_v2_final_explanation_complete_v1" / "shap_runtime.csv"
GPU_HISTORY_FILE = OUT / "monitor_gpu_history.jsonl"
GPU_HISTORY = deque(maxlen=241)
GPU_LOCK = threading.Lock()
GPU_LATEST = None


def load_gpu_history():
    try:
        with GPU_HISTORY_FILE.open(encoding="utf-8") as file:
            for line in file:
                try:
                    GPU_HISTORY.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass


def gpu_sampler():
    global GPU_LATEST
    while True:
        reading = gpu()
        if reading:
            item = {"time": time.time(), "utilization_pct": reading["utilization"],
                    "memory_pct": round(100 * reading["memory_used_mb"] / reading["memory_total_mb"], 2)}
            with GPU_LOCK:
                GPU_LATEST = reading
                GPU_HISTORY.append(item)
            OUT.mkdir(exist_ok=True)
            with GPU_HISTORY_FILE.open("a", encoding="utf-8") as file:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
        time.sleep(15)


def gpu_snapshot():
    with GPU_LOCK:
        return GPU_LATEST, list(GPU_HISTORY)


def estimate_remaining(stage):
    old = []
    try:
        with OLD_RUNTIME.open(encoding="utf-8-sig", newline="") as file:
            old = list(csv.DictReader(file))
    except OSError:
        pass
    completed = []
    for task in TASKS:
        for row in sample_rows(task):
            path = OUT / "shap" / task / f"{row['obs_id']}_seed42.csv.gz"
            if path.is_file():
                try:
                    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as file:
                        completed.append(next(csv.DictReader(file)))
                except (OSError, StopIteration, ValueError):
                    pass
    if completed:
        sample_seconds = statistics.median(float(r["elapsed_seconds"]) for r in completed)
        per_call = statistics.median(float(r["model_seconds"]) / float(r["actual_model_calls"]) for r in completed if float(r["actual_model_calls"]) > 0)
        basis = f"本机已完成 {len(completed)} 个 SHAP 样本；ALE 按单次预测耗时推算"
    elif old:
        sample_seconds = statistics.median(float(r["elapsed_seconds"]) for r in old)
        per_call = statistics.median(float(r["actual_predict_seconds"]) / float(r["actual_model_predict_calls"]) for r in old)
        basis = f"旧环境 {len(old)} 个同协议 SHAP 样本；尚待本机首样本校准"
    else:
        return None
    remaining_queries = 0
    for curve in CURVES:
        if (OUT / "ale_1d" / f"{curve['task']}_{curve['feature']}_curve.csv").exists():
            continue
        path = ROOT / "04_explanation_config" / "ale" / f"{curve['task']}_{curve['feature']}_query_manifest.csv"
        with path.open(encoding="utf-8-sig", newline="") as file:
            remaining_queries += sum(1 for _ in csv.DictReader(file))
    if (OUT / "monitor_run.json").is_file():
        run = read_json(OUT / "monitor_run.json", {})
        if run.get("status") == "RUNNING" and run.get("command") == "Secchi边界登记v2续跑":
            database = OUT / "checkpoints" / "ale_1d_local_effects.sqlite"
            if database.is_file():
                try:
                    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=1) as conn:
                        done = conn.execute("SELECT count(*) FROM local_effect WHERE task='secchi' AND feature='ssrd_sum_3d'").fetchone()[0]
                    remaining_queries = max(0, remaining_queries - done)
                    basis += f"；Secchi已完成 {done}/328 条查询"
                except sqlite3.Error:
                    pass
    seconds = max(0, (30 - len(completed)) * sample_seconds + 2 * remaining_queries * per_call)
    if stage == "全部阶段已结束":
        seconds = 0
    return {"seconds": round(seconds), "low_seconds": round(seconds * 0.6), "high_seconds": round(seconds * 1.6), "basis": basis, "sample_seconds": round(sample_seconds), "ale_prediction_seconds": round(per_call, 2)}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def log_tail(path, limit=8):
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def gpu():
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=3)
        name, util, used, total = [v.strip() for v in raw.splitlines()[0].split(",")]
        return {"name": name, "utilization": int(util), "memory_used_mb": int(used), "memory_total_mb": int(total)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def progress():
    run = read_json(OUT / "monitor_run.json", {})
    stages = read_json(OUT / "run_status.json", [])
    stage_names = ["run_prediction_audit.py", "run_shap.py", "run_ale.py", "run_quality_audit.py"]
    completed_stages = {v.get("script") for v in stages}
    stage = next((v for v in stage_names if v not in completed_stages), "全部阶段已结束")
    audits = sum((OUT / "audit" / f"{task}.json").is_file() for task in TASKS)
    shap = sum((OUT / "shap" / task / f"{row['obs_id']}_seed42.csv.gz").is_file()
               for task in TASKS for row in sample_rows(task))
    ale = sum((OUT / "ale_1d" / f"{v['task']}_{v['feature']}_curve.csv").is_file() for v in CURVES)
    quality = any(v.get("script") == "run_quality_audit.py" and v.get("exit_code") == 0 for v in stages)
    failures = []
    reproduction_mismatch = [task for task in TASKS if (read_json(OUT / "audit" / f"{task}.json", {}) or {}).get("batch_reproduction") == "NUMERICAL_REPRODUCTION_FAILED"]
    if reproduction_mismatch:
        failures.append({"unit": "预测复现审查", "status": "NUMERICAL_REPRODUCTION_FAILED", "error": f"{len(reproduction_mismatch)}/10 个任务与冻结的整批预测有数值差异；P1 重复结果另见审查文件。"})
    for path in sorted((OUT / "status").glob("*.json")) if (OUT / "status").is_dir() else []:
        if run.get("status") == "RUNNING" and path.stem.startswith("ale_") and run.get("started_at"):
            if path.stat().st_mtime < datetime.fromisoformat(run["started_at"]).timestamp():
                continue
        item = read_json(path, {})
        if item.get("status") in ("FAILED", "BLOCKED"):
            failures.append({"unit": path.stem, "status": item["status"], "error": item.get("error", "")})
    log = OUT / "logs" / stage
    lines = log_tail(log.with_name(log.name + ".log"), 6) if stage != "全部阶段已结束" else log_tail(OUT / "monitor_launcher.log", 6)
    if not lines and stage == "run_shap.py":
        lines = ["首个 SHAP 核心样本正在计算；每个样本完成后写入结果并更新计数。"]
    completed = audits + shap + ale + int(quality)
    gpu_now, gpu_history = gpu_snapshot()
    return {"run": run, "stage": stage, "stages": stages, "counts": {"audit": audits, "shap": shap, "ale": ale, "quality": int(quality), "completed": completed, "total": 61},
            "failures": failures, "log": lines, "gpu": gpu_now, "gpu_history": gpu_history, "eta": estimate_remaining(stage), "server_time": time.time()}


def sample_rows(task):
    path = ROOT / "04_explanation_config" / "shap" / "core_samples.csv"
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [r for r in csv.DictReader(file) if r.get("task") == task and r.get("phase") == "CORE_3"]


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>水质模型解释计算进度</title>
<style>:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#172638;background:#f2f6fa}*{box-sizing:border-box}body{margin:0;padding:28px}.wrap{max-width:1000px;margin:auto}.card{background:#fff;border:1px solid #dce5ed;border-radius:16px;padding:22px;margin:14px 0;box-shadow:0 4px 16px #1730440a}h1{margin:0 0 5px;font-size:26px}h2{margin:0 0 16px;font-size:18px}.muted{color:#607385}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.metric{background:#f4f8fb;border-radius:12px;padding:15px}.value{font-size:25px;font-weight:750}.label{font-size:13px;color:#607385}.bar{height:14px;border-radius:20px;background:#e5edf4;overflow:hidden;margin-top:18px}.fill{height:100%;background:#24876e;width:0;transition:width .5s}.stage{font-size:19px;font-weight:700;margin:15px 0}.pill{display:inline-block;border-radius:30px;padding:5px 11px;background:#e8f5f0;color:#176d55;font-weight:700}.warn{background:#fff3df;color:#825000;padding:12px 14px;border-radius:10px;margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f8fb;padding:14px;border-radius:10px;font-size:12px;max-height:220px;overflow:auto}.issue{border-left:4px solid #d78d21;padding:8px 12px;margin:8px 0;background:#fff9ef;overflow-wrap:anywhere}footer{color:#708195;font-size:12px;margin:18px 0}</style>
<div class="wrap"><div class="card"><h1>水质模型解释计算进度</h1><div class="muted">WQ77 修复版 · TabICLv2 · 本机实时状态 · <span id="clock">—</span></div></div><div class="card"><span class="pill" id="run">读取中</span><div class="stage" id="stage">—</div><div class="grid"><div class="metric"><div class="value" id="total">—</div><div class="label">已完成 / 计划单元</div></div><div class="metric"><div class="value" id="audit">—</div><div class="label">预测审查 / 10</div></div><div class="metric"><div class="value" id="shap">—</div><div class="label">SHAP 核心样本 / 30</div></div><div class="metric"><div class="value" id="ale">—</div><div class="label">ALE 曲线 / 20</div></div><div class="metric"><div class="value" id="quality">—</div><div class="label">质量汇总 / 1</div></div></div><div class="bar"><div class="fill" id="fill"></div></div><p class="muted" id="elapsed">—</p></div><div class="card"><h2>设备与异常</h2><div id="gpu" class="muted">—</div><div id="issues"></div></div><div class="card"><h2>当前阶段日志</h2><pre id="log">等待日志…</pre></div><footer>每 3 秒刷新。进度按已落盘结果计数；“已计算”与最终质量通过分别核对。预登记的 Secchi ssrd_sum_3d 曲线可能因两条样本越过原分箱边界而阻塞。</footer></div>
<script>const $=x=>document.getElementById(x);const names={'run_prediction_audit.py':'预测复现与 P1 审查','run_shap.py':'SHAP 核心样本计算','run_ale.py':'ALE 曲线计算','run_quality_audit.py':'结果质量审查'};async function tick(){try{const p=await(await fetch('/progress?t='+Date.now(),{cache:'no-store'})).json();const c=p.counts,r=p.run||{};$('clock').textContent=new Date().toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});$('run').textContent=r.status==='RUNNING'?'运行中':r.status==='FINISHED'?'全部完成':r.status==='COMPLETED_WITH_ISSUES'?'已结束，存在未完成项':'尚未启动';$('stage').textContent=names[p.stage]||p.stage;$('total').textContent=c.completed+' / '+c.total;$('audit').textContent=c.audit;$('shap').textContent=c.shap;$('ale').textContent=c.ale;$('quality').textContent=c.quality;$('fill').style.width=(c.completed/c.total*100)+'%';$('elapsed').textContent=r.started_at?'启动时间：'+new Date(r.started_at).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false})+' · 已运行 '+Math.floor((Date.now()-new Date(r.started_at))/60000)+' 分钟':'等待启动';$('gpu').textContent=p.gpu?`${p.gpu.name} · 使用率 ${p.gpu.utilization}% · 显存 ${p.gpu.memory_used_mb}/${p.gpu.memory_total_mb} MB`:'设备状态暂不可用';$('issues').replaceChildren();p.failures.forEach(x=>{let d=document.createElement('div');d.className='issue';d.textContent=x.unit+' · '+x.status+' · '+x.error;$('issues').append(d)});$('log').textContent=p.log.length?p.log.join('\\n'):'当前阶段尚无日志';}catch(e){$('stage').textContent='进度读取中断：'+e.message}}tick();setInterval(tick,3000)</script></html>'''

PAGE = PAGE.replace('<div class="bar">', '<div class="metric" style="margin-top:14px"><div class="value" id="eta">估算中…</div><div class="label">预计剩余时间 · <span id="finish">—</span></div><div class="label" id="eta_basis">—</div></div><div class="bar">')
PAGE = PAGE.replace('</html>', '''<script>
function duration(sec){let days=Math.floor(sec/86400),hours=Math.round((sec%86400)/3600);return days+'天 '+hours+'小时'}
async function etaTick(){try{let p=await(await fetch('/progress?t='+Date.now(),{cache:'no-store'})).json(),e=p.eta;if(!e)return;document.getElementById('eta').textContent='约 '+duration(e.seconds)+'（粗估 '+duration(e.low_seconds)+'–'+duration(e.high_seconds)+'）';document.getElementById('finish').textContent='预计完成：'+new Date(Date.now()+e.seconds*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});document.getElementById('eta_basis').textContent=e.basis;}catch(e){document.getElementById('eta').textContent='估算暂不可用'}}etaTick();setInterval(etaTick,15000)
</script></html>''')
PAGE = PAGE.replace('<div class="card"><h2>当前阶段日志</h2>', '''<div class="card"><h2>GPU 与显存使用率</h2><div class="muted" style="margin-bottom:12px"><span style="color:#147d70;font-weight:700">● GPU 使用率</span>　<span style="color:#da8125;font-weight:700">● 显存占用率</span>　<span id="chartLatest">等待首次采样…</span></div><canvas id="gpuChart" style="display:block;width:100%;height:250px"></canvas><div class="muted" style="font-size:12px">最近 1 小时 · 每 15 秒采样 · 纵轴 0–100%</div></div><div class="card"><h2>当前阶段日志</h2>''')
PAGE = PAGE.replace('</html>', '''<script>
let chartPoints=[];
function drawGpuChart(){const canvas=document.getElementById('gpuChart'),width=canvas.clientWidth,height=250,scale=window.devicePixelRatio||1;canvas.width=Math.round(width*scale);canvas.height=Math.round(height*scale);const ctx=canvas.getContext('2d');ctx.setTransform(scale,0,0,scale,0,0);ctx.clearRect(0,0,width,height);const left=44,right=width-12,top=10,bottom=height-28;ctx.font='12px system-ui';ctx.fillStyle='#718096';ctx.strokeStyle='#e4ebf0';for(const v of [0,25,50,75,100]){const y=bottom-(bottom-top)*v/100;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillText(v+'%',3,y+4)}const now=Date.now()/1000,start=now-3600;ctx.fillText('60 分钟前',left,bottom+19);ctx.fillText('现在',Math.max(left,right-27),bottom+19);const points=chartPoints.filter(p=>p.time>=start);for(const [field,color] of [['utilization_pct','#147d70'],['memory_pct','#da8125']]){ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.beginPath();points.forEach((p,i)=>{const x=left+(right-left)*(p.time-start)/3600,y=bottom-(bottom-top)*Math.max(0,Math.min(100,p[field]))/100;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke();if(points.length===1){const p=points[0],x=left+(right-left)*(p.time-start)/3600,y=bottom-(bottom-top)*p[field]/100;ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill()}}}
async function chartTick(){try{const p=await(await fetch('/progress?t='+Date.now(),{cache:'no-store'})).json();chartPoints=p.gpu_history||[];const last=chartPoints.at(-1);document.getElementById('chartLatest').textContent=last?`GPU ${last.utilization_pct}% · 显存 ${last.memory_pct}%`:'等待首次采样…';drawGpuChart()}catch(e){document.getElementById('chartLatest').textContent='采样暂不可用'}}chartTick();setInterval(chartTick,15000);window.addEventListener('resize',drawGpuChart)
</script></html>''')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/progress":
            body = json.dumps(progress(), ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif path == "/":
            body = PAGE.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    load_gpu_history()
    threading.Thread(target=gpu_sampler, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", 8768), Handler).serve_forever()
