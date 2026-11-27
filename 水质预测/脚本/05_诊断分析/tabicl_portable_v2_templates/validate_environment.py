"""检查目标机软件/GPU能力；不构造模型、不执行预测。"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from portable_common import BUNDLE_ROOT, load_config, verify_static_manifest


def main() -> int:
    cfg = load_config()
    issues = []
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "shap", "torch", "tabicl", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            issues.append(f"missing package: {name}")
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        cuda_runtime = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        vram_bytes = torch.cuda.get_device_properties(0).total_memory if cuda_available else None
        cudnn = torch.backends.cudnn.version()
    except Exception as exc:
        cuda_available, cuda_runtime, gpu_name, vram_bytes, cudnn = False, None, None, None, None
        issues.append(f"torch/CUDA检查异常：{exc!r}")
    if sys.version_info[:2] != (3, 11):
        issues.append(f"需要Python 3.11，当前为{platform.python_version()}")
    if packages.get("torch") != "2.6.0+cu124":
        issues.append(f"Torch版本需为2.6.0+cu124，当前为{packages.get('torch')}")
    if not cuda_available:
        issues.append("CUDA GPU不可用；禁止自动CPU回退")
    if gpu_name and "4060" not in gpu_name:
        issues.append(f"目标设备预期RTX 4060，当前检测到{gpu_name}；请人工确认")
    if vram_bytes and vram_bytes < 7 * 1024**3:
        issues.append(f"GPU显存低于约定的8GB级设备：{vram_bytes / 1024**3:.2f} GiB")
    issues.extend(verify_static_manifest())
    nvidia = None
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if p.returncode == 0:
            nvidia = p.stdout.strip()
    except Exception:
        pass
    report = {
        "status": "ENVIRONMENT_READY_FOR_CROSS_DEVICE_CHECK" if not issues else "ENVIRONMENT_VALIDATION_FAILED",
        "issues": issues, "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "cuda_runtime": cuda_runtime, "cudnn": cudnn,
        "gpu_name": gpu_name, "gpu_total_vram_bytes": int(vram_bytes) if vram_bytes else None,
        "nvidia_smi": nvidia, "ram_cache_mode": cfg.get("RAM_CACHE_MODE"),
        "max_process_ram_gb": cfg.get("max_process_ram_gb"), "bundle_root": str(BUNDLE_ROOT),
        "no_model_prediction_called": True,
    }
    out = BUNDLE_ROOT / "logs/environment_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())

