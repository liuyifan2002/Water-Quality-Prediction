"""校验checkpoint文件hash及序列化结构；不初始化预测模型。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from portable_common import BUNDLE_ROOT, load_config, resolve, sha256, verify_static_manifest


def main() -> int:
    cfg = load_config()
    p = resolve(cfg["model_dir"] + "/tabicl-regressor-v2-20260212.ckpt")
    expected = "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
    errors = verify_static_manifest()
    actual = sha256(p) if p.is_file() else None
    if actual != expected:
        errors.append(f"checkpoint SHA256不匹配：{actual}")
    structure = None
    try:
        import torch
        obj = torch.load(p, map_location="cpu", weights_only=True)
        if not isinstance(obj, dict) or "config" not in obj or "state_dict" not in obj:
            errors.append("checkpoint缺少TabICL要求的config/state_dict")
        else:
            structure = {"top_level_keys": sorted(obj.keys()), "state_tensor_count": len(obj["state_dict"]),
                         "config_keys": sorted(obj["config"].keys())}
    except Exception as exc:
        errors.append(f"checkpoint解析失败：{exc!r}")
    report = {"status": "CHECKPOINT_VALID" if not errors else "CHECKPOINT_INVALID", "errors": errors,
              "path_relative": p.relative_to(BUNDLE_ROOT).as_posix(), "sha256": actual,
              "expected_sha256": expected, "structure": structure, "model_prediction_called": False}
    out = BUNDLE_ROOT / "logs/checkpoint_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

