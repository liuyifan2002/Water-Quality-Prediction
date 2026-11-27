"""逐字复用79号脚本背景选择函数；仅迁移端按原规则执行。"""
import numpy as np
import pandas as pd
def representatives(frame: pd.DataFrame, x: np.ndarray, n: int, seed: int) -> np.ndarray:
    """固定站点-年份覆盖后，以77维环境空间max-min补足；不读取目标标签。"""
    meta = frame[["station_id", "date"]].copy()
    meta["year"] = pd.to_datetime(meta.date).dt.year
    rng = np.random.default_rng(seed)
    candidates = [int(rng.choice(g.index.to_numpy())) for _, g in meta.groupby(["station_id", "year"], sort=True)]
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        candidates = list(range(len(frame)))
    scale = np.nanstd(x, axis=0); scale[scale == 0] = 1
    z = (x - np.nanmean(x, axis=0)) / scale
    selected = [candidates[int(rng.integers(len(candidates)))]]
    pool = candidates + [i for i in range(len(frame)) if i not in set(candidates)]
    while len(selected) < min(n, len(frame)):
        remain = np.asarray([i for i in pool if i not in selected], dtype=int)
        dist = np.min(((z[remain, None, :] - z[np.asarray(selected)[None, :], :]) ** 2).mean(axis=2), axis=1)
        selected.append(int(remain[np.argmax(dist)]))
    return np.asarray(selected, dtype=int)
