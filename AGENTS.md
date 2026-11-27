# AGENTS.md — Project Operating Contract

**Repository:** `E:\研究生\codex` (git branch `full-station`)
**Project:** Hong Kong nearshore water-quality prediction (香港近岸水质预测) — MSc thesis research pipeline
**Audience:** every AI agent (Claude Code, Codex, agy, Hermes) and human collaborator working in this repo.
**Language:** this contract is written in English on purpose. Deliverables that are *meant to be Chinese* (thesis text, 中文 README sections, 研究报告) stay Chinese — see §11.

Read this file **before** touching anything. It is the distilled result of the Codex sessions from 2026-05-21 through 2026-11-26 plus the in-repo documentation under `水质预测/文档/`.

---

## 1. What this repo is

A reproducible research pipeline that fuses harbour water-quality observations with remote sensing (Sentinel-2 / Landsat / VIIRS), ERA5 meteorology, rainfall, tide and night-light features to predict four indicators across three depth layers:

| Target | Unit | Layers |
|---|---|---|
| Chl-a (`chla_ug_l`) | µg/L | Surface / Middle / Bottom |
| Secchi (`secchi_m`) | m | Surface only |
| TN (`tn_mg_l`) | mg/L | Surface / Middle / Bottom |
| TP (`tp_mg_l`) | mg/L | Surface / Middle / Bottom |

That is **10 prediction tasks**. Scope: 94 monitoring stations, 2014–2024, Hong Kong nearshore waters.

Two things live in this repo and must not be confused:

```
E:\研究生\codex\                  ← git repo root (branch full-station)
├─ AGENTS.md                      ← this file (repo-wide contract)
├─ README.md
├─ .gitignore
└─ 水质预测\                      ← the actual project; all real work happens here
   ├─ AGENTS.md                   ← older, Chinese-only directory rules (still valid, narrower scope)
   ├─ README.md
   ├─ 配置\ 数据\ 脚本\ 结果\ 文档\
   ├─ runs\                       ← per-experiment workspaces (registered experiments)
   ├─ 交付\                       ← portable bundles handed to third parties
   ├─ 论文修订交付_20261121\       ← versioned thesis revisions (V4 → V6.4)
   ├─ 归档_历史版本\ 待确认_未归类\
   └─ codex\worktrees\            ← local worktree artifacts
```

Side worktrees that exist on this machine and may be stale — check `git worktree list` before trusting them:
`E:\研究生\codex_chla_optimization` (branch `chla-optimization`), and detached Codex worktrees under `C:\Users\刘逸凡\.codex\worktrees\{0d10,71dc,9e62,b9b1,9ff7}`.

---

## 2. Canonical data (frozen — never silently rebuild)

| Object | Path | Status |
|---|---|---|
| Official feature version | `WQ77_REPAIRED_20261115_V1` | **frozen**, 30,719 records × 77 strict features, 94 stations, 10 water zones |
| Model table (mainline) | `数据/处理后数据/正式主表/model_table_3depth_2014_2024.csv` | canonical input for mainline training |
| Water-quality master | `数据/处理后数据/正式主表/water_quality_master_2014_2024.csv` | canonical |
| Satellite features | `数据/处理后数据/正式主表/station_date_satellite_features_2014_2024.csv` | canonical |
| Candidate validation & recompute | `结果/扩展实验/候选数据验收与一致性重算_v1_20261115/` | source of the frozen 332 task-fold re-runs |
| Retired | the pre-repair 77-feature experiments and every derivative figure | **out of the thesis**; do not resurrect |

Rules:

- `WQ77_REPAIRED_20261115_V1` is the **only** official 77-feature version for the thesis. Do not substitute older tables, and do not "improve" it in place.
- Any new data candidate goes to `数据/处理后数据/衍生版本/<name>_<date>/` and is admitted to the formal track only after an explicit controlled A/B.
- Never write credentials, tokens, cookies, `.cdsapirc`, or account material into the repo. (`.cdsapirc` currently sits in `待确认_未归类/` — leave it out of git.)

---

## 3. Evaluation protocols — three distinct ones, never mix them

| Protocol | Where used | Meaning |
|---|---|---|
| **Random 80/20, `random_state=42`** | formal mainline scripts (`脚本/04_模型训练/05_train_full_station_models.py` etc.) | held-out sample performance under the same distribution. Documented in `文档/正式主链运行说明.md`. |
| **5-fold Station Holdout** | formal thesis evaluation of the current 10-task results | stations are grouped so a station never appears in both train and test. This is the protocol behind the paper's "正式十任务" tables. |
| **Random Strict** | independent experiments (`RANDOM_10TASK_V1`, `RANDOM_TABULAR_MODEL_ZOO_V2`, `TABICLV2_*_ADAPTATION_V1`) | random split kept fixed across all tasks/models. **Independent experiment only — never presented as the paper's formal protocol.** |

**Wording law (thesis-critical):** a random held-out split may be called `随机留出测试` / `随机比例留出` — **never** `严格时间外推`, `未来年份预测`, or `时间外推能力`. The archived strict time-extrapolation outputs (`归档_历史版本/过期输出/严格时间外推_20260516/`) are historical sensitivity artifacts only.

**Known protocol caveat:** Random Strict lets the same sampling event appear in train and test across depth layers. Whenever Random Strict numbers are quoted, keep them visibly separated from Station Holdout numbers (`runs/vpr_tab_stage_a/` report does this correctly — copy that pattern).

---

## 4. Pipeline architecture

```
配置 → 数据准备 → 特征构建 → 建模总表 → 模型训练 → 诊断 → 结果汇总/绘图 → 论文材料
```

Numbered mainline scripts (run in order; see `文档/正式主链运行说明.md` for the authoritative list):

| Step | Script |
|---|---|
| 1 | `脚本/01_数据准备/01_prepare_station_samples.py` |
| 2 | `脚本/01_数据准备/02_build_water_quality_master.py` |
| 3 | `脚本/02_特征构建/03_merge_satellite_features.py` |
| 4 | `脚本/02_特征构建/04_build_model_table.py` |
| 5 | `脚本/04_模型训练/05_train_full_station_models.py` |
| 6 | `脚本/06_结果汇总与绘图/06_summarize_full_station_results.py` |
| 7+ | `07_plot_*`, `08_shap_*`, `09_diagnose_chla_surface`, `11..15_build_thesis_*` |

Module boundaries (enforced — see §6):

- `脚本/01_数据准备/` — observation/station sample preparation
- `脚本/02_特征构建/` — remote-sensing merge, ERA5, rainfall, tide, night-light, feature tables
- `脚本/03_GEE遥感导出/` — Earth Engine JavaScript exporters
- `脚本/04_模型训练/` — mainline training **and** extension-experiment training
- `脚本/05_诊断分析/` — peaks, residuals, autocorrelation, coverage
- `脚本/06_结果汇总与绘图/` — summary tables, thesis figures, SHAP, PPT assets
- `脚本/历史版本/` — retired scripts, including the 2026-05-21 trial-project snapshot
- `runs/<experiment_name>/` — one directory per registered experiment; results, configs, reports, OOF predictions live here
- `交付/<bundle_name>/` — portable, self-contained handover bundles (with hash manifests)

Result tiers:

- `结果/正式结果/full_station/` — mainline outputs **only**
- `结果/扩展实验/<实验名>_<日期>/` — every experiment
- `结果/诊断结果/` — diagnostics
- `结果/论文图表/` — thesis-grade figures
- `结果/PPT材料/` — presentation assets

Configuration: `配置/项目路径配置.json` (all directory keys), `配置/正式主链配置.json` (scope, targets, canonical inputs/outputs), `配置/历史_v0配置.json`.

---

## 5. Technology stack & environments

Base dependencies (`requirements.txt`): `pandas`, `scikit-learn`, `xgboost`, `cdsapi`, `xarray`, `netCDF4`.

Model families actually in use: XGBoost (baseline everywhere), TabICLv2 (tabular foundation model), LightGBM, HistGradientBoosting, ExtraTrees, RandomForest, TabM, FT-Transformer (multi-task, M0/M1/M2), GPBoost (uncertainty), MMoE (multi-gate experts).

**Environment isolation is mandatory.** Heavy or conflicting families get their own venv; never install them into the mainline interpreter:

| Family | Environment | Requirement file |
|---|---|---|
| GPBoost | `.venv-gpboost` | `配置/GPBoost_扩展实验requirements.txt` |
| MMoE | `.venv-mmoe` (to be created; MMoE **must not** share the GPBoost env) | `配置/MMoE_独立环境requirements.txt` |
| TabICLv2 | dedicated env, historically `runs/tabiclv2_pilot/环境/.venv` (≈5 GB) | — |

Never copy a venv into a deliverable or a portable bundle: the TabICLv2 pilot venv was ~5.03 GB and is reconstructed from the requirement file, not shipped.

---

## 6. Development conventions (hard rules)

Before writing any code, answer these five questions in the reply — then code:

1. Which module directory does this belong to?
2. Is it mainline, extension experiment, diagnostics, or thesis output?
3. Does it need a new script? If so, what is the exact name?
4. Does it affect existing mainline results (`结果/正式结果/full_station/`)?
5. Which documents must be updated?

Then:

- **One script = one stage.** Never mix data prep, feature building, training, plotting and thesis output in a single file.
- **Naming:** `<NN>_<english_description>.py`, e.g. `05_train_full_station_models.py`. Banned: `test.py`, `new.py`, `final.py`, `修改版.py`.
- **No new files at `脚本/` root.** Everything goes into a numbered module directory.
- **Experiments:** output goes to `runs/<实验名>/` or `结果/扩展实验/<实验名>_<日期>/`. Temporary or failed runs never enter `结果/正式结果/`.
- **Thesis revisions:** each round gets its own directory under `论文修订交付_20261121/` (`V6_2_图表修复与全文一致性修订/`, `V6_4_七图定点修订_20261122/`, …). **Never overwrite a previous version's DOCX/PDF.**
- **Archive, don't delete:** retired scripts → `脚本/历史版本/`; expired outputs → `归档_历史版本/过期输出/`; unclear ownership → `待确认_未归类/`.
- **Git:** use `git mv` for moves; commit messages must state the change class (结构整理 / 路径修复 / 文档更新 / 扩展实验 / 正式主链修订) and separate mainline from experiment impact.
- **Doc update is part of the change:** every structural edit updates `文档/文件迁移记录.md` or `文档/后续重构计划.md`.
- **Do not change algorithm, training parameters, feature logic, or plot style** unless the user explicitly asks.
- **Do not re-path silently.** Moving files to make output paths resolve differently is forbidden.

Verification after any change:

1. Paths still resolve.
2. Mainline script inputs/outputs unchanged (or the change is declared).
3. Python syntax check passes.
4. Key CSVs: row count, column count, field names checked.
5. If the mainline was touched: the formal results are still reproducible.

---

## 7. Pitfalls recovered from history (do not relearn these)

**Encoding / locale**

- Scripts and logs must use **ASCII paths and English log messages**. A Chinese-path + Chinese-log script hit repeated encoding failures (2026-07-03); the fix that stuck was a clean rewrite rather than patching the old file. Keep new tooling ASCII-only; Chinese belongs in the *content*, not in the plumbing.
- Windows console output can mangle Chinese (`GBK`/`cp936`). Read files as UTF-8 explicitly; never rely on the console's default codepage.

**Numerical reproducibility of explanations**

- TabICLv2 SHAP/ALE is **not bit-reproducible across process rebuilds**: in the 2026-11-20 audit, 18/30 samples differed from independent recomputation by more than `1e-6` (max ≈ `0.0143`), even though same-process reruns were identical. Additivity was fine (max error `3.55e-15`), completeness was fine (2310 values). **Consequence:** never claim "reproduced" until a cross-process fixed-point check passes; the explanation chapter is still not submission-ready.
- A suspected root cause is unstable preprocessing-view ordering. It is a lead, not a proven cause.

**Experiment design traps**

- Vertical residual enhancement (`VPR_TAB_STAGE_B_V1`) **failed**: B1 did not beat the frozen B0 on any of the four branches, and B2 stayed below B0. Do not extend it to all ten tasks.
- Model-Zoo V2 OOF predictions were produced on the **old** master table, not on `WQ77_REPAIRED`. They may serve as exploratory leads only; per-task verdicts cap at **CONDITIONAL**.
- Pooled R² hides fold-level collapse: TP-middle fold 4 dropped from 0.419 to −0.289 (stations VM12, VT11) while the other four folds improved. Always report both pooled and per-fold metrics.
- Cross-protocol numbers are not comparable (`Secchi`: 0.701 random vs 0.602 station-holdout vs 0.627 new station-holdout). Always state the protocol next to the number.

**Data-source dead ends (verified, don't retry blindly)**

- **SCSORA / HelloSea** (`doi:10.12378/geodb.2026.2.005.V1`): detail page opens, but file listing requires a login token — guests get 0 files, no direct URLs. Blocked without an account.
- **HYCOM GOFS 3.1**: NCSS single-day requests time out; the **OPeNDAP/THREDDS** path works (`GLBy0.08/expt_93.0/{ts3z,uv3z}/<year>`). Use OPeNDAP subsetting.
- **GPM IMERG**: blocked by Earthdata/EULA registration — do not merge into the mainline before the download chain exists.
- **Bathymetry (GEBCO/GLOBathy)**: mechanism plausible, but no stable deployable input was ever produced.

---

## 8. Negative results — the do-not-repeat list

These were proven not to promote. Repeating them wastes compute and muddies the record:

1. Rewriting the whole mainline around `NASA OceanColor` as the primary source (coverage OK, unified A/B gave no stable promote).
2. Endless "high-res patch" stacking with high missingness and manifest-only patches.
3. Re-running weak tide proxy / weak coastal distance / weak zone refinement as single-source boosts.
4. Blind bathymetry retries (GEBCO, GLOBathy).
5. Forcing GPM IMERG into the mainline before the download chain is solved.
6. Extending vertical-residual enhancement to all ten tasks (failed in controlled test).

Partial-signal sources worth **controlled** A/B only: `Copernicus physics` (strongest first candidate), `enhanced_tide_proxy`, `station depth`, `distance_to_mouth + zone refinement`. Recommended combined candidate for TN/TP/Secchi: `distance_to_mouth + zone_refinement + station_depth + enhanced_tide_proxy`.

Documented mainline current-best scorecard (2026-05-21 handover, protocol as stated in that report): TN surface ≈ 0.521 is the closest task to R² 0.60; Chl-a mainline values were 0.176 / 0.200 / 0.127 (S/M/B), which the trial project beat at 0.330 / 0.365 / 0.344.

---

## 9. Where the current work stands (as of 2026-11-26)

**Results**

- `RANDOM_TABULAR_MODEL_ZOO_V2`: locked via inner 5-fold OOF only, then evaluated once on the fixed Random Strict test set — **10/10 tasks R² ≥ 0.5**, minimum Chl-a middle 0.503.
- `ADAPTIVE_ENSEMBLE_V2`: 10/10 tasks ≥ 0.5, but **not** an improvement over V1 (mean ΔR² −0.00056: 3 better, 2 worse, 5 unchanged).
- `TABICLV2_WATER_QUALITY_ADAPTATION_V1`: 9/10 tasks ≥ 0.5 on Random Strict; only 2/10 tasks single-model first.
- `MECHANISM_SCREEN_V1`: strongest remaining lead is Chl-a-middle dynamic-expert routing (TabM vs TabICLv2, +0.013 from equal-weight fusion); high-value underestimation suggests distribution-aware regression for Chl-a middle/surface and TP middle.

**Thesis**

Latest deliverable: `论文修订交付_20261121/V6_4_七图定点修订_20261122/水质预测硕士学位论文_V6.4_图件修订.docx`. Version history V4 → V5 → V5.1 → V6 → V6.1 → V6.2 (×2) → V6.3 (×2) → V6.4, each in its own folder.

**Open TODOs**

1. **Explanation chapter is not submission-ready** — TabICLv2 SHAP/ALE cross-process stability must be settled (fixed-point reproduction on a small sample first; do not recompute everything).
2. `EPD-Vertical V2` (`runs/epd_vertical_feature_ab_v2_20261126/`, candidate features `epd_vertical_candidate_features_2014_2024.csv`, built by `脚本/02_特征构建/17_build_epd_vertical_in_situ_candidate.py`) — controlled A/B of in-situ vertical T/S/DO/turbidity is the live experiment.
3. Thesis-wide number/figure consistency pass after the V6.4 figure set stabilizes.
4. Refactor backlog in `文档/后续重构计划.md`: path/config extraction, splitting `05_train_full_station_models.py` and `04_build_model_table.py`, standardizing result writers, triaging `待确认_未归类/`.
5. Data gaps that block R² ≥ 0.60: exact sampling datetime, current/discharge/water-level, bottom temperature/profile, a real freshwater-influence/plume quantification.

---

## 10. Definition of done for an agent task

- Deliverable exists at the stated path (absolute path in the report), not merely described.
- Numbers in any report are copied from files that exist; no invented metrics.
- Protocol named for every metric quoted.
- Nothing frozen was overwritten; new version directories were created instead.
- Docs updated (`文件迁移记录.md` / `后续重构计划.md`) when structure changed.
- Verification steps above executed and their results shown.

---

## 11. Multi-agent language rules

- **Prompts to sub-agents (Claude Code, agy, Codex) are written in English**, with code, paths, identifiers and error text preserved verbatim.
- **Chinese deliverables are generated directly in Chinese** — never drafted in English and machine-translated. Outer instructions stay English plus the explicit constraint: *"Output the final article/copy/comments directly in natural, idiomatic Simplified Chinese without translating from English."*
- **Reports back to the user are Simplified Chinese**, keeping technical terms in English (`Diff`, `API`, `MCP`, `git worktree`, `OOF`, `R²`).
- Thesis wording constraints in §3 are language-independent and override style preferences.

---

## 12. Provenance

Distilled from: `水质预测/AGENTS.md`, `README.md`, `文档/{项目架构说明,目录结构规范,正式主链运行说明,后续重构计划,文件迁移记录}.md`, `文档/主线接管_20260521/`, and Codex session rollouts `01a1da4d` (2026-11-14), `01a11cfc` (2026-10-09), `01a1fe34` / `01a1fe35` (2026-11-21).

When an in-repo document contradicts this file, the in-repo document wins for its own scope — and this file should be corrected.
