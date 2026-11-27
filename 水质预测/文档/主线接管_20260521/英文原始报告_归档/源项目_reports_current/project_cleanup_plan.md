# 项目整理计划

更新时间：2026-05-21 16:05:32

Project Organizer Agent 默认不删除、不移动正式文件。本计划汇总可执行与需人工确认事项。

## 文件分类概览

- temporary_file: 14913
- raw_or_external_data: 2069
- current_canonical: 187
- historical_experiment: 170
- other: 131
- script: 66
- credential_or_auth_related: 16
- docs: 3

## 风险级别概览

- medium: 15214
- high: 2085
- low: 256

## 建议动作

- 根目录污染候选：0 个，详见 `root_pollution_report.md`。
- 重复输出候选组：9 组，详见 `duplicate_outputs_report.md`。
- 缓存目录候选：0 个，详见 `cache_cleanup_plan.md`。
- 历史实验结果保留为归档，不直接删除。
- 原始数据、外部下载数据、凭证和认证文件保持只读保护。
