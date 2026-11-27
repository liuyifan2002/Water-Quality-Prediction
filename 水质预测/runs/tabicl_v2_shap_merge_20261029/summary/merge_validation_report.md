# TabICLv2 SHAP 回传与本地合并校验报告

- 外部ZIP SHA256：`c23e418fd86943763c6986ee1ccd6d83b07c54582e45b94548ed9d9e0da0be59`，与侧车文件一致；ZIP CRC、路径安全、内部28项SHA256均通过。
- 预登记样本清单：回传与本地SHA256一致；冻结77特征及顺序、fold、obs_id、seed、checkpoint、imputer、train/test顺序哈希、加和误差均通过。
- 原始本地SHAP：19份；回传SHAP：28份；重名：8份；回传独有：20份。
- 合并SHAP：39份，覆盖10任务；CORE_3 30/30，SENSITIVITY_1 9/10。
- 本地ALE 1D已成曲线：9份；全部`ale_1d`文件已作为快照复制，未将不完整的支持文件视为曲线。
- 重名8份的上下文和query哈希一致，但文件SHA256不同；保留本地为主版本，回传副本在`remote_overlap_variants/shap/`，差异详见`summary/overlap_comparison.csv`。跨GPU输出不声明位级一致。
- 本合并仅整合解释结果；不运行模型，不修改冻结预测、split、原始SHAP或ALE，也不改本地运行注册表和看板状态。
- `feature_importance_by_task.csv`仅反映每任务已预登记的3–4个样本，不是全测试集全局重要性，也不能替代P1质量门槛。

## 未完成与注意事项

- TP底层的SENSITIVITY_1仍缺1份；ALE 1D和ALE 2D未完成全部预登记范围。
- `chla_middle_wind_speed_sq_curve.csv`原状态为`COMPUTED_WITH_NUMERICAL_LIMITATION`，本合并原样保留；不解释为稳定通过。
- 本机`progress_status.json`仍显示旧`RUNNING`时间戳；合并未据此推断正在运行，亦未自动重启实验。

补充：本地 ALE 2D 的 3 份候选/支持文件也已复制，但没有已完成曲线；8 份重名 SHAP 的样本内前十特征集合均相同。
