# 已知限制（V2）

- V2 family 结果是 Chl-a、TN、TP 的当前正式空间验证结果；历史 pilot 仅作重复验证记录。
- Secchi 为单一任务，保持 pooled OOF R²=0.602049、fold R²均值=0.584848。
- Random、Year 与 Station 回答不同泛化问题，不能相互替代。历史 representative summary 的 station R²为 fold mean，已明确标注，不冒充 pooled OOF。
- modern baseline 只覆盖四个代表任务。
- 多源消融的冻结77特征不含真实遥感变量，不能据此判断遥感贡献。
- 分海区结果属于历史扩展实验，其 split 和 feature set 与 family V2 不同，不作为V2核心数值替代。
