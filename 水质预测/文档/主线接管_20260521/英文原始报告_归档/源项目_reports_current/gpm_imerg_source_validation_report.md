# GPM IMERG source validation 报告

本轮只做 source validation，不构建正式 runoff 特征，不进入 A/B。

## 核心状态
- Earthdata 凭据登录：`成功`
- 登录状态：`login_ok`
- CMR collection 发现：`True`
- granule 发现：`True`
- 样本 granule 下载状态：`blocked`
- 下载备注：`earthdata_eula_not_accepted`

## 数据源判断
- 数据机构：NASA GPM / GES DISC
- 产品：GPM IMERG Final Daily (`GPM_3IMERGDF` V07)
- 是否免费：是
- 是否需要账号：是，免费 Earthdata 账号
- 时间分辨率：daily
- 空间分辨率：0.1 degree
- 2014-2024 覆盖：是
- 香港研究区覆盖：是
- 是否适合 lag features：是

## 本轮结论
- 登录和产品发现都通过了，但样本下载被 GES DISC 数据条款接受步骤拦住。
- 这说明数据源本身可用，但在自动下载前，需要账户完成一次性的 EULA/terms 接受。
- 在完成该步骤前，建议把 CHIRPS 保留为无需账号的备选公开源。
