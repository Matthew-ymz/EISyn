# HCP Yeo7 30 被试 Phi 分解扩展设计

## 目标

将 HCP Schaefer-500 Yeo7-PC1 history-source \(\Phi^{EID}\) 的模块级贪婪分解，从固定的 5 名被试扩展为数据目录中发现的全部 30 名被试，并保持与已完成的全体被试 Phi/null 分析相同的建模与 null 协议。

## 固定分析协议

- 每名被试独立在前 900 个时间点上拟合 Yeo7 PC1 与八阶 \(\Delta\)-Ridge（\(\alpha=10\)）。
- 每个 Yeo7 网络的全部 8 个历史 PC1 滞后值作为一个不可拆模块。
- 对 observed 与每名被试 20 个独立非零 circular-shift null 都重拟合模型并运行 greedy top-3 分解。
- 跨被试核频率使用同一 null replicate 编号组成 matched cohort，计算 observed top-3 频率相对 null 频率的经验检验。

## 实现与产物

1. 分解脚本自动发现数据根目录下的全部有效被试；保留 `--subjects` 以支持显式子集复现。
2. 为自动发现行为添加测试，确保默认路径选中 30 名 HCP 被试而不改变显式子集行为。
3. 重跑 30 名被试的 observed 和 20-null 分解，写入既有结果目录中的 JSON、Markdown 报告和热图（PNG/SVG/PDF）。
4. 更新 `docs/reports/brain.md`，使 HCP 模块分解的范围、表格、图注和解释边界与新结果一致。

## 验证

- 单元测试验证默认被试发现与显式 `--subjects` 覆盖。
- 运行脚本并检查 `summary.json` 中为 30 条被试记录、每条含 20 个 null greedy 结果。
- 渲染检查热图：被试行完整、标注可读、色条和任何图例不遮挡数据。
- 报告中所有人数、null 样本数、p 值分辨率和图文件引用与产物一致。

## 非目标

- 不增加 null 数量，不调整 PCA、Ridge、分解算法或 atom 定义。
- 不将 greedy atom 解释为唯一或经多重比较校正的生物学结构。
