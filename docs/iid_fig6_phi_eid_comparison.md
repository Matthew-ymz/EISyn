# IID Fig. 6 与 whole-system Phi^EID 对照

![Whole-system PhiEID comparison](../fig/iid_fig6_phi_eid_comparison/whole_system_phi_eid_phase_comparison.png)

本文档给出 Mediano et al. (2025) Fig. 6 的近似复现与 PEID 对照。当前结果使用 Lausanne-33 count 派生的 83 区近似矩阵，不使用论文中未随数据包公开的 HCP Lausanne-83 精确结构连接矩阵。

## 实验设计

1. 扫描全局耦合强度 `G`，记录每个 `G` 下的平均 firing rate。这个曲线用于说明系统从低 firing-rate regime 进入高 firing-rate regime，但不再用单个最大斜率点作为可靠相变标签，因为相邻几个 `G` 的离散斜率很接近：
  - `G=1.8`: `d rate / dG ≈ 13.06`
  - `G=1.9`: `d rate / dG ≈ 13.03`
  - `G=2`: `d rate / dG ≈ 11.87`
2. 在相同 `G` 扫描上计算两类信息指标：`Phi^R` 使用经验 lagged distribution，whole-system `Phi^EID` 使用标准化最大熵源干预下的线性 Gaussian transition。
3. 为测试 `Phi^R` 对采样分布的敏感性，额外构造三条 pilot 曲线：全部时间点、靠近中位 activity 的 middle-state rows、远离中位 activity 的 tail-biased rows。每条曲线各自寻找 `G>1.0` 范围内的最大值位置。
4. `G=1.0` 是扫描边界点，主图绘制和峰值识别对 `Phi^R` 与 `Phi^EID` 都统一使用 `G>1.0`；原始数值保留在缓存中用于审计。

## 指标口径

- `Phi^R`：在滞后样本的经验 Gaussian 分布上计算 pairwise whole-minus-sum 与 double-redundancy 修正，再对脑区 pair 求平均。因此它会随经验采样窗口、重采样权重和状态分布改变。
- `Phi^EID`：先拟合全脑线性 Gaussian transition `X_(t+tau)=A X_t+eps`，再在独立标准化最大熵源干预下计算 whole-system `I_do(X_t;X_(t+tau))-sum_i I_do(X_t^i;X_(t+tau))`。该量等价于源侧条件 total correlation，因此数值非负。

## 当前结果

- 平均 firing rate 在 `G≈1.7-1.9` 附近进入快速上升区，但最大斜率点本身不够稳定，因此图中不再画相变虚线。
- `Phi^R` 的主曲线峰值位置为 `G ≈ 1.8`，落在 firing-rate 快速上升区附近。
- 对 `Phi^R` 使用不同采样分布时，识别到的峰值位置会改变：
  - `Full-pair cache`: `G* ≈ 1.8`
  - `Uniform pilot`: `G* ≈ 1.6`
  - `Middle-state rows`: `G* ≈ 1.7`
  - `Tail-biased rows`: `G* ≈ 1.8`
- whole-system `Phi^EID` 的峰值位置为 `G ≈ 1.7`，最小值为 `4.927`，保持非负。

## 结论

这张图要表达的不是 `Phi^R` 完全无效，而是它的临界点判断依赖经验状态分布。主 `Phi^R` 曲线在本次近似复现中确实可以标出 `G≈1.8` 附近的相变；但当 source sampling distribution 改变时，`Phi^R` 的峰值会移动到 `G≈1.6` 或 `G≈1.7`。相比之下，whole-system `Phi^EID` 使用统一的最大熵干预分布，并且保持非负，更适合作为机制口径下的稳定相变对照指标。

## G = 1.0 的高 Phi 值如何解释

`G=1.0` 处的 whole-system `Phi^EID` 或部分 `Phi^R` 采样曲线偏高不应解释为物理相变。它没有对应 firing-rate 曲线的最大斜率，也不对应论文 Fig. 6 中的临界区。更合理的解释是估计口径造成的边界/瞬态效应：低耦合、低 firing-rate regime 中的 lagged dynamics 更接近自保持和低噪声线性预测，线性 Gaussian `EI` 会因为残差协方差较小而升高。这个点说明估计器对边界 regime 敏感，不是相变证据。因此主图绘制和峰值识别对 `Phi^R` 与 `Phi^EID` 均使用 `G>1.0`，原始 `G=1.0` 数值保留在缓存中用于审计。

## 文献与边界

- Zotero item `26Q48H8Y`：Mediano et al. (2025), *Toward a unified taxonomy of information dynamics via Integrated Information Decomposition*。
- Zotero item `MYATYWAJ`：Yang, Wang, and Zhang (2026), *Partial Effective Information Decomposition for Synergistic Causality*。
- 本实验的核心解释是：`Phi^R` 是基于经验状态分布的 ΦID 派生指标，适合作为相变附近信息动力学变化的描述量；`Phi^EID` 是机制干预口径下的 whole-system source-side synergy，更适合作为非负、机制归一化的相变对照指标。

## Reproducibility Metadata

```json
{
  "mode": "lausanne33_approximation",
  "source_results": "exp/brain/result_lausanne_fig6/count_00_fig6b_mean_rate.npz",
  "state_series": "excitatory_rate",
  "tau": 1,
  "ridge": 1e-06,
  "g_stride": 1,
  "main_phi_r_source": "cached_full_pair_curve",
  "plot_and_peak_detection_omit_g": 1.0,
  "caveat": "Approximate reproduction using Lausanne-33 count-derived 83-region matrices, not the exact HCP Lausanne-83 paper matrix."
}
```
