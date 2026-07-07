# HCP Lausanne-83 PhiEID Pilot 展示

本文展示当前 `HCP-YA Lausanne-83 PhiEID pilot` 管线的第一次运行结果。需要先说明：这次运行使用的是 `--synthetic` smoke 数据，不是真实 HCP fMRI。它的价值是验证下载/ROI/拟合/null/分解/绘图/报告这一整条流程是否可跑通，并检查指标读数是否会被 null 检验拦住；它不能作为 HCP 神经科学结论。

## 1. 当前结论

当前 smoke 结果不支持“观测 PhiEID 显著高于 null”。

| 项目 | 结果 |
|---|---:|
| 运行数 | 10 |
| ROI 数 | 83 |
| 每个 run 的 null 次数 | 4 |
| 平均观测 raw PhiEID | 201.799 bits |
| 平均 null raw PhiEID | 201.934 bits |
| 平均差值，观测减 null | -0.135 bits |
| median empirical p-value | 0.600 |
| p-value <= 0.05 的 run 数 | 0 / 10 |
| Ridge validation correlation 均值 | 0.051 |
| Ridge RMSE / persistence RMSE 均值 | 1.469 |

这里最重要的读数有两个。第一，观测 PhiEID 没有高于 circular-shift null，平均值还略低于 null。第二，Ridge 的一步预测弱于 persistence baseline，说明当前 synthetic smoke 的线性 transition model 本身没有学到强可用动力学。因此后续的模块分解只能当作流程检查，不能解释成真实或稳定的高阶脑区机制。

![HCP Lausanne-83 PhiEID null comparison](../../fig/hcp_lausanne_phi_eid_null_comparison.png)

图 1 显示每个 synthetic run 的观测 raw PhiEID 与 null 均值。10 个 run 中有的观测值高于 null，有的低于 null；右侧 p-value 分布没有靠近 0.05。这个结果符合“流程能跑通，但当前 smoke 数据没有给出显著高阶整合信号”的判断。

## 2. 分解结果

尽管总体 null 检验没有通过，分解图仍然有用：它说明分解代码、模块映射、ROI burden 输出和图形导出都能正常工作。

![HCP Lausanne-83 PhiEID decomposition](../../fig/hcp_lausanne_phi_eid_decomposition.png)

当前 smoke 中，模块级 greedy atom 的前几项集中在包含 DMN、FPN、Lim 和 Sub 的大块组合上：

| Rank | 模块 atom | Mean atom PhiEID |
|---:|---|---:|
| 1 | DMN + FPN + Lim + Sub | 44.060 |
| 2 | DMN + Vis + FPN + Lim + Sub | 43.124 |
| 3 | DMN + Som + Vis + VAN + FPN + Lim + Sub | 38.539 |
| 4 | DMN + Som + Vis + FPN + Lim + Sub | 30.040 |
| 5 | DMN + FPN + Sub | 14.062 |

ROI leave-one-out burden 的 top candidates 包括：

| Rank | ROI | Module | Mean burden |
|---:|---|---|---:|
| 1 | ctx-lh-transversetemporal | Som | 3.102 |
| 2 | ctx-rh-paracentral | Som | 3.073 |
| 3 | Right-Hippocampus | Sub | 3.058 |
| 4 | ctx-rh-transversetemporal | Som | 3.050 |
| 5 | ctx-lh-supramarginal | VAN | 2.576 |
| 6 | ctx-lh-parsorbitalis | FPN | 2.554 |
| 7 | ctx-lh-temporalpole | Lim | 2.552 |
| 8 | ctx-rh-parahippocampal | Lim | 2.066 |
| 9 | ctx-rh-parsopercularis | FPN | 2.065 |
| 10 | Right-Thalamus-Proper | Sub | 2.050 |

这些结果目前只能解释为 synthetic smoke 下的候选集中位置。由于总体 PhiEID 没有超过 null，而且模型预测也不强，不能把这些模块或 ROI 写成真实 HCP 中的高阶整合中心。

## 3. 方法口径

当前 pipeline 使用 Lausanne/Desikan-83 ROI 命名口径。真实 HCP 模式下，脚本会读取 HCP S1200 的 cleaned resting-state fMRI volume 和 `MNINonLinear/aparc+aseg.nii.gz`，再按 FreeSurfer label 提取 83 个非 `Unknown` ROI 时间序列。

动力学模型采用一步预测：

$$
\mathbf{x}_t \rightarrow \mathbf{x}_{t+1},
$$

其中 $\mathbf{x}_t$ 是 83 维 ROI 状态。主模型为 Ridge transition，MLP 作为可选非线性 surrogate。whole-state PhiEID 使用 Gaussian log-det screening：

$$
\Phi^{EID}
= EI(\mathbf{x}_t;\mathbf{x}_{t+1})
- \sum_i EI(x_t^i;\mathbf{x}_{t+1}).
$$

null 使用每个 ROI 独立 circular shift，保留单个 ROI 的自相关结构，同时破坏跨 ROI 同步关系。经验 p-value 为：

$$
p
= \frac{1 + \#\{\Phi_{\mathrm{null}} \ge \Phi_{\mathrm{obs}}\}}
{1 + N_{\mathrm{null}}}.
$$

## 4. 解释边界

这次结果的边界很清楚：

- 当前是 synthetic smoke，不是真实 HCP。
- null 次数只有 4，足够检查流程，不够做稳定统计。
- Ridge validation correlation 约 0.051，且 RMSE 高于 persistence baseline。
- full 83D PhiEID 用 Gaussian log-det 是轻量筛查；full-dimensional TM 对当前 pilot 过重，后续应只在低维模块级复核。
- ROI burden 是 leave-one-out candidate score，不是 83D exhaustive high-order atom。

## 5. 下一步

下一步应先让真实 HCP 路径跑通，而不是继续解释 smoke 分解。

1. 安装真实 HCP ROI 提取所需依赖：`nibabel` 和 `nilearn`。
2. 配置 HCP S1200 下载路径或 AWS/HCP open-access 权限。
3. 先跑 1 个 subject、1 个 run、`null_reps=2` 的真实 smoke。
4. 如果 Ridge 预测仍弱于 persistence，应先改模型或预处理，不进入神经解释。
5. 如果真实 PhiEID 稳定高于 null，再把样本扩到 5-10 人，并增加 null 次数到 20 或更高。

当前可汇报的一句话结论是：**HCP Lausanne-83 PhiEID pilot 管线已经跑通，但 smoke 结果没有显示观测 PhiEID 高于 null；在真实 HCP 数据和更强预测模型验证前，不能声称发现了脑区高阶整合结构。**
