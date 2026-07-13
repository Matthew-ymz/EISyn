# HCP Schaefer-500 动力学与 PhiEID 实验汇总

## 结论与推荐配置

本实验使用 30 名 HCP S1200 被试的 `REST1_LR` Schaefer-500 静息态时序，每名被试有 1200 个时间点和 500 个皮层 parcel。当前最稳妥的预测主模型固定为：

$$
\widehat{\mathbf{x}}_{t+1}
=
\mathbf{x}_t+\widehat{\Delta\mathbf{x}}_t,
\qquad
\Delta\mathbf{x}_t
=
\mathbf{x}_{t+1}-\mathbf{x}_t,
$$

其中以当前状态 \(\mathbf{x}_t\) 预测变化量 \(\Delta\mathbf{x}_t\)，使用 Ridge `alpha=100` 和历史阶数 `p=1`。该固定配置在群体内的平均验证 skill ratio 为 0.878074；30/30 名被试的最终测试误差低于 persistence。

这是一项预测调优结论，不改变既有 500 维 one-step $\Phi^{EID}$ 的定义或解释。现有 null 结果只构成估计器诊断，尚不支持对真实跨脑区整合做群体推断。

## 数据与共同设置

- 数据：30 名被试、每人 \(1200\times500\) 的 `Schaefer500` BOLD parcel 时序。
- 输入处理：不额外执行去趋势、全序列 z-score、滤波、GSR 或 scrubbing；每个模型仅从相应训练段估计数值标准化参数。
- 预测基准：persistence，即 \(\widehat{\mathbf{x}}_{t+1}=\mathbf{x}_t\)。
- 预测指标：

$$
R_{\mathrm{skill}}
=
\frac{\mathrm{RMSE}_{\mathrm{model}}}
{\mathrm{RMSE}_{\mathrm{persistence}}}.
$$

小于 1 表示模型优于 persistence。

## 原始 one-step PhiEID 基线

最初使用固定 `alpha=1`、lag 1 的 Ridge 状态模型，并在 Gaussian log-det 口径下计算：

$$
\Phi^{EID}
=
EI(\mathbf{x}_t;\mathbf{x}_{t+1})
-
\sum_{i=1}^{500}EI(x_{t,i};\mathbf{x}_{t+1}).
$$

30 名被试均得到有限正的 signed raw $\Phi^{EID}$：均值 1051.818 bits、中位数 1053.334 bits、范围 940.714–1123.707 bits。

但该固定模型的预测诊断未通过：平均 $R_{\mathrm{skill}}=1.322$，30/30 名被试均未超过 persistence。因此这些 raw $\Phi^{EID}$ 数值只能作为可计算性与量级基线，不能作为可靠整合证据，也不能与不同分区数或估计口径直接比较。

## 30 被试预测调优

### 选择协议

每名被试独立执行时间嵌套选择：前 900 点为开发段，后 300 点仅用于最终测试；开发段内以三个 expanding-window validation folds 选择参数。候选范围为：

- `alpha ∈ {1e-4, 1e-3, ..., 1e6}`；
- 历史阶数 `p ∈ {1, 2, 3, 5, 8}`；
- 比较固定状态 Ridge、调优状态 Ridge、调优 Δ-Ridge(`p=1`) 与调优 Δ-Ridge(history)。

候选搜索不计算 $\Phi^{EID}$；只有预测评分进入选择，避免重复高维 log-det 计算。

### 最终测试结果

| 模型 | Test skill ratio（均值） | Test skill ratio（中位数） | 优于 persistence |
|---|---:|---:|---:|
| 固定状态 Ridge，`alpha=1, p=1` | 1.233593 | 1.226911 | 1/30 |
| 调优状态 Ridge，`p=1` | 0.904982 | **0.888636** | 27/30 |
| 调优 Δ-Ridge，`p=1` | **0.881074** | 0.892226 | **30/30** |
| 调优 Δ-Ridge，history | 0.881252 | 0.892074 | **30/30** |

预先设定的成功条件是中位数小于 1，且至少 24/30 名被试优于 persistence。三个调优模型都达到该条件。状态模型的中位数略低，而 Δ-Ridge 的均值更低且对所有被试均超过 persistence；因此采用 Δ-Ridge 作为固定、易复现的主模型。

### 参数筛选结果与固定化依据

| 模型 | 个体选择分布 | 群体平均 validation skill ratio | 固定建议 |
|---|---|---:|---|
| 调优状态 Ridge | `alpha=1000`: 23/30；`alpha=100`: 7/30；均为 `p=1` | `alpha=1000`: 0.866287 | 用于直接状态对照时固定 `alpha=1000, p=1` |
| 调优 Δ-Ridge，`p=1` | `alpha=100`: 18/30；`alpha=1000`: 12/30 | `alpha=100`: **0.878074**；`alpha=1000`: 0.884954 | **主模型固定 `alpha=100, p=1`** |
| 调优 Δ-Ridge，history | `(100,1)`: 18/30；`(1000,1)`: 10/30；`(1000,2)`: 2/30 | `(100,1)`: **0.878074** | 不保留额外历史项 |

因此参数并非逐被试完全相同，但结论足够集中：不存在稳定的高阶历史需求，且 Δ-Ridge 的群体平均最优正则为 `alpha=100`。后续复现实验默认固定该配置，而不再进行被试内网格搜索；若需要保留直接状态模型作对照，则使用 `alpha=1000, p=1`。

## PhiEID 与 circular-shift null 诊断

现有 null 分析使用的是调优状态模型，而非上述固定 Δ-Ridge 主模型；因此其角色是估计器诊断，不能与主预测模型混为一谈。

### 单被试、20-null smoke

对 `sub-100206`，使用前 900 点、按该被试调优得到的状态模型 `alpha=1000`，对每个 ROI 独立 circular shift：

| observed raw Phi（bits） | null mean（bits） | observed − null mean（bits） | empirical p |
|---:|---:|---:|---:|
| 8.213938 | 18.734441 | -10.520503 | 1.000000 |

该 20-null pilot 用时 49.63 秒；其负的 observed-minus-null 差值不支持“observed raw Phi 高于 independent-shift null”的解释。

### surrogate 行为诊断

在同一被试与训练预算下，observed raw Phi 为 8.213938 bits，平均 parcel lag-1 自相关为 0.742836：

| surrogate | null Phi 均值（bits） | observed − null（bits） |
|---|---:|---:|
| global circular shift | 8.043499 | 0.170440 |
| independent circular shift | 18.641999 | -10.428061 |
| independent phase surrogate | 18.671803 | -10.457865 |

该模式说明 independent surrogate 会明显改变当前 Gaussian log-det 指标的量级。它不是人群显著性检验，也不能排除预处理、协方差正则或 surrogate 定义的影响。需要在固定主预测模型下，完成多被试、多 null 次数及去趋势/混杂控制敏感性分析后，才能讨论 $\Phi^{EID}$ 的脑科学含义。

## 复现与产物

- 原始 Phi baseline：`scripts/run_hcp_schaefer500_raw_phi.py`；结果：`results/hcp_schaefer500_raw_phi/summary.json`。
- 动力学调优：`scripts/run_hcp_schaefer500_tuned_dynamics.py`；结果：`results/hcp_schaefer500_tuned_dynamics/summary.json`。
- null screening：`scripts/run_hcp_schaefer500_tuned_phi_null.py` 和 `scripts/run_hcp_schaefer500_tuned_phi_null_diagnostic.py`；结果位于对应的 `results/` 目录。

smoke 报告已并入上述方法验证与 null 诊断，不再单独维护。
