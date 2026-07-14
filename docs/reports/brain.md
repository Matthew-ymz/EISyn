# 脑科学实验：83 ROI 临界识别与 HCP500/1000 Yeo7 Phi 分解

## 结论

本报告保留两个互补实验。

1. **83 ROI 临界相变识别。**在 83 区 DMF 中，对全部兴奋性与抑制性门控变量实施独立均匀干预。无裁剪的 8-seed 主确认在 Kuramoto 理论临界点 $K_c=1.5958$ 附近识别到 $\Phi^{EID}$ 宽峰：$G=1.7$ 的均值为 $12.384\pm0.041$ bits，$G=1.6\text{-}1.8$ 的临界窗显著高于前、后窗口。整体 EI 与区域 EI 和在全部种子上严格下降，且归一化后与 Kuramoto 曲线方向对齐。
2. **HCP500/1000 PCA–Yeo7 Phi 分解。**在相同 30 名 HCP REST1_LR 被试中，Schaefer-500（$p=8,\alpha=10$）与重新验证的 Schaefer-1000（$p=5,\alpha=1$）均在 30/30 名被试中高于独立 PC1 circular-shift null。两种粒度的全七网络核都常进入 top-3，但不高于 matched null；相对地，缺少 Limbic 的六网络核均高于 matched null cohort（500：17/30 对 8.65/30；1000：12/30 对 6.35/30；各 20-null 未校正 $p=0.047619$）。

这两项实验分别回答不同问题：DMF 实验检验 $\Phi^{EID}$ 是否能定位可控模型中的临界动力学带；HCP 实验检验降维后的真实静息态网络动力学中是否存在高于同步破坏 null 的跨网络高阶结构。它们不构成对特定脑机制、因果方向或精确稀疏 atom 的证明。

## 目录

1. **83 ROI DMF：临界相变识别**
   1. 全状态最大熵干预与临界判据
   2. 临界窗中的 $\Phi^{EID}$ 峰与 EI 形状对齐
   3. 确定性 / 简并性分解与变化率
   4. 解释边界
2. **HCP500 PCA–Yeo7：Phi 分解**
   1. 数据、降维与动力学表征
   2. History-source $\Phi^{EID}$ 与 circular-shift null
   3. Yeo7 模块历史分解
3. **HCP1000 PCA–Yeo7：Phi 分解**
   1. 数据、降维与模型选择
   2. History-source $\Phi^{EID}$ 与 circular-shift null
   3. Yeo7 模块历史分解与 500 对照
4. **讨论：解释边界与可复现性**
   1. 结论的适用范围
   2. 结果与图形产物
5. [**附录 A：Kuramoto 振子数与 whole-state $\Phi^{EID}$ 曲线形状**](#appendix-a)
   1. [临界峰的 EI 与 effectiveness 机制](#appendix-a-1)
   2. [时间窗、相变前检测与系统规模边界](#appendix-a-2)

## 1. 83 ROI DMF：临界相变识别

### 1.1 全状态最大熵干预与临界判据

该实验使用 83 区 DMF、原始结构连接、直接长程兴奋耦合和 JFIC。每个试验同时干预全部兴奋性与抑制性门控变量，并将完整 166 维未来状态作为 target：

$$
\mathbf{s}_E,\mathbf{s}_I\overset{\mathrm{ind}}{\sim}U(0.30,0.70)^{83}.
$$

该分布是声明的生理支持 $[0.30,0.70]^{166}$ 内的最大熵分布；初态和 300 个 Euler 步的演化均不裁剪。主确认在 8 个独立种子、2048 个干预样本下评估 $G=1.0,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.2,3.0$。Gaussian block conditional-total-correlation 用于 166 维连续 EI；高维 transport map 在此样本规模下计算代价过高。

临界位置以可比 Kuramoto 全状态参照的理论 $K_c=1.5958$ 定义，而不再以旧的放电率阴影区替代。主判据是 $\Phi^{EID}$ 在该临界点附近形成内部峰/平台，并在高耦合端回落。

### 1.2 临界窗中的 $\Phi^{EID}$ 峰与 EI 形状对齐

$$
\Phi^{EID}=EI_{\mathrm{do}}(\mathbf{s}_t;\mathbf{s}_{t+\tau})-
\sum_{i=1}^{166}EI_{\mathrm{do}}(s_{t,i};\mathbf{s}_{t+\tau}).
$$

![DMF 全状态临界确认](../../fig/dmf_fullstate_maxent_critical_confirmation.png)

*图 1. 无裁剪的全状态 DMF 主确认。竖虚线为 Kuramoto 理论临界点 $K_c=1.5958$，灰带表示 DMF 的临界平台 $G=1.6\text{-}1.8$；阴影为 8 个种子的标准误。*

$\Phi^{EID}$ 在 $G=1.6\text{-}1.8$ 形成宽峰，最高均值位于 $G=1.7$：$12.384\pm0.041$ bits。它相对 $G=1.6$ 高 $0.106$ bits（配对单侧 $p=0.039$，7/8 个种子为正）；临界窗均值同时高于前、后窗口（两项 $p<1.2\times10^{-10}$，均为 8/8 个种子同向）。到 $G=3.0$，$\Phi^{EID}$ 降至 $8.144\pm0.022$ bits。

整体 EI 与区域 EI 和在全部 8 个种子、每个相邻耦合点上严格下降。归一化后，它们与 Kuramoto 的相应曲线在共同区间的秩相关均为 $1.00$；协同曲线的秩相关为 $0.967$。这支持临界识别来自“整体 EI 相对于局部 EI 和的暂时优势”，而不是总 EI 的上升。

![DMF 与 Kuramoto 的形状对齐](../../fig/dmf_kuramoto_fullstate_shape_alignment.png)

*图 2. 跨模型比较只使用归一化形状与峰位；DMF 的 Gaussian 近似与 Kuramoto 的 transport-map 估计不用于比较绝对 bits。*

### 1.3 确定性 / 简并性分解与变化率

令 $H_0$ 为本 sweep 的固定目标参考熵，则

$$
D_{\mathrm{whole}}=H_0-H(\mathbf{T}\mid\mathbf{S}),\qquad
G_{\mathrm{whole}}=H_0-H(\mathbf{T}),
$$

$$
D_{\Sigma}=166H_0-\sum_{i=1}^{166}H(\mathbf{T}\mid S_i),\qquad
G_{\Sigma}=166G_{\mathrm{whole}}.
$$

![DMF 原始确定性 / 简并性曲线](../../fig/dmf_fullstate_maxent_detdeg_integrated_raw.png)

*图 3. 原始 bits 曲线。左轴为整体项，右轴为区域项之和；双轴仅保留真实量级，不能据此比较变化速度。*

![DMF 归一化确定性 / 简并性与变化率](../../fig/dmf_fullstate_maxent_detdeg_integrated_rate.png)

*图 4. 左图将每个种子内的四项缩放到各自 sweep 的 $[0,1]$ 范围；右图为相对于 $G$ 的变化率。因此该图比较的是形状与速度，而非 bits 大小。*

整体确定性在临界前接近平台、随后较平缓地下降；区域确定性和与两项简并性则在临界窗前快速上升，并在 $G\approx1.7$ 后翻转为下降。整体与区域简并性归一化后完全重合，因为 $G_{\Sigma}=166G_{\mathrm{whole}}$ 是定义上的比例关系，并不表示两种独立速度。简并性在临界窗的峰表示目标熵相对 $H_0$ 最低；高耦合端它回落，表示目标熵回到参考水平。区域确定性和的高耦合回落说明单一区域对未来完整状态的预测性变弱，而完整 source 的联合预测优势仍保留到临界窗附近。这一相对差异产生 $\Phi^{EID}$ 的临界峰。

### 1.4 解释边界

$U(0,1)^{166}$ 是物理立方体上的绝对最大熵初态，但在该阴性对照中 $\Phi^{EID}$ 随 $G$ 下降，不能识别临界峰。因此本节的结论依赖于明确声明的 $[0.30,0.70]$ 生理支持，而不适用于无条件的 $[0,1]^{166}$ 干预。

该实验识别的是此 DMF、该代理结构连接、该时间窗和该连续高斯估计口径下的临界样动力学窗口，不对应人体大脑的固定耦合常数。确定性/简并性曲线提供与峰相容的熵分解，而不是同步、饱和或因果机制的直接测量。

## 2. HCP500 PCA–Yeo7：Phi 分解

### 2.1 数据、降维与动力学表征

数据为 30 名 HCP S1200 被试的 `REST1_LR` Schaefer-500 BOLD 时序，每名被试包含 $1200\times500$ 个时间点与皮层 parcel。500 个 parcel 按 Yeo7 标签分为 Vis、SomMot、DorsAttn、SalVentAttn、Limbic、Cont 和 Default；每个网络仅保留训练段拟合的一维 PCA（PC1），形成 7 维网络状态 $\mathbf{x}_t$。

Phi 实验使用 `sub-100206` 的时间验证选出的八阶 $\Delta$-Ridge，$p=8$、$\alpha=10$。模型以当前与过去七个时刻的网络状态为 source：

$$
\mathbf{h}_t=
\left[\mathbf{x}_t^\top,\mathbf{x}_{t-1}^\top,\ldots,\mathbf{x}_{t-7}^\top\right]^\top
\in\mathbb{R}^{56},
\qquad
\Delta\mathbf{x}_{t+1}=\mathbf{x}_{t+1}-\mathbf{x}_t.
$$

PCA、标准化器和 Ridge 均仅以每名被试的前 900 个时间点拟合；后 300 点不参与本 Phi 计算、参数选择或 null 重拟合。这里的目标是比较固定表征与固定模型下的 observed 和 null，而不是把该单被试模型推广为全体被试的预测最优模型。

### 2.2 History-source $\Phi^{EID}$ 与 circular-shift null

本实验的量是 56 维历史 source 到下一时刻 7 维状态的量，而不是 7 维一阶 whole-state $\Phi^{EID}$：

$$
\Phi^{EID}_{\mathrm{hist}\to\mathrm{next}}
=EI\!\left(\mathbf{h}_t;\mathbf{x}_{t+1}\right)
-\sum_{j=1}^{56}EI\!\left(h_{t,j};\mathbf{x}_{t+1}\right).
$$

估计使用 Gaussian log-det 口径，而非 TM。这样可在 56 维 history source、30 名被试、重复 null 重拟合和贪婪分解下保持可计算性；代价是结果依赖 Gaussian 近似，不能直接外推到非高斯的精确干预信息量。

null 对 7 条 PC1 时序分别施加独立、非零的 circular shift，并在相同的 $p$、$\alpha$ 与 900 点训练预算下重新拟合模型。它保留每条网络 PC1 的边际取值与自相关结构，但破坏网络间的时间对齐，检验观测到的跨网络结构是否超出网络内时间结构本身。

在 30 被试、每人 20-null 的扩展中，observed $\Phi^{EID}$ 的均值与中位数为 6.188481 与 6.068454 bits；observed-minus-null-mean 的均值与中位数为 1.984600 与 2.051671 bits，范围为 0.492521–4.096287 bits。30/30 名被试的 observed 都高于各自 null 均值，且未校正经验 $p<0.05$；由于仅有 20 个 null，每个被试的最小 p 值分辨率为 $1/21=0.047619$。

![30 被试 Yeo7-PC1 observed-minus-null PhiEID](../../results/hcp_schaefer500_yeo7_pc1_phi_null_all/observed_minus_null.png)

该结果支持在这一固定的 reduced-state、history-source 定义与 null 下，跨网络时间对齐带来额外的高阶结构；它不排除低频漂移、运动、生理噪声、PCA 表征选择或 Gaussian 近似造成的影响。

### 2.3 Yeo7 模块历史分解

在全部 30 名被试上进行贪婪分解。为避免将同一网络的八阶历史误作 8 个独立脑区，属于同一个 Yeo7 网络的全部 8 个滞后 PC1 值被绑定为一个不可拆模块原子；候选空间因此是 7 个网络模块，而不是 56 个逐滞后变量。

| 跨被试协同核 | 进入 top-3 | top 时原子贡献均值 | 固定集合未校正 $p<0.05$ |
|---|---:|---:|---:|
| 全部 7 个 Yeo 网络 | **20 / 30** | 1.314357 bits | 20 / 20 |
| Vis + SomMot + DorsAttn + SalVentAttn + Cont + Default | **17 / 30** | 1.227709 bits | 17 / 17 |
| SomMot + DorsAttn + SalVentAttn + Cont + Default | 9 / 30 | 0.956657 bits | 8 / 9 |
| Vis + SomMot + DorsAttn + SalVentAttn + Default | 8 / 30 | 1.086753 bits | 7 / 8 |

![30 被试模块级 greedy PhiEID 核](../../results/hcp_schaefer500_yeo7_module_phi_decomposition/top_core_consistency.png)

全 7 网络核最常出现（20/30），但它在 matched null cohort 中反而更常出现（均值 26.65/30；经验 $p=1$），因此不能将其读为真实数据特异的协同核。缺少 Limbic 的六网络广域核在真实数据中为 17/30，而 matched null cohort 的频率为 $8.65/30$（最大 12/30；经验 $p=1/21=0.047619$）；两个较小的候选核也分别为 9/30 对 $1.40/30$、8/30 对 $2.45/30$，同为该分辨率下的未校正 $p=1/21$。这支持真实静息态中若干非全网络模块核的出现频率和贡献高于该 circular-shift null；但 20 个 null 的 p 值分辨率有限，且统计未校正跨模块集合与 greedy 选择，因此不构成唯一生物学 atom 的确证。

## 3. HCP1000 PCA–Yeo7：Phi 分解

### 3.1 数据、降维与模型选择

同一 30 名 `REST1_LR` 被试的 `Schaefer1000` 矩阵为 $1200\times1000$。1000 个 parcel 按同一 Yeo7 顺序分为 Vis 162、SomMot 194、DorsAttn 122、SalVentAttn 121、Limbic 60、Cont 129 与 Default 212 个 parcel；每名被试的各网络 PC1 均只以前 900 点拟合并投影完整时序。

在 `sub-100206` 的训练段内以 600/700/800 三个时间验证折，从 $p\in\{1,2,3,5,8\}$ 与既有 Ridge $\alpha$ 网格选择模型。最优冻结配置为五阶 $\Delta$-Ridge，$p=5$、$\alpha=1$，平均 validation skill ratio 为 0.794433；因此 source 是 35 维网络历史，target 为下一时刻 7D 网络状态。后 300 点未参与 PC1、模型或参数选择。

### 3.2 History-source $\Phi^{EID}$ 与 circular-shift null

对每名被试固定上述表征与模型，并以每条 PC1 独立、非零 circular shift 后重拟合同一模型生成 20 个 null。1000-parcel observed $\Phi^{EID}$ 的均值/中位数为 7.783676/7.734082 bits；observed-minus-null-mean 的均值/中位数为 2.997670/3.032261 bits，范围为 0.814450–6.085586 bits。30/30 名被试的 observed 均高于其 null 均值，未校正经验 p 均小于 0.05（最小分辨率 $1/21=0.047619$）。

![30 被试 Schaefer1000 Yeo7-PC1 observed-minus-null PhiEID](../../results/hcp_schaefer1000_yeo7_pc1_phi_null_all/observed_minus_null.png)

### 3.3 Yeo7 模块历史分解与 500 对照

分解中，同一网络的全部五个 PC1 历史滞后绑定为一个不可拆模块原子；每个 observed 与 null 都完整运行 greedy top-3。1000-parcel 的常见核如下。

| 跨被试协同核 | 进入 top-3 | top 时原子贡献均值 | matched-null 频率均值；经验 p |
|---|---:|---:|---:|
| 全部 7 个 Yeo 网络 | **21 / 30** | 1.637465 bits | 21.80 / 30；0.761905 |
| Vis + SomMot + DorsAttn + SalVentAttn + Cont + Default | **12 / 30** | 1.575094 bits | 6.35 / 30；0.047619 |
| Vis + SomMot + DorsAttn + SalVentAttn + Default | 8 / 30 | 1.500212 bits | 2.40 / 30；0.047619 |
| Vis + SomMot + DorsAttn + Cont + Default | 7 / 30 | 1.579938 bits | 2.90 / 30；0.047619 |

![30 被试 Schaefer1000 模块级 greedy PhiEID 核](../../results/hcp_schaefer1000_yeo7_module_phi_decomposition/top_core_consistency.png)

| 描述性比较 | Schaefer-500 | Schaefer-1000 |
|---|---:|---:|
| observed $\Phi^{EID}$ 均值（bits） | 6.188481 | 7.783676 |
| observed − null 均值（bits） | 1.984600 | 2.997670 |
| observed 高于 null mean | 30 / 30 | 30 / 30 |
| 全七网络核 top-3 频率 | 20 / 30 | 21 / 30 |
| 缺 Limbic 六网络核 top-3 频率 | 17 / 30 | 12 / 30 |

两种粒度都复现了跨网络时间对齐高于 circular-shift null 的方向性证据，并都将缺 Limbic 的六网络广域核识别为高于 matched-null 频率的候选结构。绝对 bits、最优滞后阶数和 atom 频率受分区粒度、PC1 表征与单被试调参影响；上表仅作描述性对照，不能当作空间粒度的正式统计检验，也不将 greedy 核解释为唯一生物学 atom。

## 4. 讨论：解释边界与可复现性

### 4.1 结论的适用范围

- DMF 结果支持 $\Phi^{EID}$ 在该代理结构连接、无裁剪全状态干预与 $[0.30,0.70]^{166}$ 生理支持下定位 Kuramoto 对齐的临界窗；它不等同于人体大脑存在同一精确耦合常数，也不外推到绝对最大熵 $[0,1]^{166}$ 干预。
- HCP 结果来自 REST1_LR、30 名被试、7 个 PC1 网络状态及固定的八阶 Ridge；尚未检验独立 run、去趋势、运动或生理混杂回归、GSR、其他分区或其他 null 构造。
- HCP 的全体被试 Phi 结果使用 20 个 null，p 值分辨率有限，且未校正跨被试、跨模块集合和 greedy 选择造成的多重比较。
- 贪婪 atom 用于描述候选协同结构；它依赖分解顺序与候选空间，不是 exhaustive 的唯一高阶分解。

### 4.2 结果与图形产物

| 实验 | 关键图与结果 |
|---|---|
| 83 ROI 临界识别 | `fig/dmf_fullstate_maxent_critical_confirmation.{png,svg,pdf}`、`fig/dmf_kuramoto_fullstate_shape_alignment.{png,svg,pdf}`、`fig/dmf_fullstate_maxent_detdeg_integrated_raw.{png,svg,pdf}`、`fig/dmf_fullstate_maxent_detdeg_integrated_rate.{png,svg,pdf}`、`results/dmf_fullstate_uniform_support/confirm_c050_h020_tau300_n2048_no_clip_seeds3_10.npz` |
| HCP500 Yeo7-PCA Phi/null | `results/hcp_schaefer500_yeo7_pc1_phi_null/summary.json`、`results/hcp_schaefer500_yeo7_pc1_phi_null_all/summary.json`、对应 null 图 |
| HCP500 Yeo7 模块分解 | `results/hcp_schaefer500_yeo7_module_phi_decomposition/summary.json`、`results/hcp_schaefer500_yeo7_module_phi_decomposition/top_core_consistency.png` |
| HCP1000 Yeo7-PCA Phi/null | `results/hcp_schaefer1000_yeo7_ridge_selection/summary.json`、`results/hcp_schaefer1000_yeo7_pc1_phi_null_all/summary.json`、对应 null 图 |
| HCP1000 Yeo7 模块分解 | `results/hcp_schaefer1000_yeo7_module_phi_decomposition/summary.json`、`results/hcp_schaefer1000_yeo7_module_phi_decomposition/top_core_consistency.png` |

<a id="appendix-a"></a>

## 附录 A：Kuramoto 振子数与 whole-state $\Phi^{EID}$ 曲线形状

为避免把方程差异误读成振子数效应，这里重新使用同一个经典全局耦合 Kuramoto 方程，只改变振子数：

$$
\dot{\theta}_i
=\omega_i+\frac{K}{N}\sum_{j=1}^{N}\sin(\theta_j-\theta_i).
$$

除振子数外，两组实验使用同一协议：频率 $\omega_i$ 从零均值 Gaussian 抽样，随后对每个 seed 去均值并重缩放到 `sigma=1`；`N=2` 时这个协议退化为一对符号相反、标准差为 1 的频率。source 是全部振子的当前相位特征，target 是全部振子的未来相位状态，而不是整体速度；两组都直接计算与 Part2 大脑动力学 $\Phi^{EID}$ 相同的源侧 whole-minus-sum 结构：

$$
\Phi^{EID}
=
EI_{\mathrm{do}}(\{\mathbf{s}_t^i\}_{i=1}^{N};\mathbf{y}_{t+\tau})
-\sum_{i=1}^{N} EI_{\mathrm{do}}(\mathbf{s}_t^i;\mathbf{y}_{t+\tau})
.
$$

其中 $\mathbf{s}_t^i=(\cos\theta_i(t),\sin\theta_i(t))$ 是第 $i$ 个振子的相位特征，$\mathbf{y}_{t+\tau}=\{(\cos\theta_i(t+\tau),\sin\theta_i(t+\tau))\}_{i=1}^{N}$ 是系统整体未来相位状态。$\Phi^{EID}$ 由该定义保证非负，因此无需引入人工非负截断。

下文首先以 `N=64` Oracle 结果解释临界峰的机制；振子数对照作为系统规模边界证据，统一放在附录末尾。

<a id="appendix-a-1"></a>

### A.1 临界峰的 EI 与 effectiveness 机制

为了检查这个峰值来自哪一项，进一步把 `N=64` Oracle whole-state 结果分解为联合 EI 与单独 EI 之和：

![Large-N Kuramoto EI decomposition](../../fig/classic_network_dynamics_benchmark/large_kuramoto_n64_ei_decomposition.png)

分解结果显示，$EI_{\mathrm{do}}(\{\mathbf{s}_t^i\}_{i=1}^{N};\mathbf{y}_{t+\tau})$ 和 $\sum_i EI_{\mathrm{do}}(\mathbf{s}_t^i;\mathbf{y}_{t+\tau})$ 都随 $K$ 增大而整体下降。这不是反常现象，因为这里的 EI 衡量的是最大熵相位干预下，当前相位状态有多少可区分信息保留到未来 whole-state target 中。`K=0` 时，每个振子近似独立转动，当前相位到未来相位接近一一映射，所以联合 EI 和单独 EI 之和都很高，并且二者几乎相等，$\Phi^{EID}\approx0$。

随着 $K$ 增大，同步吸引会压缩相位差自由度，许多不同初始相位会被映射到更相似的未来状态，因此总的可区分信息下降。临界前沿附近，单个振子对未来全系统状态的解释力下降得更快，而联合状态仍保留对集体相位关系的解释力，所以两项差值扩大，$\Phi^{EID}$ 在 `K≈1.7` 达峰。到强同步区后，系统接近低维同步流形，联合 EI 本身也明显降低，差值随之回落。换言之，临界峰不是因为总 EI 最大，而是因为整体相对于部分之和的不可分解优势最大。

同一组 `N=64` Oracle 结果还可以按 effectiveness 的 determinism/degeneracy 口径拆开。这里固定参考熵 $H_0$ 为本 sweep 中最大的 Gaussian target entropy，并定义

$$
Det(\mathcal{S};\mathbf{Y})=H_0-H(\mathbf{Y}\mid \mathcal{S}),\qquad
Deg(\mathcal{S};\mathbf{Y})=H_0-H(\mathbf{Y}).
$$

其中 $\mathcal{S}$ 可以是全部振子的联合 source，也可以是某个单振子 source；$\mathbf{Y}$ 是 whole-state future target。为避免将四个高度相关的曲线拆散，左图把 whole-source determinism 与 degeneracy 合并到同一**线性**轴，从而突出 determinism 的低谷；右图在单一对数轴上并列 singleton-sum 的两项，保留其跨数量级的共同膨胀与接近。两图的同一条竖虚线标出 whole-source determinism 的最小点，便于把这两个尺度上的变化对齐。

![Large-N Kuramoto determinism and degeneracy decomposition](../../fig/classic_network_dynamics_benchmark/n64_detdeg/large_kuramoto_oracle_nsource_whole_state_phi_sweep_determinism_degeneracy.png)

这个分解补足了临界峰的解释。whole-state determinism 从 `K=0` 的约 `1110.05` bits 下降，在 `K=2.0` 附近降到约 `475.95` bits，随后强同步区又回升到 `K=4.0` 的约 `1078.10` bits；whole-state degeneracy 则从近零单调升高到 `K=4.0` 的约 `1044.37` bits。也就是说，强耦合同步并不是简单地让整体映射“更确定”；它同时把许多微观相位状态折叠到相似的未来同步状态，导致 degeneracy 急剧增加。EI 是二者的差，因此强同步区即便 determinism 回升，也会被更大的 degeneracy 抵消。

右图显示了为什么 $\Phi^{EID}$ 在临界附近最大。单振子口径的 degeneracy 被对每个 source 重复计算，随 $K$ 增大从 `K=1.0` 的约 `696.91` bits 快速升到 `K=4.0` 的约 `66839.89` bits；singleton-sum determinism 也在强同步区急剧放大，到 `K=4.0` 约 `66860.03` bits。两者都变大且彼此接近，说明单个振子在高同步区会获得大量共享的、重复的 whole-state 预测信息，但这些信息主要是同一个同步流形的冗余读出。临界附近则不同：联合状态仍能保留相位关系和集体模式，而单振子解释已经开始失效，所以 whole-minus-sum 差值在 `K≈1.7` 达到约 `279.63` bits。

<a id="appendix-a-2"></a>

### A.2 时间窗、相变前检测与系统规模边界

#### A.2.1 时间窗鲁棒性：避免强同步后，短窗不复现临界内部峰

基准 whole-state 曲线的 `tau=4` 结果保留为主对照。为检验其峰值是否只是高 $K$ 同步饱和造成的，新增一个严格配对的 multi-horizon Oracle sweep：对每个 seed，频率向量、均匀相位 intervention states 和 natural readout states 都固定并复用于全部 $(K,\tau)$ 条件；只改变统一的预测时间窗 $\tau\in\{0.5,0.75,1,1.5,2,4\}$，而不允许 $\tau$ 随 $K$ 自适应变化。所有条件仍使用 `N=64`、3 个 seeds、whole-state future phase target 与同一 N-source transport-map estimator。

![Paired large-N Kuramoto horizon sweep](../../fig/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_sweep.png)

图 A 以未来 target 的 raw global order 的 $99\%$ 分位数 $R_{0.99}$ 审计强同步。预先设定 guard 为：对所有 $K$ 都要求 $R_{0.99}<0.8$。`tau=0.5` 在最强耦合 `K=4` 仍只有 $R_{0.99}=0.583$，完全通过；`tau=0.75` 为 $0.746$，也通过（仅约 $0.37\%$ target samples 的 $R\ge0.8$）。从 `tau=1` 起该 guard 开始失效：`tau=1` 仅 `K=4` 失败，`tau=1.5` 在 `K=3.2,4` 失败，`tau=2` 在 `K\ge2.6` 失败，而 `tau=4` 在 `K\ge2.2` 失败。

关键结果在图 B：**通过 guard 的两个短窗并没有给出与原图相同的临界内部峰。** `tau=0.5` 的 $\Phi^{EID}$ 从 `K=0` 的约 $0$ bits 持续升至 `K=4` 的 $229.69$ bits；`tau=0.75` 同样在 `K=4` 最大，为 $262.20$ bits。因此，在目标尚未进入强同步区的有限短时间内，耦合增强主要表现为 whole-state 联合可预测性的持续增强，而非在 $K_c\approx1.596$ 附近形成回落前的峰。随着时间窗变长，最大值才逐步向低 $K$ 移动：`tau=1` 的峰在 `K=4`（$279.54$ bits），`tau=1.5` 在 `K=3.2`（$281.00$ bits），`tau=2` 在 `K=2.6`（$280.27$ bits），配对的 `tau=4` 在 `K=1.8`（$278.92$ bits），与原 `tau=4` 图中 `K\approx1.7` 的峰一致到扫描分辨率。

因此，原始临界前沿峰的正确表述应收紧为：它是**中等有限观测时间（此处约 $\tau=4$）下**，在高 $K$ 同步吸引已压缩 whole-state 信息后出现的 whole-minus-sum 优势峰；它不是对所有预测时间窗都成立的、时间尺度无关的临界指标。短窗结果同时排除了一个较弱的替代解释：该峰并非仅由高 $K$ target 已完全同步所产生，因为在明确未强同步的 `tau=0.5,0.75` 条件下，曲线反而没有内部峰。

#### A.2.2 更长时间窗：峰位穿过而非收敛于理论 $K_c$

为直接检验“继续增大 $\tau$ 后，峰是否会停在临界相变点”的假设，保持同一配对 protocol、`N=64`、3 个 seeds 和 full-sample TM estimator，将时间窗扩展为 $\tau\in\{4,6,8,10,12\}$。扫描在转变区加密到 $K=0.8,0.9,\ldots,2.6$，并保留 $K=0,0.4,3.2,4.0$ 锚点，以区分内部峰和扫描端点峰。

![Long-horizon paired large-N Kuramoto sweep](../../fig/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_long_horizon_refined.png)

结果不支持单调收敛后固定在理论 $K_c=1.596$ 的解释。随着 $\tau$ 从 4 增至 12，$\Phi^{EID}$ 的内部峰位依次为 $K_{\rm peak}=1.8,1.6,1.5,1.4,1.3$（峰值分别为 $278.92,274.52,272.83,271.65,269.61$ bits）。因此，`tau=6` 的 $K_{\rm peak}=1.6$ 只是在当前 $0.1$ 网格上恰好贴近 $K_c$；继续增加时间窗后，峰越过 $K_c$ 并持续移向更低的 $K$，而非在 $K_c$ 停留。所有这些峰都是加密区内部点，且其 $R_{0.99}$ 仅为 $0.644,0.561,0.523,0.492,0.492$，strong fraction 均为零；故该左移不是由峰落在高 $K$ 强同步 guard 失效区造成的。

更稳妥的结论是：$K_{\rm peak}(\tau)$ 是有限时间有效信息的时间尺度依赖 crossover，可能在某一中等时间窗掠过临界区，但不能把 $\tau\to\infty$ 的峰位等同于静态 Kuramoto 临界点。长窗极限还可能受相位混合和吸引子压缩控制；若要定义渐近临界指标，需要另行研究固定有限尺寸下的长时间衰减、再做 $N\to\infty$ 的有限尺寸标度，而不能从当前峰位外推。

#### A.2.3 相变前检测：共同早期弛豫窗中的 $\Phi(\tau)$ 谱

前述长窗峰位不能直接用作预警器。为检验能否在 future target 尚未同步时识别系统的**最终动力学区间**，对全部 $K\in[0,4]$ 保留同一短时间窗，而不是为高 $K$ 自适应延长或截短 horizon。已有的 `tau=0.5,0.75` 结果与新增的 $\tau\in\{0.1,0.2,0.3,0.4,0.6\}$ 配对合并，得到共同谱 $\tau\in\{0.1,0.2,0.3,0.4,0.5,0.6,0.75\}$。所有 $(K,\tau)$ 条件都满足 $R_{0.99}<0.8$；即使在 $K=4$、$\tau=0.75$，$R_{0.99}\approx0.75$，因此该谱只观测初始相位分布向同步吸引子弛豫的早期，而没有把已同步 target 当作特征。

![Pre-transition Kuramoto Phi-tau phase detection](../../fig/classic_network_dynamics_benchmark/large_kuramoto_pretransition_phi_tau_phase_detection.png)

图 B 显示：超临界 $K>K_c$ 条件在整个共同早期窗内已有更陡、更高的 whole-state $\Phi^{EID}(\tau)$ 谱，而此时图 A 证明其 target 尚未发生强同步。以已知的 $K_c=1.596$ 作为模拟中的超临界参考标签，只输入 7 个早期 $\Phi(\tau)$ 值，使用 leave-one-$K$-out（完整留出该 $K$ 的 3 个 seed）逻辑回归，得到超临界识别 AUROC 为 $0.983$。将每一条谱除以自身最大值、仅保留形状后，AUROC 仍为 $0.972$；因此区分力不只是 $\Phi$ 的整体幅度，时间尺度上的增长形状也携带信息。图 C 展示了留出 $K$ 后的预测概率。

##### A.2.3.1 识别算法与 AUROC 的计算

这个实验不是在单条真实轨迹上拟合未来标签，而是一个受控的 Oracle 可辨识性检验。数据单位是一个固定耦合和随机 seed 的组合 $(K,s)$。共有 17 个 $K$ 值、3 个 seed，因此有 $17\times3=51$ 个样本。对每个样本，先从同一 seed 的均匀初始相位 intervention support 出发，分别积分到 7 个早期 horizon，并计算 whole-state N-source 指标。输入特征向量为

$$
\mathbf{x}_{K,s}=
\left[
\Phi^{EID}_{K,s}(0.1),
\Phi^{EID}_{K,s}(0.2),
\Phi^{EID}_{K,s}(0.3),
\Phi^{EID}_{K,s}(0.4),
\Phi^{EID}_{K,s}(0.5),
\Phi^{EID}_{K,s}(0.6),
\Phi^{EID}_{K,s}(0.75)
\right].
$$

这里的每个 $\Phi^{EID}_{K,s}(\tau)$ 都是同一 whole-state 目标和同一 N-source transport-map estimator 下的

$$
\Phi^{EID}=EI_{\mathrm{do}}(\mathbf{S};\mathbf{Y}_{\tau})
-\sum_{i=1}^{64}EI_{\mathrm{do}}(\mathbf{s}_i;\mathbf{Y}_{\tau}),
$$

其中 $\mathbf{S}$ 是 64 个振子的联合当前相位特征，$\mathbf{s}_i$ 是第 $i$ 个振子的二维相位特征，$\mathbf{Y}_{\tau}$ 是 $\tau$ 后的 128 维 whole-state phase target。保留一个特征前，先审计自然 readout target 的 $R_{0.99}$；只有本实验中全部 $51$ 个样本都满足 $R_{0.99}<0.8$ 的共同 horizon 才进入上式。故模型没有看到已经强同步的 future target。

二分类标签不由 $\Phi$、早期 $R$ 或长时间 $R$ 阈值产生，而是由生成模型中已知的理论边界独立给出：

$$
y_K=\mathbb{I}(K>K_c),\qquad K_c=1.595769\ldots .
$$

这样标签表示“若继续演化，该参数属于超临界动力学区间”，而不是声称有限 $N$ 系统在一个任意 order 阈值处发生严格相变。每个样本另计算到 $\tau=20$ 的 raw order，作为连续审计量，但它不参与标签和分类器训练。

评估采用真正的 leave-one-$K$-out（LOKO）流程。对每个待测耦合 $K_*$：

1. 从训练集删除 $K_*$ 的全部 3 个 seed，只用其余 $16\times3=48$ 个样本。
2. 仅在这 48 个训练样本上，对每个特征维度计算均值 $\mu_j^{\mathrm{train}}$ 和标准差 $\sigma_j^{\mathrm{train}}$，并做标准化：

   $$
   \widetilde{x}_{ij}=\frac{x_{ij}-\mu_j^{\mathrm{train}}}{\sigma_j^{\mathrm{train}}}.
   $$

3. 在标准化后的训练集拟合固定正则强度 $C=1$ 的 logistic regression：

   $$
   \widehat p_{K,s}=\sigma\left(b+\mathbf{w}^{\mathsf T}\widetilde{\mathbf{x}}_{K,s}\right),
   \qquad
   \sigma(z)=\frac{1}{1+e^{-z}}.
   $$

4. 用该模型预测被完整留出的 3 个 $(K_*,s)$ 样本；遍历 17 个 $K_*$ 后，得到 51 个没有使用自身 $K$ 训练过的预测概率 $\widehat p_{K,s}$。

AUROC 不取某一个分类阈值，而检验这些概率是否把超临界样本整体排在次临界样本之前。令 $\mathcal{P}$ 是 24 个正类样本（8 个超临界 $K$、每个 3 个 seed），$\mathcal{N}$ 是 27 个负类样本（9 个次临界 $K$、每个 3 个 seed），则

$$
\operatorname{AUROC}
=
\frac{1}{|\mathcal{P}|\,|\mathcal{N}|}
\sum_{p\in\mathcal{P}}\sum_{n\in\mathcal{N}}
\left[
\mathbb{I}(\widehat p_p>\widehat p_n)
+\frac{1}{2}\mathbb{I}(\widehat p_p=\widehat p_n)
\right].
$$

本结果的 raw-spectrum AUROC 为 $0.9830247$，即 648 个正负样本对中有 637 对被正确排序（无并列时为 $637/648$）。shape-only 版本先将每个样本的谱除以该谱的最大值，再重复完全相同的 LOKO 流程；其 AUROC 为 $0.9722222$，即 630/648 对正确排序。后者是“谱形仍可分”的证据，而不是额外使用了 $K$、order parameter 或未来同步状态。

这里应把次临界状态称为**去相干／次临界动力学**，而不是默认称为“混沌相”：经典全局耦合 Kuramoto 的 $K<K_c$ 解一般可以是非同步的准周期运动，但不由本实验自动证明为严格混沌。有限 $N$ 下同步是 crossover，长时间 raw order 因而保留为连续审计量而未被任意阈值二分。当前结果的含义是：在这个已知方程、已知 $K_c$ 的 Oracle setting 中，早期 $\Phi(\tau)$ 谱可以预报未来进入超临界区；要转化为真实观测数据的预警器，仍需在未知参数、噪声、部分观测和独立时变轨迹上重新校准。

#### A.2.4 系统规模边界：只有大系统提供临界峰参照

![Kuramoto oscillator-count appendix](../../fig/part1_kuramoto_size_phi_eid_appendix.png)

该对照只展示 Oracle $\Phi^{EID}$ 与 corrected order，不再混入学习模型读出。**小 $N=2$ classic Kuramoto。** 在相同 whole-state 口径下，Oracle $\Phi^{EID}$ 没有形成清楚的内部临界峰；它在当前扫描范围内主要随强耦合增强，到 `K=4.0` 约为 `0.96` bits。`N=2` 的 corrected order 也不是热力学意义下的相变曲线，而是有限二振子锁相读数。

**大 $N=64$ classic Kuramoto。** 在完全相同的方程、source partition、whole-state target 和 $\Phi^{EID}$ 公式下，corrected global order 从低 $K$ 的近零状态进入高 $K$ 同步饱和区。理论临界耦合为 $K_c\approx1.596$；有限时间读出下最大斜率出现在 `K=2.2`。对应 Oracle N-source $\Phi^{EID}$ 从 `K=0` 的 `0` bits 升高，在 `K=1.7` 达峰，约 `279.63` bits；随后进入强同步区后明显回落，`K=4.0` 约 `13.58` bits。

这个边界对照说明，在方程形式、source/target 和 EI 分解公式都固定后，是否出现临界内部峰主要取决于系统规模。`N=2` 没有经典 Kuramoto 的热力学同步相变，所以不能期待它给出与大系统相同的 $\Phi^{EID}$ 峰；`N=64` 才提供清晰的 order-parameter 转变参照。

因此，Kuramoto 临界相变实验的核心证据链是三步：order parameter 给出同步转变区，whole-state $\Phi^{EID}$ 在转变前沿形成峰值，determinism/degeneracy 分解说明该峰来自“联合相位构型仍可区分、单振子读出快速冗余化”的差异，而不是来自总 EI、determinism 或 degeneracy 任一单项的简单最大化。
