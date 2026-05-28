# 非负单调协同 ODE 图中指标的计算说明

本文档解释当前非负单调协同 ODE 实验中几类图的纵轴、色值和信息指标如何计算，并说明为什么协同 EI 高时联合点火收益更高。

## 1. 动力学与参数

三节点动力学为

$$
\begin{aligned}
\dot{x}_0 &= -\lambda_s x_0,\\
\dot{x}_1 &= -\lambda_s x_1,\\
\dot{x}_2 &= -\lambda_t x_2
+\alpha \log(1+\max(gx_0x_1,0))
+(1-\alpha)\rho(x_0+\epsilon x_1).
\end{aligned}
$$

当前默认高协同实验使用

$$
\alpha=0.9,\quad g=1,\quad \lambda_s=1,\quad \lambda_t=1,\quad \rho=1,\quad \epsilon=0.01.
$$

这里不用 $\log(gx_0x_1)$，因为全零初始条件和单源点火都会让 $x_0x_1=0$。实现使用

$$
m(x_0,x_1)=\log(1+\max(gx_0x_1,0)),
$$

所以 log 的实际输入至少为 1，不会取到 0；在非负状态空间中，联合项也始终非负且单调。

点火阶段把被干预的源节点固定在指定幅度。目标节点满足

$$
\dot{x}_2=-x_2+F_\alpha(\delta_0,\delta_1),
$$

其中

$$
F_\alpha(\delta_0,\delta_1)
=\alpha\log(1+\delta_0\delta_1)
+(1-\alpha)(\delta_0+0.01\delta_1).
$$

给定点火持续时间 $T$ 时，

$$
x_2(T;\delta_0,\delta_1)=F_\alpha(\delta_0,\delta_1)(1-e^{-T}).
$$

后续响应图的纵轴或色值都对应目标变量取值 $x_2(T)$，不是目标输入项 $F_\alpha$。

## 2. 固定持续时间下的点火响应曲线

固定持续时间曲线使用 $T=8$。横轴是总点火成本 $C$，四条曲线对应：

| 曲线 | 干预幅度 | 目标输入项 |
|---|---:|---|
| No ignition | $\delta_0=0,\delta_1=0$ | $0$ |
| Node 0 ignition | $\delta_0=C,\delta_1=0$ | $(1-\alpha)C$ |
| Node 1 ignition | $\delta_0=0,\delta_1=C$ | $(1-\alpha)0.01C$ |
| Pair ignition | $\delta_0=C/2,\delta_1=C/2$ | $\alpha\log(1+C^2/4)+(1-\alpha)(0.505C)$ |

纵轴计算为

$$
x_2(T)=F_\alpha(1-e^{-8}).
$$

在默认高协同设置 $\alpha=0.9$ 下，$C=4$ 时：

| 指标 | 数值 |
|---|---:|
| 最佳单源点火响应 | 0.3999 |
| 联合点火响应 | 1.6499 |
| 联合 / 最佳单源 | 4.1262 |
| 双源剩余响应 | 1.2461 |

因此，高协同设置下联合点火明显优于单源点火。

## 3. 干预幅度与持续时间的二维曲面

二维曲面使用同一联合点火规则：

$$
\delta_0=\delta_1=C/2.
$$

横轴是总点火成本 $C$，纵轴是点火持续时间 $T$，颜色表示目标变量取值

$$
x_2^{01}(T;C)
=\left[0.9\log(1+C^2/4)+0.0505C\right](1-e^{-T}).
$$

这个曲面展示了两个事实：沿时间方向，$1-e^{-T}$ 增加并逐渐饱和；沿成本方向，$\log(1+C^2/4)$ 非负、单调增加，但边际收益逐渐变小。因此当前曲面不再出现正弦相位导致的负响应。

## 4. 双源剩余响应

双源剩余响应定义为

$$
\Delta_{\mathrm{pair}}(C)
=x_2^{01}(T;C)-x_2^0(T;C)-x_2^1(T;C)+x_2^\varnothing(T;C).
$$

固定 $T=8$、$\alpha=0.9$ 时，最大双源剩余响应出现在当前成本网格的最大值：

$$
C=4,\quad \Delta_{\mathrm{pair}}=1.2461.
$$

这不是因为响应有峰值，而是因为非负单调联合项在当前成本区间内持续增强。

## 5. EI 与 Syn 的计算

EI 分解使用非负随机双源干预分布：

$$
(\delta_0,\delta_1)\sim \mathrm{Unif}([0,2]^2).
$$

每个样本经过同一动力学得到目标变量

$$
Y_T=x_2(T;\delta_0,\delta_1).
$$

估计时使用 `target_noise_fraction=0.1`，避免确定性标量映射把低 $\alpha$ 下的联合 EI 估得过高。

单源特征提升为

$$
\phi_i(\delta_i)=(\delta_i,\delta_i^2,\delta_i^3),
$$

单源 EI 为

$$
EI_i(T)=I(\phi_i(\delta_i);Y_T).
$$

双源联合特征为

$$
\phi_{01}(\delta_0,\delta_1)
=(\delta_0,\delta_1,\delta_0\delta_1,\delta_0^2,\delta_1^2),
$$

联合 EI 为

$$
EI_{01}(T)=I(\phi_{01}(\delta_0,\delta_1);Y_T).
$$

协同项和协同比例为

$$
Syn_{01}(T)=EI_{01}(T)-EI_0(T)-EI_1(T),
$$

$$
SynRatio_{01}(T)=\frac{Syn_{01}(T)}{EI_{01}(T)}.
$$

默认高协同设置 $\alpha=0.9$、$T=8$ 的分解为

| 指标 | 数值 |
|---|---:|
| $EI_0$ | 0.4384 |
| $EI_1$ | 0.2328 |
| $EI_{01}$ | 2.0323 |
| $Syn_{01}$ | 1.3610 |
| $SynRatio_{01}$ | 0.6697 |

这说明目标变量的信息主要来自两个源变量的联合状态，而不是任一源变量的单独变化。

## 6. alpha 对照：低协同与高协同

alpha 扫描用于展示协同 EI 与联合干预收益之间的关系。低 $\alpha$ 时，目标主要由节点 0 的单源通道解释；高 $\alpha$ 时，目标主要由 $\log(1+x_0x_1)$ 联合项解释。

| 设置 | $SynRatio$ | 最佳单源响应 | 联合响应 | 联合 / 最佳单源 | 双源剩余 |
|---|---:|---:|---:|---:|---:|
| $\alpha=0.05$ | 0.0273 | 3.7987 | 1.9988 | 0.5262 | -1.8379 |
| $\alpha=0.9$ | 0.6744 | 0.3999 | 1.6499 | 4.1262 | 1.2461 |

这正是点火问题要表达的机制：协同 EI 低时，联合干预并不比最佳单源干预更有收益；协同 EI 高时，联合干预可以打开单源无法有效激发的非线性通道。

## 7. EI 与 Syn 为什么随时间基本平稳

时间曲线计算同一组随机干预样本在不同演化时间 $T$ 下的

$$
EI_0(T),\quad EI_1(T),\quad EI_{01}(T),\quad Syn_{01}(T).
$$

在当前目标方程中，

$$
Y_T=F_\alpha(\delta_0,\delta_1)(1-e^{-T}).
$$

不同 $T$ 主要给目标变量乘上一个正的尺度因子。互信息在连续变量的可逆尺度变换下保持不变；实现中噪声也按目标标准差比例加入，所以估计值随时间基本保持水平。

## 8. 六节点关键节点对联合点火

为避免三节点网络只验证一个预设节点对，新增受控六节点实验。节点 $0,\ldots,4$ 是候选源节点，节点 $5$ 是目标节点，目标节点不参与点火。六个节点之间有固定有向网络连接，每个节点都有不同的入邻居，其中节点 $0\to1\to2\to0$ 构成三节点 feedback loop。

基础网络动力学为

$$
\dot{x}_k=-\lambda_k x_k+\sum_j A_{kj}x_j,\quad k=0,\ldots,5.
$$

目标节点额外接收协同读出和单源读出：

$$
\dot{x}_5=-\lambda_t x_5
+\sum_j A_{5j}x_j
+\alpha\sum_{(i,j)\in \mathcal{E}_{syn}} w_{ij}\log(1+gx_i x_j)
+(1-\alpha)\sum_{i=0}^4 b_i x_i.
$$

默认嵌入三条真实协同通道：

| 节点对 | 协同权重 |
|---|---:|
| $(0,1)$ | 1.00 |
| $(2,3)$ | 0.65 |
| $(1,4)$ | 0.40 |

所有候选源节点对共 10 个都会计算 pair Syn，并在同一个最大总成本 $C=4$ 下比较联合点火收益。联合点火仍使用等分成本：

$$
\delta_i=\delta_j=C/2.
$$

每个节点对的点火效果直接用联合点火后的目标响应表示：

$$
Response_{ij}=x_5^{ij}(T;C).
$$

同时保留双源剩余响应作为补充诊断：

$$
Surplus_{ij}=x_5^{ij}(T;C)-x_5^i(T;C)-x_5^j(T;C)+x_5^\varnothing(T;C).
$$

多节点实验的核心检验不是只看预设的 $(0,1)$，而是看 10 个候选节点对中 Syn 排名是否能把嵌入协同通道排到前面，并且 Syn 与 $Response_{ij}$、$Surplus_{ij}$ 是否呈正相关。若相关不够高，说明当前协同指标在该网络和干预分布下只能部分反映点火效果；文档与图中按实际结果报告，不强行改写为正结论。

## 9. 图像文件

![固定持续时间下的点火响应曲线](../results/network_revival_three_node_synergy/figures/sin_synergy_ignition_target_response.png)

![联合点火的幅度-持续时间曲面](../results/network_revival_three_node_synergy/figures/sin_synergy_pair_duration_surface.png)

![双源剩余响应](../results/network_revival_three_node_synergy/figures/sin_synergy_ignition_pair_surplus.png)

![EI 与 Syn 随演化时间变化](../results/network_revival_three_node_synergy/figures/sin_synergy_ei_time_curve.png)

![EI 协同分解](../results/network_revival_three_node_synergy/figures/sin_synergy_ei_decomposition.png)

![alpha 对照：协同 EI 与联合点火收益](../results/network_revival_three_node_synergy/figures/sin_synergy_alpha_sweep_summary.png)

![六节点 pair Syn 热图](../results/network_revival_three_node_synergy/figures/multi_node_pair_synergy_heatmap.png)

![六节点 Syn 与联合点火目标响应](../results/network_revival_three_node_synergy/figures/multi_node_synergy_vs_pair_response.png)

![六节点代表节点对点火响应曲线](../results/network_revival_three_node_synergy/figures/multi_node_pair_response_curves.png)

![六节点节点对排序摘要](../results/network_revival_three_node_synergy/figures/multi_node_pair_summary_rankings.png)
