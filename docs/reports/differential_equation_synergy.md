# Differential Equation Synergy Experiments

本文暂存从 `Part1.md` 移出的 Kuramoto、Wilson-Cowan 和 Coupled Rossler 三个微分方程协同实验。它们仍使用四种二源协同读出：WMS、SURD synergy、同一 fitted MLP 上的 SHAP interaction，以及 MLP+PEID synergy。

## Kuramoto Phase Coupling

按照 SIS 的修正口径，Kuramoto 也做一个零耦合锚点明确的参数扫描。三相位系统为

$$
\begin{aligned}
\dot x &= 1.0+\kappa\sin(w-x),\\
\dot y &= 1.1+\kappa\sin(w-y),\\
\dot w &= 0.9 .
\end{aligned}
$$

该实验使用 `kappa=[0,0.05,0.1,0.2,0.3]` 和 `4` 个 seed。WMS、SURD 与 SHAP 使用同一批自然相位 readout states；MLP 由自然状态和独立均匀相位状态混合训练；MLP+PEID 在独立均匀相位 intervention states 上读取 fitted vector field。为避免近常数目标被分位数离散放大成假信息，histogram PEID 对数值上近常数的目标向量退化为单一 bin。

- **零耦合锚点**：`kappa=0` 时 WMS、SURD 和修正后的 MLP+PEID synergy 均为 `0`，SHAP interaction 为 `2.7e-8`，符合无相互作用预期。
- **正耦合区间**：MLP+PEID 在 `kappa=0.05` 即升至 `1.0899 ± 0.0133` bits，此后在 `1.08-1.09` bits 附近保持平台。这说明它稳定区分了无耦合与有相位门控，但互信息读数对目标幅值缩放不敏感，不能解释为 $\kappa$ 的线性强度计。
- **自然轨迹读出**：SHAP interaction 随 `kappa` 从约 `0.046` 增至 `0.165`，更像幅值敏感的响应面指标。WMS 和 SURD 则明显受自然相位访问分布影响，WMS 在高 `kappa` 转为负值，SURD 随 `kappa` 增大下降，因此不适合作为结构耦合强度本身。

## Wilson-Cowan Natural-Trajectory Gain Sweep

Wilson-Cowan 使用结构可加的三节点 fork：

$$
\dot w=-w+\sigma_g(w),\qquad
\dot x=-x+\sigma_g(w),\qquad
\dot y=-y+\sigma_g(w),
\qquad
\sigma_g(u)=\frac{1}{1+\exp[-g(u-1)]}.
$$

这里扫描 `g=[1,2,3.5,5.1,7.5]`，并平均 `{w,x}->dx` 与 `{w,y}->dy`。方程没有显式二源乘积或相位差项，因此该 panel 是神经动力学中的结构交互负对照。为避免单条轨迹快速收敛到固定点，每个 seed 使用 `12` 个独立初值，每条轨迹保留短暂瞬态并抽取 `150` 个样本；训练池和 readout 池使用不同初值。

Panel e 中 MLP+SHAP interaction 始终很小，从约 `0.0030` 增至 `0.0249`，符合结构近似可加的预期。与此同时，SURD synergy 从约 `0.3204` 在 `g=2` 升至 `0.4146`，随后降至 `0.1477`；自然轨迹 WMS 随高增益变得更负，在 `g=7.5` 约为 `-0.5348`。独立均匀干预下的 MLP+PEID 则保持正值，从 `g=1` 的约 `0.3381` 增至 `g=3.5` 的 `0.8074`，高增益下仍约为 `0.7375-0.8002`。这一区分说明自然轨迹上的负净协同来自两个单源信息的冗余重叠，而不是 PEID 计算得到负的 synergy 原子。

## Coupled Rossler Natural-Trajectory Coupling Sweep

两个 Rossler 振子通过相位差项双向耦合：

$$
\dot x_i=-y_i-z_i+\kappa\sin(x_j-x_i),\qquad
\dot y_i=x_i+0.165y_i,\qquad
\dot z_i=2+z_i(x_i-5.5).
$$

这里扫描 `kappa=[0,0.1,0.25,0.5,0.75]`，并平均 `{x0,x1}->dx0` 与 `{x0,x1}->dx1`。数据生成与 Wilson-Cowan 相同，MLP 只用多初值自然轨迹训练，四种方法共享 held-out 自然轨迹 readout 池。

Panel f 没有恢复出“零耦合为零、随耦合增强”的结构曲线。即使 `kappa=0`，自然轨迹 WMS 与 SURD 仍分别约为 `0.5254` 和 `0.8158`；MLP+SHAP interaction 约为 `0.0522-0.0631`。独立均匀干预下的 MLP+PEID 也在整个扫描上保持约 `0.050-0.053`，而非随 coupling 单调增加。该负结果说明：自然轨迹训练的 MLP 没有把耦合强度变化可靠外推到干预域；同时自然轨迹观测读出受到吸引子相关影响。预测器和信息读出都不能仅凭这类曲线被解释为耦合强度估计器。
