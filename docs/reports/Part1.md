# 对比方法介绍

同一条模拟时间序列先用于训练一个 MLP 一步转移模型，输入为 `[x_t, y_t, z_t, w_t]`，输出为 `[x_{t+1}, y_{t+1}, z_{t+1}, w_{t+1}]`。随后在同一轨迹或固定 MLP 上读出几类量：

- **Granger/ablation**：把某个 source 的输入列替换为均值，记录目标预测 MSE 的增量。它回答“去掉这个变量会不会损害预测”。
- **Neural Granger**：对每个 target 单独训练带 group-lasso 的 cMLP，并读取第一层按 source lag group 聚合的权重范数。它回答“target-wise 非线性预测器是否使用这个 source 的历史输入”，仍是 pairwise 预测结构读出。
- **SHAP 类归因**：在同一 fitted MLP 上只保留一个常用背景替换式 SHAP 基线，用经验背景替换未给定特征。单特征 SHAP 报告 mean absolute attribution；二阶 SHAP interaction 报告 `x:y` 的 mean absolute interaction。前者回答“某个特征分到多少预测贡献”，后者回答“两个特征的非加性预测贡献有多大”。交互项：在同一 fitted MLP 的最大熵干预预测面上，用标准化主效应加一个二阶乘积项拟合目标输出。它回答“固定这个预测器时，响应面是否含有可由 `x:y` 近似的二阶非加性形状”。
- **PCMCI-CMIknn**：在自然时间序列上先用 PC 式条件筛选控制其他候选父变量，再用基于近邻估计的条件互信息检验 lag-1 source 与 target 是否仍存在条件依赖。图中报告 CMIknn 检验统计量的绝对值。它回答“控制其他观测历史后，这个 source 的上一时刻是否仍为 target 提供预测信息”；它是 pairwise 滞后条件依赖读出，不直接分解二源协同。
- **Whole-minus-sum（WMS）**：直接在自然轨迹的经验联合分布上计算

$$
\operatorname{WMS}(X,Y;Z)=I(\{X,Y\};Z)-I(X;Z)-I(Y;Z).
$$

  正值表示联合源提供的信息超过两个单源信息之和，负值则表示冗余占优。WMS 是净协同减净冗余的汇总量，不是单独的 PID/PEID synergy 原子。
- **MLP+PEID**：先用 MLP 近似动力学机制，再对源变量施加独立最大熵干预，计算单源 EI、联合 EI 及联合源相对单源之和的协同信息。它回答“哪些单源或源集合在干预语义下对目标产生不可约的机制约束”。后文 coupled standard map 的统一数据对比改用自然 test states，因此该处读数是自然轨迹上的 PEID-style residual，而不是正式最大熵干预 PEID。
- **SURD**：直接在自然轨迹的 `(x_t,y_t,z_{t+1})` 上，按原论文方式先用 transport map 估计逐目标状态的 specific MI：

$$
R_{xy}(z)=\min\{i_x(z),i_y(z)\},\quad
U_x(z)=i_x(z)-R_{xy}(z),\quad
U_y(z)=i_y(z)-R_{xy}(z),\quad
S_{xy}(z)=i_{xy}(z)-\max\{i_x(z),i_y(z)\}.
$$

最后对目标状态积分得到 `Rxy/Ux/Uy/Sxy`，满足 `Rxy + Ux + Uy + Sxy = I({x,y};z)`。

# 共同驱动 + sine 协同

这个例子把两种容易混淆的结构放进同一个动力系统：一方面，`w` 是 `x`、`y` 背后的共同原因；另一方面，`x`、`y` 对 `z` 的作用不是两条可分离的 pairwise 边，而是一个二源协同项。系统为

$$
\begin{aligned}
w_{t+1} &= 0.78w_t + \eta^w_t,\\
x_{t+1} &= 0.42x_t + 0.82w_t + \eta^x_t,\\
y_{t+1} &= 0.38y_t + 0.76w_t + \eta^y_t,\\
z_{t+1} &= 0.22z_t + \alpha\sin\left(x_t y_t\right) + \eta^z_t.
\end{aligned}
$$

- pairwise 层面：`w -> x`、`w -> y`；
- 高阶层面：`{x, y} -> z`；
- 非结构边：`w -> z` 不是直接机制边，单独的 `x -> z`、`y -> z` 也只是 sine 协同项的 pairwise 投影。

TODO：这里也要多个seed重复实验

![alpha 扫描下的 SHAP 与 PEID 对照](../../fig/granger_peid_mlp_comparison/sine_alpha_shap_peid_sweep.png)

<img src="../../fig/granger_peid_mlp_comparison/sine_alpha_neural_granger_sweep.png" alt="alpha 扫描下的 Neural Granger 单独读出" width="420">

# 共同驱动增强但结构协同固定

这里固定 `alpha=1`，只改变 `x,y` 的共同驱动强度 `beta`。生成式中 `beta` 增大只让 `x` 与 `y` 在观测轨迹上更相关，并没有增强 `z_{t+1}` 中的 `sin(x_t y_t)` 结构项。因此理论预期是：`{x,y}->z` 的 PEID 协同不应因为 `beta` 增大而单调增加。

动力学为

$$
\begin{aligned}
w_{t+1} &= 0.78w_t + \eta^w_t,\\
x_{t+1} &= 0.42x_t + 0.82\left(\beta w_t + \sqrt{1-\beta^2}\,\xi^x_t\right) + \eta^x_t,\\
y_{t+1} &= 0.38y_t + 0.76\left(\beta w_t + \sqrt{1-\beta^2}\,\xi^y_t\right) + \eta^y_t,\\
z_{t+1} &= 0.22z_t + \sin\left(x_t y_t\right) + \eta^z_t.
\end{aligned}
$$

其中 `beta=0` 时，`x` 与 `y` 主要由各自私有扰动驱动；`beta=1` 时，它们的新增驱动完全共享同一个 `w_t`。`\sqrt{1-\beta^2}` 是 `beta` 的互补私有驱动权重，使共享驱动项和私有驱动项的平方权重和保持为 1；这样 beta 扫描主要改变源变量之间的观测相关性，而不是简单放大或缩小 `x,y` 的总驱动强度。`z` 的结构项始终是同一个 `sin(x_t y_t)`，因此 beta 不改变二源机制本身。

![beta 扫描单边作用曲线](../../fig/granger_peid_mlp_comparison/sine_beta_single_source_readout_sweep.png)

![beta 扫描高阶协同曲线](../../fig/granger_peid_mlp_comparison/sine_beta_synergy_readout_sweep.png)

# Coupled Standard Map Six-Method Comparison

![Six-method comparison](../../fig/coupled_standard_map_method_comparison/coupled_standard_map_six_method_comparison.png)

图中并非没有 PEID 误差：所有曲线都使用 `4` 个独立 seed（`0,1,2,3`）的均值，浅色阴影表示跨 seed 的 `mean ± std`。PEID 阴影大多很窄，且与同色曲线重合，因此视觉上不如 SURD 的阴影明显。PEID joint angle-pair synergy 在 `J=0.2,0.4,0.6,0.8,1.0` 处的跨 seed 标准差分别为 `0.0085,0.0086,0.0124,0.0054,0.0031` bits；其均值随 `J` 严格单调上升，与解析 `J^2/2` 的排序 Spearman 相关为 `1.000`。在所有正耦合 runs 中，`q1+q2` 都是最强 pair，true-pair top rate 为 `1.000`。这些结果说明“正耦合下识别真实角度对及其趋势”对当前 seed 扰动稳定，但 `4` 个 seed 只支持稳定性检查，不能替代置信区间或更大规模重复实验。

双转子 coupled standard map 的冲量方程为

$$
I_{1,t}=K\sin q_{1,t}+J\sin(q_{2,t}-q_{1,t})+\epsilon_{1,t},
$$

$$
I_{2,t}=K\sin q_{2,t}-J\sin(q_{2,t}-q_{1,t})+\epsilon_{2,t}.
$$

状态更新为

$$
p_{i,t+1}=\operatorname{wrap}(p_{i,t}+I_{i,t}),\qquad
q_{i,t+1}=\operatorname{wrap}(q_{i,t}+p_{i,t+1}),\qquad i\in\{1,2\}.
$$

这里的 `wrap` 把任意实数按 $2\pi$ 周期折回基本区间 $[-\pi,\pi)$：

$$
\operatorname{wrap}(a)=((a+\pi)\bmod 2\pi)-\pi.
$$

例如，$\pi+0.1$ 与 $-\pi+0.1$ 表示圆周上的同一点，经过 `wrap` 后使用后者作为状态值。该操作体现角度和动量的周期边界，避免状态随迭代无限漂移；它只改变周期坐标表示，不改变冲量方程中的直接来源关系。

因此动量 `p1,p2` 不直接进入 `I1,I2` 的结构方程。真实二阶来源是 `q1+q2`。对耦合项求混合二阶导数可得

$$
\frac{\partial^2 I_1}{\partial q_1\partial q_2}=J\sin(q_2-q_1),\qquad
\frac{\partial^2 I_2}{\partial q_1\partial q_2}=-J\sin(q_2-q_1).
$$

在均匀角度基准下，`sin^2(q2-q1)` 的平均值为 `1/2`，所以解析 interaction ground truth 为

$$
\mathbb E\left[\left(\frac{\partial^2 I_i}{\partial q_1\partial q_2}\right)^2\right]=\frac{J^2}{2},\qquad i\in\{1,2\}.
$$

上方 ground-truth 曲线图同时画出三个解析基准：Same-rotor angle strength 为 `(K^2+J^2)/2`，Other-rotor angle strength 为 `J^2/2`，joint angle-pair interaction 也是 `J^2/2`；momentum control 的结构真值为 `0`。

直观地说，Same-rotor angle EI 指 `q1->I1` 和 `q2->I2` 这类“本转子角度对本转子冲量”的单源读数；Other-rotor angle EI 指 `q2->I1` 和 `q1->I2` 这类“另一个转子角度对当前冲量”的单源读数。原先的 own/cross 标签分别对应这里的 same-rotor / other-rotor。

## MLP+PEID 与结构真值的逐曲线对照

这里的 MLP+PEID 在自然 test states 上读取拟合 MLP 的 predicted impulses，因此严格说是“自然轨迹访问区域中的 PEID-style residual”，而不是源侧最大熵干预下的正式 PEID；解析 ground truth 则是均匀角度基准下的 squared-derivative strength。两者量纲和分布语义不同，不能要求数值重合，只能检查零结构、排序、单调性和真源识别是否一致。

- **Other-rotor angle EI**：由 `0.0082` bits 随 `J` 上升至 `0.4704` bits，与结构项 $J\sin(q_2-q_1)$ 及解析真值 $J^2/2$ 的增强方向吻合；在 `J=0` 时接近零。
- **Joint angle-pair synergy**：由 `J=0` 时的 `-0.1317` bits 上升至 `J=1` 时的 `0.8514` bits，并与 $J^2/2$ 保持 Spearman `rho=1.000`。它在所有正耦合 runs 中都把 `q1+q2` 排为最强 pair，因而正确恢复了真实二阶来源。负的零耦合残差说明自然轨迹相关性、有限分箱和模型预测面仍会影响绝对值，不能把该曲线直接解释为解析 interaction strength。
- **Momentum control**：动量不直接进入冲量方程，结构真值为零。其 EI 从 `J=0` 时的 `0.0555` bits 很快降至约 `0.002-0.003` bits，正耦合区间基本符合零控制预期；`J=0` 的残差同样提示有限样本与自然轨迹分布效应。
- **Same-rotor angle EI**：从 `2.3309` bits 随 `J` 增大降至 `0.9293` bits，而解析 squared-derivative strength `(K^2+J^2)/2` 应随 `J` 上升。因此这条曲线与结构真值的趋势不吻合。原因是单源 EI 衡量自然轨迹分布上单独观察本转子角度所提供的信息；耦合增强后，目标冲量中由另一个角度及联合角度对携带的信息增加，本转子角度的单源信息份额可以下降。该结果说明自然轨迹 MLP+PEID 能稳定识别耦合相关来源，但不能把每条单源 EI 曲线都当作 squared-Jacobian 强度的代理。

综上，当前实验可靠支持的结论是：MLP+PEID 稳定恢复了正耦合下的真实角度 pair、other-rotor 增强趋势和 momentum 零控制；它不支持“所有 PEID 曲线都与解析 ground truth 一致”的更强结论。
