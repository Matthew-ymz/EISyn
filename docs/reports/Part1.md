# 对比方法介绍

同一条模拟时间序列先用于训练一个 MLP 一步转移模型，输入为 `[x_t, y_t, z_t, w_t]`，输出为 `[x_{t+1}, y_{t+1}, z_{t+1}, w_{t+1}]`。随后在同一轨迹或固定 MLP 上读出几类量：

- **Granger/ablation**：把某个 source 的输入列替换为均值，记录目标预测 MSE 的增量。它回答“去掉这个变量会不会损害预测”。
- **Neural Granger**：对每个 target 单独训练带 group-lasso 的 cMLP，并读取第一层按 source lag group 聚合的权重范数。它回答“target-wise 非线性预测器是否使用这个 source 的历史输入”，仍是 pairwise 预测结构读出。
- **SHAP 类归因**：在同一 fitted MLP 上只保留一个常用背景替换式 SHAP 基线，用经验背景替换未给定特征。单特征 SHAP 报告 mean absolute attribution；二阶 SHAP interaction 报告 `x:y` 的 mean absolute interaction。前者回答“某个特征分到多少预测贡献”，后者回答“两个特征的非加性预测贡献有多大”。交互项：在同一 fitted MLP 的最大熵干预预测面上，用标准化主效应加一个二阶乘积项拟合目标输出。它回答“固定这个预测器时，响应面是否含有可由 `x:y` 近似的二阶非加性形状”。
- **PCMCI-CMIknn**：在自然时间序列上先用 PC 式条件筛选控制其他候选父变量，再用基于近邻估计的条件互信息检验 lag-1 source 与 target 是否仍存在条件依赖。图中报告 CMIknn 检验统计量的绝对值。它回答“控制其他观测历史后，这个 source 的上一时刻是否仍为 target 提供预测信息”；它是 pairwise 滞后条件依赖读出，不直接分解二源协同。
- **Whole-minus-sum（WMS）**：直接在指定 readout samples 的经验联合分布上计算
$$
\operatorname{WMS}(X,Y;Z)=I(\{X,Y\};Z)-I(X;Z)-I(Y;Z).
$$
  正值表示联合源提供的信息超过两个单源信息之和，负值则表示冗余占优。WMS 是净协同减净冗余的汇总量，不是单独的 PID/PEID synergy 原子。
- **MLP+PEID**：先训练 MLP 近似动力学机制，再在指定 readout states 上计算单源 EI、联合 EI 及协同信息。最大熵干预版本回答“哪些源集合在干预语义下对目标产生不可约的机制约束”。
- **SURD**：直接在指定 readout samples 的 `(x_t,y_t,z_{t+1})` 上，按原论文方式先用 transport map 估计逐目标状态的 specific MI：

$$
R_{xy}(z)=\min\{i_x(z),i_y(z)\},\quad
U_x(z)=i_x(z)-R_{xy}(z),\quad
U_y(z)=i_y(z)-R_{xy}(z),\quad
S_{xy}(z)=i_{xy}(z)-\max\{i_x(z),i_y(z)\}.
$$

最后对目标状态积分得到 `Rxy/Ux/Uy/Sxy`，满足 `Rxy + Ux + Uy + Sxy = I({x,y};z)`。



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

TODO：补充一个因果图

其中 `beta=0` 时，`x` 与 `y` 主要由各自私有扰动驱动；`beta=1` 时，它们的新增驱动完全共享同一个 `w_t`。`\sqrt{1-\beta^2}` 是 `beta` 的互补私有驱动权重，使共享驱动项和私有驱动项的平方权重和保持为 1；这样 beta 扫描主要改变源变量之间的观测相关性，而不是简单放大或缩小 `x,y` 的总驱动强度。`z` 的结构项始终是同一个 `sin(x_t y_t)`，因此 beta 不改变二源机制本身。

![beta 扫描单源与高阶协同组合曲线](../../fig/granger_peid_mlp_comparison/sine_beta_combined_readout_sweep.png)

该 beta 扫描中的 Oracle+PEID 现在不再从每条自然轨迹抽取干预盒，也不随 seed 重采样。它直接在固定支持 `x,y∈[-1.8,1.8]`、`z∈[-1.25,1.25]` 上复用同一批对称独立干预样本，并用真实方程 `z_next=0.22z+sin(xy)` 作为目标。因此 Oracle+PEID 是固定干预协议下的真实方程基准；由于真实结构项不含 `beta`，其高阶协同曲线在所有 beta 上保持 `0.6027` bits，跨 seed 标准差为 `0`，且 `U_x` 与 `U_y` 在对称采样下相同。MLP+PEID 仍使用每条自然轨迹训练出的 fitted MLP 和该轨迹分位数定义的干预样本，用来衡量 learned surrogate 是否保持这一结构不变性。

# 四方法协同读出比较

每个系统比较 WMS、SURD synergy、同一 fitted MLP 上的 SHAP interaction 和 MLP+PEID synergy。Coupled Hénon 使用 `12` 个 seed（`0-11`）；其余五个 Part1 panel 使用 `3` 个 seed（`0,1,2`）。估计方法曲线为 seed 均值，浅色区域为跨 seed 的 `mean ± std`。

TODO：Coupled Hénon也改成只使用3个seed。

统一协议如下：

- Standard Map、Rulkov、Cournot、Ikeda y_tau 和 Nicholson-Bailey 都使用覆盖注册干预域的 broad one-step samples：一批 broad states 训练/验证 MLP，另一批 held-out broad states 同时供 WMS、SURD、SHAP 和 MLP+PEID 读出。
- Coupled Hénon 从完整注册状态盒均匀抽取初值，并对每个初值跟随一次真实映射；训练、验证和测试初值池彼此独立。这是同一 broad one-step 思路在混沌 Hénon 映射上的实现。
- Coupled Hénon 的网络宽度、学习率和权重衰减只按验证集一步预测 NRMSE 选择，不读取 Oracle 或 PEID。冻结配置后才运行 `12` seed PEID，并计算 Oracle 作为事后机制一致性诊断。
- 在同一系统、参数和 seed 下，SHAP 与 MLP+PEID 使用同一个 fitted MLP；JSON 中同时记录 readout-state digest 和 MLP digest，用来审计 SHAP/PEID 是否确实共享数据和模型。
- 正式 MLP+PEID 数值均使用 transport map。Standard Map、Rulkov、Cournot、Ikeda y_tau 和 Nicholson-Bailey 的 PEID states 与 WMS/SURD/SHAP 的 held-out readout states 相同。
- 图中 MLP+PEID 直接报告 transport-map 返回的 TM Syn，不再做 `max(0, TM Syn)` 截断。若估计值为小负数，则保留在图和 JSON 中，解释为有限样本、密度模型或 surrogate 误差诊断，而不是手动投影到非负轴。
- panel a 使用 `symlog` 纵轴，并在图内明确标注；这是为了同时保留 Standard Map 上 SURD 的极端退化估计和约 `0.03-0.18` bits 的 PEID 趋势，不改变任何原始数值。
- 对由扫描参数显式关闭的结构交互，主图也使用同一套生成数据和同一 fitted MLP 的估计值，不再把零耦合点替换为结构真值。`raw_*` 字段仍保留为同值审计列。若零点残差明显大于 Oracle 零映射的近零 TM Syn，则说明 surrogate 在 broad readout 上仍有形状误差，而不是说明真实机制存在协同。

按上述定义，六个 panel 在各自系统内部的算法比较是公平的：同一参数和 seed 只生成一套 broad one-step 训练任务和一套 held-out broad readout 任务，模型型读出不使用额外真实数据，观测型读出也不再停留在自然轨迹池。需要区分的是，六个动力学系统并非只替换方程文本；它们还保留了与系统数值性质匹配的状态域、预测时间尺度、样本量和周期特征处理。因此这里保证的是“系统内方法公平”，而不是“跨系统所有数值超参数完全相同”。



![Six-system four-method synergy comparison](../../fig/part1_synergy_comparison/six_system_four_method_synergy_panels.png)


## Coupled Standard Map

双转子 Coupled Standard Map 的冲量方程为

$$
\begin{aligned}
I_{1,t}&=K\sin q_{1,t}+J\sin(q_{2,t}-q_{1,t}),\\
I_{2,t}&=K\sin q_{2,t}-J\sin(q_{2,t}-q_{1,t}).
\end{aligned}
$$

冲量 $I_i$ 先更新动量，再由更新后的动量推进角度：

$$
\begin{aligned}
p_{i,t+1}&=\operatorname{wrap}\left(p_{i,t}+I_{i,t}\right),\\
q_{i,t+1}&=\operatorname{wrap}\left(q_{i,t}+p_{i,t+1}\right),\\
\operatorname{wrap}(a)&=\left((a+\pi)\bmod 2\pi\right)-\pi .
\end{aligned}
$$

**协同源和目标**：只计算 `q1+q2->I1`。源变量是两个转子角度 `q1,q2`，目标是第一转子的冲量 `I1`。Part1 当前使用无噪声 broad one-step 冲量样本训练和读出；真实动力学中的角度对协同强度应随 `J` 增大而增强。

SURD 的大波动主要是当前 transport-map SURD 估计器在周期系统和退化目标上的方差。单独读取 `I1` 后，`J=0` 的 SURD synergy 为 `1398.0441 ± 110.3588` bits，这是退化确定性目标上的估计失败；`J=0.2` 后回到 `1.2884 ± 0.2169` bits，但仍不稳定追踪解析耦合趋势。相比之下，方程中的结构协同强度按 $J^2/2$ 增长，因此稳定的机制读出应呈随 `J` 增强的趋势。

## Rulkov Neuron Map

该系统是神经科学中常用的快慢变量迭代模型。Rulkov 映射为

$$
\begin{aligned}
x_{t+1}&=\frac{\alpha}{1+x_t^2}+y_t,\\
y_{t+1}&=y_t-\mu(x_t-\sigma),
\qquad \mu=0.001,\quad \sigma=-1 .
\end{aligned}
$$

**协同源和目标**：只计算 `x+y->x_tau`。快变量下一步由非线性项 $\alpha/(1+x_t^2)$ 与慢变量 $y_t$ 共同决定；

TODO：解释这个动力学里为什么存在一部分信息是协同信息，即两个源变量都没办法提供的那部分信息，是不是类似于AND门里的弱协同？另外解释为什么MLP+SHAP的交互项归因会是0

## Coupled Hénon Map

离散时间耦合 Hénon 映射为

$$
\begin{aligned}
x_{t+1}&=(1-\kappa)(1-1.4x_t^2+y_t)+\kappa x_tz_t, & y_{t+1}&=0.3x_t,\\
z_{t+1}&=(1-\kappa)(1-1.4z_t^2+w_t)+\kappa z_tx_t, & w_{t+1}&=0.3z_t.
\end{aligned}
$$

**协同源和目标**：只计算 `x+z->x_tau`。正耦合通过乘积项 $\kappa x_tz_t$ 引入明确的二源机制。

TODO：解释一下为什么WMS和SURD这两个方法，方差会那么大。

## Cournot Duopoly

Cournot 离散迭代模型为

$$
q_{1,t+1}=q_{1,t}+\lambda q_{1,t}(a-c_1-2bq_{1,t}-bq_{2,t}),\qquad
q_{2,t+1}=q_{2,t}+\lambda q_{2,t}(a-c_2-bq_{1,t}-2bq_{2,t}).
$$

**协同源和目标**：只计算 `q1+q2->q1_tau`。

平台来自信息量与结构幅值的区别。$\lambda$ 直接缩放利润梯度中的联合项；当该联合项已主导 $q_{1,t+1}$ 的排序和可分辨结构后，继续放大主要改变输出尺度，并不会等比例增加互信息。

## Ikeda Optical Cavity

Ikeda 离散映射为

$$
x_{t+1}=1+u(x_t\cos\theta_t-y_t\sin\theta_t),\qquad
y_{t+1}=u(x_t\sin\theta_t+y_t\cos\theta_t),\qquad
\theta_t=0.4-\frac{6}{1+x_t^2+y_t^2}.
$$

**协同源和目标**：只计算 `x+y->y_tau` 这一条一步映射协同读出。


## Nicholson–Bailey Host–Parasitoid Map

宿主密度 $H_t$ 与寄生蜂密度 $P_t$。离散映射为

$$
H_{t+1}=R H_t e^{-aP_t},\qquad
P_{t+1}=H_t\left(1-e^{-aP_t}\right),\qquad R=1.6.
$$

**协同源和目标**：只计算 `H+P->H_tau`。

TODO：这些实验中，只有a和c没有平台期，其他实验里MLP+PEID都出现了平台，能否从数学上的角度都推导一下，为什么有的动力学有平台效应有的没有。
