# 对比方法介绍

同一条模拟时间序列先用于训练一个 MLP 一步转移模型，输入为 `[x_t, y_t, z_t, w_t]`，输出为 `[x_{t+1}, y_{t+1}, z_{t+1}, w_{t+1}]`。随后在同一轨迹或固定 MLP 上读出几类量：

- **Neural Granger**：对每个 target 单独训练带 group-lasso 的 cMLP，并读取第一层按 source lag group 聚合的权重范数。它回答“target-wise 非线性预测器是否使用这个 source 的历史输入”，仍是 pairwise 预测结构读出。
- **SHAP 类归因**：在同一 fitted MLP 上只保留一个常用背景替换式 SHAP 基线，用经验背景替换未给定特征。单特征 SHAP 报告 mean absolute attribution；二阶 SHAP interaction 报告 `x:y` 的 mean absolute interaction。前者回答“某个特征分到多少预测贡献”，后者回答“两个特征的非加性预测贡献有多大”。交互项：在同一 fitted MLP 的最大熵干预预测面上，用标准化主效应加一个二阶乘积项拟合目标输出。它回答“固定这个预测器时，响应面是否含有可由 `x:y` 近似的二阶非加性形状”。
- **PCMCI-CMIknn**：在自然时间序列上先用 PC 式条件筛选控制其他候选父变量，再用基于近邻估计的条件互信息检验 lag-1 source 与 target 是否仍存在条件依赖。图中报告 CMIknn 检验统计量的绝对值。它回答“控制其他观测历史后，这个 source 的上一时刻是否仍为 target 提供预测信息”；它是 pairwise 滞后条件依赖读出，不直接分解二源协同。
- **Whole-minus-sum（WMS）**：直接在指定 readout samples 的经验联合分布上计算
$$
\operatorname{WMS}(X,Y;Z)=I(\{X,Y\};Z)-I(X;Z)-I(Y;Z).
$$
  正值表示联合源提供的信息超过两个单源信息之和，负值则表示冗余占优。WMS 是净协同减净冗余的汇总量，不是单独的 PID/PEID synergy 原子。
- **MLP+PEID**：先训练 MLP 近似动力学机制，再在指定 readout states 上计算单源 EI、联合 EI 及协同信息。最大熵干预版本回答“哪些源集合在干预语义下对目标产生不可约的机制约束”。
- **SURD**：直接在指定 readout samples 的 `(x_t,y_t,z_{t+1})` 上，按原论文方式使用 transport map 估计逐目标状态的 specific MI：

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

对应的因果结构：

![案例因果图](../../fig/granger_peid_mlp_comparison/causal_graph2.png)

这样 beta 扫描主要改变源变量之间的观测相关性，而不是简单放大或缩小 `x,y` 的总驱动强度。`z` 的结构项始终是同一个 `sin(x_t y_t)`，因此 beta 不改变二源机制本身。

![beta 扫描单源与高阶协同组合曲线](../../fig/granger_peid_mlp_comparison/sine_beta_combined_readout_sweep.png)

MLP+PEID 保持每条自然轨迹训练出的 fitted MLP 完全不变；模型训练完成后，仅把 PEID 的源侧干预支持统一固定为
`x,y∈[-1.8,1.8]`，并独立均匀采样。上下文变量 `z,w` 仍按对应自然轨迹的分位数支持采样。
因此该对照只改变 PEID readout 的源侧支持域，不改变 MLP 训练数据、训练过程或模型参数。

# 四方法协同比较

每个系统比较 WMS、SURD synergy、同一 fitted MLP 上的 SHAP interaction 和 MLP+PEID synergy。六个 panel 现在统一使用 `3` 个 seed（`0,1,2`）。估计方法曲线为 seed 均值，浅色区域为跨 seed 的 `mean ± std`。

按上述定义，每个 panel 内所有算法的源变量和目标变量完全一致。五个 broad one-step panel 共享 held-out readout states；Kuramoto panel 中，MLP 训练与 WMS、SURD、SHAP 使用完全相同的自然轨迹状态和目标，只有 MLP+PEID 与 Oracle PEID 的最终 readout 使用独立最大熵干预分布。这样既控制了模型训练数据差异，也保留了观测同步分布与干预机制分布的语义比较。六个动力学系统仍保留与各自数值性质匹配的状态域、预测时间尺度、样本量和周期特征处理。


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

**协同源和目标**：只计算 `q1+q2->I1`。源变量是两个转子角度 `q1,q2`，目标是第一转子的冲量 `I1`。真实动力学中的角度对协同强度应随 `J` 增大而增强。

SURD 的大波动主要是当前 transport-map SURD 估计器在周期系统和退化目标上的方差。SURD 不是先估计几个全局 MI 再相减，而是对许多目标锚点分别估计 specific MI，然后逐点计算

$$
S_{xy}(z)=i_{xy}(z)-\max\{i_x(z),i_y(z)\}
$$

并积分。在 `J=0` 时，$I_1=K\sin q_1$ 是近确定性、周期、多对一映射；目标条件密度集中在低维曲线上，少数锚点处的密度比误差可以产生极大的 specific MI。逐锚点的非负截断以及强制 $i_{xy}(z)\ge\max\{i_x(z),i_y(z)\}$ 虽保证 SURD 原子非负，却也会把这些正向异常保留下来，不能通过正负误差抵消。PEID 使用同一 transport-map 基础，但读出的是全局单源 EI、联合 EI 及其组合，因此没有经历同样的逐目标排序与单边投影放大。

## Wilson-Cowan Refractory Map

连续方程写为

$$
\begin{aligned}
\dot E&=-E+(1-\rho E)S\{g(w_{EE}E-w_{EI}I+P_E)\},\\
\dot I&=-I+(1-\rho I)S\{g(w_{IE}E-w_{II}I+P_I)\},
\qquad S(z)=\frac{1}{1+e^{-z}} .
\end{aligned}
$$

当前实现用 $\Delta t=0.05$ 的 one-step Euler map 作为离散读出，并把状态投影到 $E,I\in[0,1]$。本实验扫描 sigmoid gain $g$，固定其余参数为

$$
\rho=0.5,\quad w_{EE}=3.2,\quad w_{EI}=2.6,\quad
w_{IE}=2.4,\quad w_{II}=1.7,\quad P_E=0.35,\quad P_I=-0.20 .
$$

**协同源和目标**：只计算 `E+I->E_tau`。目标映射为

$$
E_{t+\Delta t}
=E_t+\Delta t\left[-E_t+(1-\rho E_t)S\{g(w_{EE}E_t-w_{EI}I_t+P_E)\}\right].
$$

这里的交互不是 Rulkov 那种可加定位效应。`gain=0` 时 sigmoid input 退化为常数，`I_t` 不再影响 `E_tau`，因此是清楚的结构零点；正 `gain` 后，sigmoid input 同时含有 $E_t$ 和 $I_t$，再与 refractory factor $(1-\rho E_t)$ 组合，因此 $E_t$ 与 $I_t$ 对 $E_{t+\Delta t}$ 的作用不能写成两个单源函数之和。一个有限差分诊断已加入测试：在 $g=4.0,\rho=0.5$ 的基准点附近，`E_tau` 的二源混合差分显著非零。

`gain=0` 关闭 `I_t` 对 `E_tau` 的机制作用；随着 `gain` 增大，population input 的非线性门控逐步变陡，`E` 与 `I` 的联合响应面被打开。这里的主要用途是替换 Rulkov，提供一个神经动力学中具有显式非可加交互项的 population-level 例子。

## Kuramoto Active-Rotator Phase Model

该 panel 使用经典 active-rotator/Kuramoto 相位动力学：

$$
\begin{aligned}
\dot{\theta}_1&=\omega_1+A\sin\theta_1+K\sin(\theta_2-\theta_1),\\
\dot{\theta}_2&=\omega_2+A\sin\theta_2,
\end{aligned}
\qquad
\omega_1=1.0,\quad \omega_2=0.9,\quad A=0.2.
$$

其中 $A\sin\theta_i$ 是 active rotator 的周期相位势，$K\sin(\theta_2-\theta_1)$ 是标准 Kuramoto 相位耦合。扫描
$K\in\{0,0.05,0.10,0.15,0.20,0.30,0.50\}$；频率失谐 $|\Delta\omega|=0.1$ 仅作为图中的物理参考尺度，在 $A\ne0$ 时不把它宣称为精确锁相阈值。

**协同源和目标**：所有算法只计算同一条 `theta1+theta2->dtheta1`，即

$$
\{\theta_{1,t},\theta_{2,t}\}\longrightarrow
\dot{\theta}_{1,t}.
$$

目标始终是第一相位的瞬时速度，不替换成未来相位，也不与第二振子的目标平均。MLP 训练、WMS、SURD 和 SHAP 使用完全相同的多初值自然轨迹状态及瞬时速度目标；JSON 中以 `train_state_digest == readout_state_digest` 和 `train_target_digest == observed_target_digest` 审计这一约束。MLP+PEID 与 Oracle PEID 的最终 readout 使用 $[-\pi,\pi)^2$ 上相互独立的均匀相位干预。因而 MLP+PEID 衡量的是仅从自然轨迹学习的 surrogate 在未见干预域上的机制泛化能力，而不是混合训练后的域内拟合。

在六系统合并图中，该 panel 使用双纵轴：SURD 单独放在右轴，其余方法保留在左轴。这样不会改变数值，只避免 SURD 在锁相转变附近的较大读数压扁 WMS、SHAP 与 MLP+PEID 的变化。


`K=0.50` 的 WMS 分量进一步说明负值来自冗余而不是负互信息估计：

$$
I(\theta_1;\dot\theta_1)=0.807,\qquad
I(\theta_2;\dot\theta_1)=0.791,\qquad
I(\theta_1,\theta_2;\dot\theta_1)=1.418,
$$

因此 $1.418-0.807-0.791=-0.180$ bits。与此同时，纯自然训练 MLP 在同一批共享自然样本上的拟合 MSE 为 `0.00017`，显著低于均值基线的 `0.01703`；但由于该误差是 in-sample 指标，不能单独证明干预域外泛化。更直接的机制泛化诊断是：在独立均匀干预 readout 上，`K=0.50` 时 MLP+PEID 为 `0.391` bits，Oracle PEID 为 `0.240` bits，二者都保持正值但存在可见 surrogate 偏差。该实验支持的结论是：WMS 是观测分布上的“协同减冗余”有符号净量，不能被解释为非负协同原子；独立最大熵干预下的 PEID 则解除源侧同步冗余，保留相位差机制的不可约联合约束。

SURD 从 `K=0.05` 的 `0.827 ± 0.232` bits 跳到 `K=0.10` 的 `3.199 ± 1.736` bits，在 `K=0.15` 达到 `4.140 ± 2.197` bits，随后回落到约 `1.2-1.9` bits。这个变化发生在 PLV 从弱同步跃迁到接近 1 的区间，但幅度和跨 seed 方差都远大于其他读出，也不随耦合平滑增强，因此不能稳健解释为物理协同峰值。更合理的诊断是：锁相使自然轨迹集中到狭窄相位差流形，target-specific 条件密度估计在转变附近变得病态；SURD 的逐目标 `max` 差分再把局部估计误差传递到非负 synergy。该曲线可以作为“观测分布几何变化的敏感诊断”，但当前 transport-map 数值不足以支持定量趋势结论。

## Coupled Hénon Map

离散时间耦合 Hénon 映射为

$$
\begin{aligned}
x_{t+1}&=(1-\kappa)(1-1.4x_t^2+y_t)+\kappa x_tz_t, & y_{t+1}&=0.3x_t,\\
z_{t+1}&=(1-\kappa)(1-1.4z_t^2+w_t)+\kappa z_tx_t, & w_{t+1}&=0.3z_t.
\end{aligned}
$$

**协同源和目标**：只计算 `x+z->x_tau`。正耦合通过乘积项 $\kappa x_tz_t$ 引入明确二源机制，同时削弱原 Hénon 分支。该 panel 使用 broad one-step states；同一参数和 seed 下，所有方法复用同一批 held-out states，SHAP 与 MLP+PEID 复用同一个 fitted MLP。

Hénon 的信息量读出统一采用每变量 `6` 个等宽 bins，以避免 transport-map specific-MI 在多峰逆映射上的不稳定。主结果中，MLP+PEID 从 `kappa=0` 的 `0.0132 ± 0.0015` bits 增至 `kappa=0.20` 的 `0.0863 ± 0.0029` bits；Oracle PEID 从 `0.0134 ± 0.0018` 增至 `0.0853 ± 0.0019` bits，二者趋势一致。SURD 同样从 `0.0131 ± 0.0009` 增至 `0.0860 ± 0.0006` bits，没有出现 Standard Map transport-map SURD 的极端异常。

分箱敏感性表明绝对值随分辨率上升：`kappa=0.20` 时，`4/6/8` bins 的 MLP+PEID 分别约为 `0.0420/0.0863/0.1342` bits；但三种分辨率均保持随 $\kappa$ 增强的排序，且 WMS、SURD、MLP+PEID 与 Oracle PEID 在同一分箱下彼此接近。因此该 panel 支持“耦合增强带来更强联合约束”的趋势结论，不支持把某个分箱下的绝对 bits 当作分辨率无关常数。

## Ikeda Optical Cavity

Ikeda 离散映射为

$$
x_{t+1}=1+u(x_t\cos\theta_t-y_t\sin\theta_t),\qquad
y_{t+1}=u(x_t\sin\theta_t+y_t\cos\theta_t),\qquad
\theta_t=0.4-\frac{6}{1+x_t^2+y_t^2}.
$$

**协同源和目标**：只计算 `x+y->y_tau` 这一条一步映射协同读出。

Ikeda panel 的平台更接近尺度不变性。因为

$$
y_{t+1}=u\,r(x_t,y_t),\qquad
r(x,y)=x\sin\theta(x,y)+y\cos\theta(x,y),
$$

且 $\theta(x,y)$ 本身由 $x^2+y^2$ 共同决定。只要 $u>0$，从机制形状看，$u$ 只是对同一个非线性联合响应 $r(x,y)$ 做正比例缩放；互信息和 PEID 对目标的可逆缩放不应线性敏感。因此 `u=0` 是关闭映射的零点，`u>0` 后机制形状立即存在，MLP+PEID 在约 `0.086` bits 附近保持平台。相反，SHAP interaction 随 `u` 增大继续上升，是因为它读的是预测输出幅值中的二阶响应面差分，而不是 scale-invariant 的机制信息。


## Nicholson–Bailey Host–Parasitoid Map

宿主密度 $H_t$ 与寄生蜂密度 $P_t$。离散映射为

$$
H_{t+1}=R H_t e^{-aP_t},\qquad
P_{t+1}=H_t\left(1-e^{-aP_t}\right),\qquad R=1.6.
$$

**协同源和目标**：只计算 `H+P->H_tau`。

Nicholson-Bailey 的平台也来自参数主要改变尺度或饱和程度，而不是持续引入新形状：

$$
H_{t+1}=R H_t e^{-aP_t}.
$$

当 $a=0$ 时，目标退化为 $RH_t$，$P_t$ 没有机制作用；当 $a>0$ 时，$P_t$ 通过指数项门控 $H_t$，联合源立即形成乘性结构。随着 $a$ 继续增大，$e^{-aP_t}$ 在当前正支持上快速压低高 $P_t$ 区域，目标的可分辨变化集中到低 $P_t$ 带；高 $P_t$ 区域之间的差异被指数饱和压缩。于是 PEID 从零点跳到约 `3` bits 后，不会随 $a$ 线性增长。
