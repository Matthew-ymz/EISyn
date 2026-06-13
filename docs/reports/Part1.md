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

对应的因果结构：

![案例因果图](../../fig/granger_peid_mlp_comparison/causal_graph2.png)

这样 beta 扫描主要改变源变量之间的观测相关性，而不是简单放大或缩小 `x,y` 的总驱动强度。`z` 的结构项始终是同一个 `sin(x_t y_t)`，因此 beta 不改变二源机制本身。

![beta 扫描单源与高阶协同组合曲线](../../fig/granger_peid_mlp_comparison/sine_beta_combined_readout_sweep.png)

MLP+PEID 仍使用每条自然轨迹训练出的 fitted MLP 和该轨迹分位数定义的干预样本，用来衡量 learned surrogate 是否保持这一结构不变性。

# 四方法协同读出比较

每个系统比较 WMS、SURD synergy、同一 fitted MLP 上的 SHAP interaction 和 MLP+PEID synergy。六个 Part1 panel 现在统一使用 `3` 个 seed（`0,1,2`）。估计方法曲线为 seed 均值，浅色区域为跨 seed 的 `mean ± std`。

统一协议如下：

- Standard Map、Rulkov、Cournot、Ikeda y_tau 和 Nicholson-Bailey 都使用覆盖注册干预域的 broad one-step samples：一批 broad states 训练/验证 MLP，另一批 held-out broad states 同时供 WMS、SURD、SHAP 和 MLP+PEID 读出。
- Coupled Hénon 从完整注册状态盒均匀抽取初值，并对每个初值跟随一次真实映射；训练、验证、测试和 held-out readout 初值池彼此独立但来自同一个 broad state-box 分布。这是同一 broad one-step 思路在混沌 Hénon 映射上的实现。
- Coupled Hénon 的网络宽度、学习率和权重衰减只按验证集一步预测 NRMSE 选择，不读取 Oracle 或 PEID。冻结配置后才运行 `3` seed PEID，并计算 Oracle 作为事后机制一致性诊断；WMS、SURD、SHAP、MLP+PEID 和 Oracle PEID 均复用同一批 held-out broad readout states。
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

**协同源和目标**：只计算 `x+y->x_tau`。快变量下一步由非线性项 $\alpha/(1+x_t^2)$ 与慢变量 $y_t$ 共同决定。这里的协同不是布尔 AND 门那种“两个输入同时为真才打开”的强门控，而是连续加法混合中的联合定位信息。令

$$
g_\alpha(x)=\frac{\alpha}{1+x^2},\qquad x_{t+1}=g_\alpha(x_t)+y_t .
$$

在最大熵干预读出下，单独给定 $x_t$ 只能确定偏移量 $g_\alpha(x_t)$，目标仍被未知的 $y_t$ 平移；单独给定 $y_t$ 只能确定基线，目标仍被未知的 $g_\alpha(x_t)$ 平移。只有联合给定 $(x_t,y_t)$ 时，目标在一维轴上的位置才被同时锁定。因此 PEID 会把一部分 joint EI 分到协同项：这部分信息不是来自显式乘积，而是来自“两个坐标共同消除目标不确定性”。当 $\alpha=0$ 时 $x_{t+1}=y_t$，$x_t$ 对目标没有机制作用，Oracle 与 MLP+PEID 的协同都只剩近零估计残差；当 $\alpha>0$ 后，$x_t$ 的非线性偏移进入目标，联合定位信息迅速出现。

为避免把干预范围差异误读成机制差异，Rulkov 的 readout intervention sampling box 已调整为与 Ikeda y_tau 相同的 $x,y\in[-1.5,1.5]$；Rulkov 原始较宽的状态投影边界仍保留，只用于避免映射更新后的状态被过窄边界裁剪。即便如此，当前 Rulkov panel 仍不能直接解读成“Rulkov 的物理协同显著强于 Ikeda”，因为 Rulkov 目标是近确定性的可加映射

$$
T_\alpha=Y+\alpha f(X).
$$

在无显式目标噪声的连续估计里，只要 $\alpha$ 从 0 变成任意正值，$X$ 就给 $T_\alpha$ 增加一个独立的微小连续扰动；理论互信息会对这种近确定性关系非常敏感，实际 transport-map 读数则受样本量、jitter、特征提升和目标尺度共同决定。因此把 $\alpha$ 网格改到 `0.001-0.1` 这类更小正值，并不一定会让当前 TM Syn 平滑降到 Ikeda 的约 `0.086` bits；在现有无噪连续估计口径下，它仍可能表现为从零点跳到一个由估计器噪声地板决定的平台。若要检验 Rulkov 机制本身的 alpha 强度，应补一个专门的 sensitivity：固定同一干预盒，加入明确目标噪声或使用非退化连续 MI 估计器，并扫描小 alpha 区间，而不是只把主图 alpha 起点改小。

MLP+SHAP interaction 接近零并不矛盾。当前 SHAP interaction 读的是 fitted MLP 响应面中的二阶非加性形状，而 Rulkov 目标对两个源是可加的：

$$
\frac{\partial^2 x_{t+1}}{\partial x_t\,\partial y_t}
=
\frac{\partial}{\partial y_t}
\left(-\frac{2\alpha x_t}{(1+x_t^2)^2}\right)=0 .
$$

如果 MLP 学到的是近似 $g_\alpha(x)+y$ 的可加函数，背景替换式 SHAP 的二阶 inclusion-exclusion 项或标准化乘积探针都应接近零。换言之，PEID 在这里强调“联合知道两个源才能确定目标值”的机制信息，SHAP interaction 强调“函数响应面是否有不可加交互形状”；两者回答的问题不同。

## Coupled Hénon Map

离散时间耦合 Hénon 映射为

$$
\begin{aligned}
x_{t+1}&=(1-\kappa)(1-1.4x_t^2+y_t)+\kappa x_tz_t, & y_{t+1}&=0.3x_t,\\
z_{t+1}&=(1-\kappa)(1-1.4z_t^2+w_t)+\kappa z_tx_t, & w_{t+1}&=0.3z_t.
\end{aligned}
$$

**协同源和目标**：只计算 `x+z->x_tau`。正耦合通过乘积项 $\kappa x_tz_t$ 引入明确的二源机制。修正 readout 协议后，Hénon 的 WMS/SURD 不再使用自然轨迹或吸引子访问样本，而是和 MLP 训练池一样来自注册状态盒上的 broad one-step 分布；同一 seed 下，WMS、SURD、SHAP、MLP+PEID 和 Oracle PEID 还复用同一批 held-out readout states。3-seed 重算后，MLP+PEID 从 `kappa=0` 的 `0.0004 ± 0.0001` bits 增至 `kappa=0.08` 的 `0.0370 ± 0.0030` bits；WMS 和 SURD 的方差也显著小于旧 natural-readout 版本，例如 `kappa=0.04` 时 WMS 为 `0.0157 ± 0.0023` bits，SURD synergy 为 `0.0155 ± 0.0015` bits。

旧版中 WMS/SURD 方差偏大的根因不是 Hénon 机制本身必须产生大方差，而是 readout 分布和 MLP 训练分布不一致：训练使用 broad state-box one-step samples，而 WMS/SURD 使用 held-out natural map states。自然状态会受到有限轨迹访问区域影响，混沌吸引子的折叠、稀疏区和局部相关结构会随 seed 改变。修正后，观测型读出和模型型读出都在同一类 broad held-out states 上比较，吸引子访问分布不再进入主图协议。剩余方差主要来自有限样本信息估计和 MLP 拟合误差；其中 WMS 仍是三个互信息估计的差

$$
I(\{X,Z\};X_\tau)-I(X;X_\tau)-I(Z;X_\tau),
$$

当三个项都由同一有限样本估计时，单项偏差和方差仍会在相减中传递；SURD 还要先估计 target-specific MI，再经过 `min`、`max` 和积分得到 synergy，确定性或近确定性映射中的局部密度误差会通过这些非线性算子传递。因此修正后的解释边界是：当前 Hénon panel 不再检验自然吸引子分布上的协同，而是检验 broad state-box 分布上的一步机制协同。

## Cournot Duopoly

Cournot 离散迭代模型为

$$
q_{1,t+1}=q_{1,t}+\lambda q_{1,t}(a-c_1-2bq_{1,t}-bq_{2,t}),\qquad
q_{2,t+1}=q_{2,t}+\lambda q_{2,t}(a-c_2-bq_{1,t}-2bq_{2,t}).
$$

**协同源和目标**：只计算 `q1+q2->q1_tau`。

平台来自信息量与结构幅值的区别。$\lambda$ 直接缩放利润梯度中的联合项；当该联合项已主导 $q_{1,t+1}$ 的排序和可分辨结构后，继续放大主要改变输出尺度，并不会等比例增加互信息。将目标写成

$$
q_{1,t+1}=q_{1,t}+\lambda q_{1,t}(a-c_1)-2\lambda bq_{1,t}^2-\lambda bq_{1,t}q_{2,t}.
$$

对任意正 $\lambda$，二源项 $-\lambda bq_{1,t}q_{2,t}$ 的符号、排序和水平集形状已经确定；在当前无噪 broad readout 中，PEID 主要读出“联合源是否不可约地约束目标”，而不是读出 $\lambda$ 的线性幅值。因此从 `lambda=0` 到 `0.05` 会出现结构开关式跃迁，但后续 `0.05-0.2` 更多是目标尺度变化和与单源项的相对权重微调，MLP+PEID 进入约 `2.4-2.6` bits 的平台。

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

把六个 panel 放在一起看，是否出现平台取决于参数进入机制的方式。以图中的协同源记为 $X,Y$，目标记为 $T_\theta$，六个系统可以按

$$
T_\theta = A(X)+B(Y)+c(\theta)C(X,Y)
$$

或其轻微推广来比较：

| panel | $X,Y,T_\theta$ | 分解 | 参数进入方式 | 对平台的含义 |
|---|---|---|---|---|
| Standard Map | $X=q_1,\;Y=q_2,\;T_J=I_1$ | $A(X)=K\sin X,\;B(Y)=0,\;c(J)=J,\;C(X,Y)=\sin(Y-X)$ | $J$ 改变角度差项相对单转子项的权重 | 不是单纯缩放目标；联合项占比持续变大，因此无早期平台 |
| Rulkov | $X=x,\;Y=y,\;T_\alpha=x_\tau$ | $A(X)=0,\;B(Y)=Y,\;c(\alpha)=\alpha,\;C(X,Y)=1/(1+X^2)$ | $C$ 实际退化为只依赖 $X$ 的非线性偏移；协同来自联合定位，不来自二阶形状 | $\alpha>0$ 后联合定位结构很快打开，随后主要改变偏移尺度，因此出现平台 |
| Coupled Hénon | $X=x,\;Y=z,\;T_\kappa=x_\tau$ | $T_\kappa=(1-\kappa)A(X;y_t)+\kappa C(X,Y)$，其中 $A(X;y_t)=1-1.4X^2+y_t,\;C(X,Y)=XY$ | $\kappa$ 一边削弱 Hénon 自身分支，一边增强乘积分支 | 结构比例 $\kappa/(1-\kappa)$ 持续改变，因此无早期平台 |
| Cournot | $X=q_1,\;Y=q_2,\;T_\lambda=q_{1,\tau}$ | $T_\lambda=A_\lambda(X)+c(\lambda)C(X,Y)$，其中 $A_\lambda(X)=X+\lambda X(a-c_1-2bX),\;c(\lambda)=\lambda,\;C(X,Y)=-bXY$ | $\lambda$ 同时缩放单源利润项和二源竞争项，但正 $\lambda$ 后符号和水平集形状固定 | 从零到正值是结构开关；之后主要是同类形状的尺度变化，因此平台明显 |
| Ikeda | $X=x,\;Y=y,\;T_u=y_\tau$ | $A=B=0,\;c(u)=u,\;C(X,Y)=X\sin\theta(X,Y)+Y\cos\theta(X,Y)$，$\theta=0.4-6/(1+X^2+Y^2)$ | $u$ 是几乎纯输出尺度，联合形状 $C$ 不随 $u$ 变 | PEID 对可逆尺度缩放不敏感，$u>0$ 后平台最典型 |
| Nicholson-Bailey | $X=H,\;Y=P,\;T_a=H_\tau$ | 精确式：$T_a=RXe^{-aY}$；小 $a$ 展开：$T_a=RX-RaXY+O(a^2Y^2)$ | 低 $a$ 时像乘积项强度 $Ra$；较高 $a$ 时指数门控饱和并压缩高 $Y$ 区域 | 从零点跳到正协同后，目标可分辨结构受指数饱和限制，因此进入平台 |

这个表说明，公式 $A+B+cC$ 只是第一层比较框架；真正决定曲线形状的是 $c(\theta)$ 是否只是固定联合形状的幅值，还是同时改变联合项与背景项的相对权重、甚至改变 $C$ 本身的形状。若 $c(\theta)C(X,Y)$ 在扫描早期已经主导联合可辨识结构，那么继续增大 $c(\theta)$ 主要是对同一目标形状做尺度变换；PEID 对这种可逆尺度变化不敏感，因此出现平台。Rulkov、Cournot、Ikeda 和 Nicholson-Bailey 都属于“结构开启后形状基本固定或很快饱和”的情形。

没有平台的 panel 则是参数仍在改变联合结构相对于其它项的形状。Standard Map 中

$$
I_1=K\sin q_1+J\sin(q_2-q_1)
$$

的 $J$ 持续改变单转子项与角度差项的相对权重，所以协同结构的占比随扫描继续增加。Coupled Hénon 中

$$
x_{t+1}=(1-\kappa)(1-1.4x_t^2+y_t)+\kappa x_tz_t
$$

同时削弱 Hénon 自身项并增强乘积项，比例 $\kappa/(1-\kappa)$ 在当前范围内还没有进入“乘积项完全主导”的区域。因此 panel a 和 c 更像持续的结构混合扫描，而不是单纯的幅值缩放扫描，MLP+PEID 不表现为早期平台。
