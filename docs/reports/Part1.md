# 对比方法介绍

同一模拟数据用于比较以下方法：

- **Neural Granger**：读取非线性预测器对各历史源变量的依赖强度。
- **SHAP interaction**：衡量两个源变量对 MLP 预测的非加性交互贡献。
- **PCMCI-CMIknn**：检验控制其他历史变量后仍存在的滞后条件依赖。
- **Whole-minus-sum（WMS）**：计算
$$
\operatorname{WMS}(X,Y;Z)=I(\{X,Y\};Z)-I(X;Z)-I(Y;Z).
$$
  正值表示协同占优，负值表示冗余占优。
- **MLP+PEID**：在 MLP 近似的动力学机制上计算干预语义下的不可约联合约束。
- **MMI-PID**：在同一观测上使用
$$
S_{\mathrm{MMI}}(X,Y;Z)=I(\{X,Y\};Z)-\max\{I(X;Z),I(Y;Z)\}.
$$
- **SURD**：按目标状态分解冗余、独有信息与协同：

$$
R_{xy}(z)=\min\{i_x(z),i_y(z)\},\quad
U_x(z)=i_x(z)-R_{xy}(z),\quad
U_y(z)=i_y(z)-R_{xy}(z),\quad
S_{xy}(z)=i_{xy}(z)-\max\{i_x(z),i_y(z)\}.
$$

其中 $S_{xy}$ 为 SURD synergy。



# 共同驱动冗余增强且二源结构协同固定

这里固定 `alpha=1`，用 `beta` 增强 `x,y` 的共同驱动和弱边 `w→z`，同时保持二源结构项 `sin(x_t y_t)` 不变。

动力学为

$$
\begin{aligned}
w_{t+1} &= 0.78w_t + \eta^w_t,\\
x_{t+1} &= 0.42x_t + 0.82\left(\beta w_t + \sqrt{1-\beta^2}\,\xi^x_t\right) + \eta^x_t,\\
y_{t+1} &= 0.38y_t + 0.76\left(\beta w_t + \sqrt{1-\beta^2}\,\xi^y_t\right) + \eta^y_t,\\
z_{t+1} &= 0.22z_t + \sin\left(x_t y_t\right) + 0.15\beta w_t + \eta^z_t.
\end{aligned}
$$

对应的因果结构：

![案例因果图](../../fig/granger_peid_mlp_comparison/causal_graph3.png)

随着 `beta` 增大，观测冗余增强，但待比较的二源机制不变。Oracle+PEID `x+y` 曲线作为固定二源结构参照。

![beta 扫描单源与高阶协同组合曲线](../../fig/granger_peid_mlp_comparison/sine_beta_combined_readout_sweep.png)

正式扫描在 `beta∈[0,1]` 上使用步长 `0.05` 的 `21` 个取值，并对每个取值运行 `4` 个 seed（`0,1,2,3`）。图中只绘制跨 seed 均值，以避免密集曲线中的误差棒遮挡趋势；各方法的跨 seed 标准差仍完整保存在结果 JSON 中。

# 五方法协同比较

每个系统比较 WMS、SURD synergy、SHAP interaction、MLP+PEID synergy 和 MMI-PID synergy。各 panel 内五种方法使用相同的源变量与目标变量；曲线为 `3` 个 seed 的均值，浅色区域表示 `mean ± std`。MI 本身不作为曲线绘制。


![Six-system five-method synergy comparison](../../fig/part1_synergy_comparison/six_system_five_method_synergy_panels.png)



## Coupled Standard Map

**领域背景**：Standard Map 是哈密顿混沌中的经典受冲击转子模型。$K$ 控制单转子非线性，$J$ 控制转子间耦合。

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

**协同源和目标**：计算 `q1+q2->I1`。随着 $J$ 增大，两个角度对第一转子冲量的联合约束增强。SURD 在弱耦合处的大幅波动主要来自周期、多对一映射下的不稳定估计，不宜解释为真实协同。

## Wilson-Cowan Refractory Map

**领域背景**：Wilson-Cowan 模型描述兴奋性与抑制性神经元群体的平均活动。sigmoid gain $g$ 控制群体对净输入的响应陡峭程度。

连续方程写为

$$
\begin{aligned}
\dot E&=-E+(1-\rho E)S\{g(w_{EE}E-w_{EI}I+P_E)\},\\
\dot I&=-I+(1-\rho I)S\{g(w_{IE}E-w_{II}I+P_I)\},
\qquad S(z)=\frac{1}{1+e^{-z}} .
\end{aligned}
$$

本实验扫描 sigmoid gain $g$，其余参数固定为

$$
\rho=0.5,\quad w_{EE}=3.2,\quad w_{EI}=2.6,\quad
w_{IE}=2.4,\quad w_{II}=1.7,\quad P_E=0.35,\quad P_I=-0.20 .
$$

**协同源和目标**：只计算 `E+I->E_tau`。目标映射为

$$
E_{t+\Delta t}
=E_t+\Delta t\left[-E_t+(1-\rho E_t)S\{g(w_{EE}E_t-w_{EI}I_t+P_E)\}\right].
$$

`gain=0` 时 $I_t$ 不影响目标，是结构零点；随着 $g$ 增大，非线性门控变陡，$E_t$ 与 $I_t$ 对目标形成更强的联合约束。

## Kuramoto Active-Rotator Phase Model

**领域背景**：Kuramoto 模型用于研究耦合振荡器的同步与锁相。Active-rotator 扩展加入周期势，$K$ 表示相位差耦合强度。

该 panel 使用经典 active-rotator/Kuramoto 相位动力学：

$$
\begin{aligned}
\dot{\theta}_1&=\omega_1+A\sin\theta_1+K\sin(\theta_2-\theta_1),\\
\dot{\theta}_2&=\omega_2+A\sin\theta_2,
\end{aligned}
\qquad
\omega_1=1.0,\quad \omega_2=0.9,\quad A=0.2.
$$

其中 $A\sin\theta_i$ 是周期相位势，$K\sin(\theta_2-\theta_1)$ 是相位耦合。

**协同源和目标**：所有算法只计算同一条 `theta1+theta2->dtheta1`，即

$$
\{\theta_{1,t},\theta_{2,t}\}\longrightarrow
\dot{\theta}_{1,t}.
$$

目标是第一振子的瞬时速度。随着 $K$ 增大，系统逐渐锁相；WMS 受同步冗余影响转为负值，而 MLP+PEID 保留相位差机制的正协同。SURD 在锁相转变附近波动较大，不宜作定量解释。

## Controlled Hénon Unique-Information Sweep

**领域背景**：Hénon 映射是经典二维耗散混沌模型。这里使用受控 Hénon-style 读出，把显式二源交互项和单源观测通道分开。

读出定义为

$$
\mathbf{z}_{t+1}
=\left[
1-1.4x_t^2+\kappa(\lambda) x_ty_t,\;
\gamma(\lambda) y_t+\epsilon_t
\right],
\qquad
\gamma:0.3\to2.0,\quad \kappa:0.5\to0.1,\quad \sigma_\epsilon=0.5 .
$$

**协同源和目标**：计算 `x+y->z_tau`。扫描参数为 `lambda`，令单源通道 $\gamma(\lambda)y_t+\epsilon_t$ 增强，同时令显式交互项 $\kappa(\lambda)x_ty_t$ 减弱。因此该 panel 用来展示：PEID 可随真实交互减弱而下降，而 MMI-PID 仍会因为弱源单源信息增加而上升；MI 本身仍只作为诊断保存在 JSON 中，不在图中绘制。

## Ikeda Optical Cavity

**领域背景**：Ikeda 映射源自非线性光学环形腔，描述光场在逐次反馈后的演化。$u$ 控制反馈中保留的场幅。

Ikeda 离散映射为

$$
x_{t+1}=1+u(x_t\cos\theta_t-y_t\sin\theta_t),\qquad
y_{t+1}=u(x_t\sin\theta_t+y_t\cos\theta_t),\qquad
\theta_t=0.4-\frac{6}{1+x_t^2+y_t^2}.
$$

**协同源和目标**：只计算 `x+y->y_tau` 这一条一步映射协同读出。

该映射可写为

$$
y_{t+1}=u\,r(x_t,y_t),\qquad
r(x,y)=x\sin\theta(x,y)+y\cos\theta(x,y),
$$

其中 $u$ 主要缩放同一非线性联合响应。因此 `u=0` 是结构零点；`u>0` 后信息协同近似保持平台，而 SHAP interaction 随输出幅值增大。


## Nicholson–Bailey Host–Parasitoid Map

**领域背景**：Nicholson-Bailey 模型描述宿主与寄生蜂的世代更新。$R$ 是宿主繁殖率，$a$ 是寄生蜂攻击效率。

宿主密度 $H_t$ 与寄生蜂密度 $P_t$。离散映射为

$$
H_{t+1}=R H_t e^{-aP_t},\qquad
P_{t+1}=H_t\left(1-e^{-aP_t}\right),\qquad R=1.6.
$$

**协同源和目标**：只计算 `H+P->H_tau`。

当 $a=0$ 时，$P_t$ 不影响目标；当 $a>0$ 时，指数项形成乘性门控。随着攻击效率继续增大，指数响应逐渐饱和，因此信息协同表现为平台而非线性增长。
