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

这里固定 `alpha=1`，用 `beta` 增强 `x,y` 的共同驱动和直接边 `w→z`，同时保持二源结构项 `sin(x_t y_t)` 不变。为避免系数呈现出调参痕迹，结构系数只取 `1` 或 `0.5`：自回归项统一设为 `0.5`，`w→x/y` 的载荷统一设为 `1`，并消除原方程中 `x/y` 的人为不对称。

动力学为

$$
\begin{aligned}
w_{t+1} &= 0.5w_t + \eta^w_t,\\
x_{t+1} &= 0.5x_t + \left(\beta w_t + \sqrt{1-\beta^2}\,\xi^x_t\right) + \eta^x_t,\\
y_{t+1} &= 0.5y_t + \left(\beta w_t + \sqrt{1-\beta^2}\,\xi^y_t\right) + \eta^y_t,\\
z_{t+1} &= 0.5z_t + \sin\left(x_t y_t\right) + 0.5\beta w_t + \eta^z_t.
\end{aligned}
$$

对应的因果结构：

![案例因果图](../../fig/granger_peid_mlp_comparison/causal_graph3.png)

随着 `beta` 增大，`w` 在 `x,y` 中形成更强的共同成分，但待比较的 `sin(x_ty_t)` 二源机制不变。正式扫描在 `beta∈[0,1]` 上使用步长 `0.05` 的 `21` 个取值，并对每个取值运行 `4` 个 seed（`0,1,2,3`）。除是否向 MLP 提供 `w_t` 外，轨迹、时间切分、训练轮数、优化器、标准化和评估协议均保持一致。

## 三维 MLP：隐藏共同驱动下的主结果

正文采用更接近隐混杂场景的三维 MLP：`w` 不进入训练或预测输入，模型只学习

$$
[x_t,y_t,z_t]\mapsto[x_{t+1},y_{t+1},z_{t+1}].
$$

`MLP+PEID` 与 `SHAP` 共享这个 fitted MLP，并在所有 `beta` 和 seed 上使用同一批固定干预读出样本，其中 `x,y∈[-1.8,1.8]`，`z∈[-1.25,1.25]`。固定读出域可避免 PEID 和 SHAP 随自然轨迹分布一起漂移。WMS、MMI-PID 与 SURD 仍直接使用同一条 `x_t,y_t,z_{t+1}` 观测读数；PCMCI、Neural Granger 与 Liang IF 保留各自的原生读数。Oracle+PEID 则在固定支持上直接评估真实方程，作为不随 `beta` 改变的二源结构参照。

![简化系数方程下的三维 MLP beta 扫描对比曲线](../../fig/granger_peid_mlp_comparison/sine_beta_simple_coefficients_3d_mlp.png)

图 a 比较 `x→z` 与 `y→z` 的单源读数，图 b 比较 `{x,y}→z` 的二源协同。曲线为 `4` 个 seeds 的均值；跨 seed 标准差保存在结果数据中，但未叠加到这张密集多方法图上，以免遮挡曲线。随着 `beta` 从 `0` 增至 `1`，自然轨迹中的 `corr(x,y)` 从 `0.014` 增至 `0.880`，observational WMS 从 `0.308` 降至 `-0.181` bits，说明观测分布由协同占优转为冗余占优。与此同时，固定真实结构的 Oracle+PEID synergy 始终为 `0.465` bits，beta 斜率的绝对值小于 `10^{-15}`。

三维 MLP 的读数没有把这一观测冗余变化误写成真实结构增强：MLP+PEID synergy 从 `0.560` bits 降至 `0.409` bits，线性斜率为 `-0.140` bits / beta（bootstrap 95% CI `[-0.175,-0.106]`）；SHAP interaction 从 `0.504` 降至 `0.383`，斜率为 `-0.0795` / beta。两者都没有随共同驱动增强而上升，但它们的绝对量纲不同：前者是固定干预域上的信息分解，后者是预测响应的背景替换式交互归因。

## 四维完整状态的敏感性检查

作为完整观测对照，第二组把 `w` 纳入一步 MLP：

$$
[w_t,x_t,y_t,z_t]\mapsto[w_{t+1},x_{t+1},y_{t+1},z_{t+1}]
$$

![beta 扫描单源与高阶协同组合曲线：wxyz MLP](../../fig/granger_peid_mlp_comparison/sine_beta_simple_coefficients_wxyz_mlp.png)

四维 `wxyz` MLP 口径下，MLP+PEID synergy 从 `beta=0` 的 `0.607` bits 降至 `beta=1` 的 `0.424` bits，线性斜率为 `-0.177` bits / beta（bootstrap 95% CI `[-0.220,-0.137]`）；SHAP interaction 从 `0.259` 增至 `0.525`，斜率为 `0.331` / beta。SHAP 在三维和四维口径下方向不同，说明它对模型是否显式条件化于共同驱动较敏感；MLP+PEID 在两个口径下均随 `beta` 下降，方向更稳定。

## 三维与四维 MLP 的配对训练对比

为单独检验隐藏共同驱动对预测训练的影响，进一步使用同一条轨迹、同一个 seed 和同一个 80/20 时间切分，训练两个只预测标量 `z_{t+1}` 的 MLP。三维模型输入为 `[x_t,y_t,z_t]`，四维模型输入为 `[x_t,y_t,z_t,w_t]`；两者均使用两层 32-unit `tanh` 网络、90 epochs、相同优化器和相同标准化协议。因此，唯一处理因素是模型是否观察 `w_t`。

![三维与四维 MLP 的 beta 配对预测对比](../../fig/granger_peid_mlp_comparison/sine_beta_3d_vs_4d_mlp_forecast.png)

图中曲线为 `4` 个 seeds 的均值，阴影为 `mean ± std`；下图直接给出每个 beta 上的配对差值 `R²(3D)-R²(4D)`。在 `beta=0` 时，两者几乎等价：三维和四维测试 `R²` 分别为 `0.966` 和 `0.964`，配对差仅为 `0.002`。随着共同驱动增强，四维模型保持稳定，测试 `R²` 的 beta 斜率仅为 `0.002`；三维模型则以 `-0.149` / beta 的斜率下降，在 `beta=1` 时达到 `0.823`，而四维模型仍为 `0.964`。全部 `84` 个 `beta × seed` 配对中，四维模型在 `78` 个配对上更优；总体平均 `R²(3D)-R²(4D)=-0.054`，bootstrap 95% CI 为 `[-0.065,-0.044]`。四个 seed 的配对差 beta 斜率均为负，说明随着 `w` 的作用增强，隐藏 `w` 带来的预测损失是稳定的，而不是由单个随机种子造成。

这个结果同时限定了前述三维 MLP+PEID 曲线的解释：三维模型在整个扫描内仍保持较高测试 `R²`（最低单次运行约 `0.786`），因此不是完全失效；但高 beta 下它学习的是边际化后的 `[x,y,z]` 转移，无法像四维模型一样直接条件化于 `w_t`。不可避免的容量差异是四维模型首层多出 `32` 个输入权重，总参数量由 `1217` 增至 `1249`（约 `2.6%`）；其余隐藏层、输出层和训练预算完全相同。

综合三维主结果、四维敏感性检查和配对预测实验，简化系数后仍得到同一结论：共同驱动增强会增加观测冗余，但不会改变固定的 `sin(x_ty_t)` 二源结构。原小数系数方程及其完整曲线移至附录 A，用作系数替换前的稳健性对照。

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

目标是第一振子的瞬时速度。该 sweep 现在覆盖
$K\in\{0,0.05,0.1,0.15,0.2,0.3,0.5,0.75,1.0,1.5,2.0\}$，
使参数范围从锁相转变区延伸到高耦合的有序同步相。

随着 $K$ 增大，系统逐渐锁相；WMS 受同步冗余影响转为负值，而 MLP+PEID 保留相位差机制的正协同。SURD 在锁相转变附近波动较大，不宜作定量解释。

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

# 附录 A：原小数系数共同驱动实验

本附录保留系数简化前的共同驱动实验。该版本固定 `alpha=1`，扫描设置、seeds、样本量、噪声、MLP 预算、PCMCI 设置、TM 估计器、干预支持和 Oracle 样本均与正文实验一致，但动力学使用原始小数系数：

$$
\begin{aligned}
w_{t+1} &= 0.78w_t + \eta^w_t,\\
x_{t+1} &= 0.42x_t + 0.82\left(\beta w_t + \sqrt{1-\beta^2}\,\xi^x_t\right) + \eta^x_t,\\
y_{t+1} &= 0.38y_t + 0.76\left(\beta w_t + \sqrt{1-\beta^2}\,\xi^y_t\right) + \eta^y_t,\\
z_{t+1} &= 0.22z_t + \sin\left(x_t y_t\right) + 0.15\beta w_t + \eta^z_t.
\end{aligned}
$$

## A.1 四维 `wxyz` MLP 结果

![原小数系数下的四维 MLP beta 扫描](../../fig/granger_peid_mlp_comparison/sine_beta_combined_readout_sweep_wxyz_mlp.png)

在原小数系数下，四维 MLP+PEID synergy 从 `beta=0` 的 `0.656` bits 降至 `beta=1` 的 `0.508` bits，线性斜率为 `-0.108` bits / beta（bootstrap 95% CI `[-0.151,-0.068]`）；SHAP interaction 从 `0.180` 增至 `0.415`，斜率为 `0.409` / beta（bootstrap 95% CI `[0.357,0.460]`）。因此，四维口径下 SHAP 对共同驱动的代理作用较敏感，而 MLP+PEID 没有把共同驱动增强解释成二源结构增强。

## A.2 三维 `xyz` MLP 结果

![原小数系数下的三维 MLP beta 扫描](../../fig/granger_peid_mlp_comparison/sine_beta_combined_readout_sweep_xyz_mlp_fixed_support.png)

原小数系数的三维 MLP 使用与正文相同的隐藏 `w` 和固定干预读出协议。MLP+PEID synergy 从 `0.661` bits 降至 `0.530` bits，线性斜率为 `-0.111` bits / beta（bootstrap 95% CI `[-0.146,-0.077]`）；SHAP interaction 从 `0.465` 降至 `0.349`，斜率为 `-0.0437` / beta（bootstrap 95% CI `[-0.0723,-0.0198]`）。三维 MLP 的两种读数均未随共同驱动增强而上升。

## A.3 与简化系数版本的结论一致性

原小数系数下，`corr(x,y)` 从 `0.014` 增至 `0.905`，observational WMS 从 `0.330` 降至 `-0.0868` bits，MMI-PID synergy 从 `0.335` 降至 `0.0256` bits；固定真实结构的 Oracle+PEID synergy 始终为 `0.603` bits。与正文的 `1/0.5` 系数版本做严格配对比较后，`corr(x,y)` 的 beta 斜率在两组中均为正，observational WMS 与 MMI-PID synergy 的斜率在两组中均为负，而且三个方向都达到 `4/4` seeds 一致。

因此，系数简化改变了部分读数的绝对值，但没有改变核心结论：共同驱动增强观测冗余，而固定的 `sin(x_ty_t)` 二源结构不随 `beta` 改变。正文采用 `1/0.5` 系数版本，是为了使系统定义更简洁、对称，并降低读者将结果误解为精细调参产物的风险。
