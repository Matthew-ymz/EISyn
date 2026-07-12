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

随着 `beta` 增大，观测冗余增强，但待比较的二源机制不变。这里保留两个 MLP 口径的对照。第一组把 `w` 当作可观测状态，`MLP+PEID` 与 `SHAP` 用四维一步 MLP
$$
[w_t,x_t,y_t,z_t]\mapsto[w_{t+1},x_{t+1},y_{t+1},z_{t+1}]
$$
读出 `{x,y}->z`；这对应完整观测转移，但会让 MLP 显式看到共同驱动。第二组把 `w` 视为不可观测混杂因子，`MLP+PEID` 与 `SHAP` 只用 `x,y,z` 训练同一个一步 MLP，
$$
[x_t,y_t,z_t]\mapsto[x_{t+1},y_{t+1},z_{t+1}],
$$
然后在所有 `beta` 和 seed 上使用同一批固定干预读出样本读取 `{x,y}->z` 的 PEID 分解和 SHAP interaction，其中 `x,y∈[-1.8,1.8]`，`z∈[-1.25,1.25]`。这样避免 PEID/SHAP 的读出域随 `beta` 的经验分布一起漂移。WMS、MMI-PID 与 SURD 仍直接使用同一条 `x_t,y_t,z_{t+1}` 观测读数；PCMCI、Neural Granger 与 Liang IF 保留原生多变量读数，用来显示它们在观测到 `w` 时对单源方向信息的响应。Oracle+PEID `x+y` 曲线作为固定二源结构参照。

**四维 `wxyz` MLP 口径。** `w` 进入 MLP 训练和预测输入，PEID/SHAP 从完整四维 surrogate 上读出 `{x,y}->z`。

![beta 扫描单源与高阶协同组合曲线：wxyz MLP](../../fig/granger_peid_mlp_comparison/sine_beta_combined_readout_sweep_wxyz_mlp.png)

**三维 `xyz` MLP 口径。** `w` 不进入 MLP 训练或预测输入，只作为隐藏共同驱动；PEID/SHAP 使用固定干预读出域。

![beta 扫描单源与高阶协同组合曲线：xyz MLP 固定读出域](../../fig/granger_peid_mlp_comparison/sine_beta_combined_readout_sweep_xyz_mlp_fixed_support.png)

正式扫描在 `beta∈[0,1]` 上使用步长 `0.05` 的 `21` 个取值，并对每个取值运行 `4` 个 seed（`0,1,2,3`）。图中只绘制跨 seed 均值，以避免密集曲线中的误差棒遮挡趋势；各方法的跨 seed 标准差仍完整保存在结果 JSON 中。四维 `wxyz` MLP 口径下，MLP+PEID synergy 从 `beta=0` 的约 `0.656` bits 降至 `beta=1` 的约 `0.508` bits，线性斜率约 `-0.108` bits / beta；SHAP interaction 从约 `0.180` 增至约 `0.415`，斜率约 `0.409` / beta。三维 `xyz` MLP 固定读出域口径下，MLP+PEID synergy 从约 `0.661` bits 降至约 `0.530` bits，线性斜率约 `-0.111` bits / beta；SHAP interaction 从约 `0.465` 降至约 `0.349`，斜率约 `-0.0437` / beta。两组实验的差别在于 MLP 是否显式观测共同驱动 `w`，以及三维实验是否把 PEID/SHAP 读出域固定为跨 beta 共享的干预支持。

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

# 附录：Kuramoto 振子数与 whole-state $\Phi^{EID}$ 曲线形状的影响

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

其中 $\mathbf{s}_t^i=(\cos\theta_i(t),\sin\theta_i(t))$ 是第 $i$ 个振子的相位特征，$\mathbf{y}_{t+\tau}=\{(\cos\theta_i(t+\tau),\sin\theta_i(t+\tau))\}_{i=1}^{N}$ 是系统整体未来相位状态。这里不再对差值做非负截断；若出现负值，应优先检查 EI 估计、source covariance 或数值正则化，而不是把负值裁掉。

下文首先以 `N=64` Oracle 结果解释临界峰的机制；振子数对照作为系统规模边界证据，统一放在附录末尾。

### 主结果：临界峰的 EI 与 effectiveness 机制

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

### 边界条件与扩展验证

#### 时间窗鲁棒性：避免强同步后，短窗不复现临界内部峰

基准 whole-state 曲线的 `tau=4` 结果保留为主对照。为检验其峰值是否只是高 $K$ 同步饱和造成的，新增一个严格配对的 multi-horizon Oracle sweep：对每个 seed，频率向量、均匀相位 intervention states 和 natural readout states 都固定并复用于全部 $(K,\tau)$ 条件；只改变统一的预测时间窗 $\tau\in\{0.5,0.75,1,1.5,2,4\}$，而不允许 $\tau$ 随 $K$ 自适应变化。所有条件仍使用 `N=64`、3 个 seeds、whole-state future phase target 与同一 N-source transport-map estimator。

![Paired large-N Kuramoto horizon sweep](../../fig/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_sweep.png)

图 A 以未来 target 的 raw global order 的 $99\%$ 分位数 $R_{0.99}$ 审计强同步。预先设定 guard 为：对所有 $K$ 都要求 $R_{0.99}<0.8$。`tau=0.5` 在最强耦合 `K=4` 仍只有 $R_{0.99}=0.583$，完全通过；`tau=0.75` 为 $0.746$，也通过（仅约 $0.37\%$ target samples 的 $R\ge0.8$）。从 `tau=1` 起该 guard 开始失效：`tau=1` 仅 `K=4` 失败，`tau=1.5` 在 `K=3.2,4` 失败，`tau=2` 在 `K\ge2.6` 失败，而 `tau=4` 在 `K\ge2.2` 失败。

关键结果在图 B：**通过 guard 的两个短窗并没有给出与原图相同的临界内部峰。** `tau=0.5` 的 $\Phi^{EID}$ 从 `K=0` 的约 $0$ bits 持续升至 `K=4` 的 $229.69$ bits；`tau=0.75` 同样在 `K=4` 最大，为 $262.20$ bits。因此，在目标尚未进入强同步区的有限短时间内，耦合增强主要表现为 whole-state 联合可预测性的持续增强，而非在 $K_c\approx1.596$ 附近形成回落前的峰。随着时间窗变长，最大值才逐步向低 $K$ 移动：`tau=1` 的峰在 `K=4`（$279.54$ bits），`tau=1.5` 在 `K=3.2`（$281.00$ bits），`tau=2` 在 `K=2.6`（$280.27$ bits），配对的 `tau=4` 在 `K=1.8`（$278.92$ bits），与原 `tau=4` 图中 `K\approx1.7` 的峰一致到扫描分辨率。

因此，原始临界前沿峰的正确表述应收紧为：它是**中等有限观测时间（此处约 $\tau=4$）下**，在高 $K$ 同步吸引已压缩 whole-state 信息后出现的 whole-minus-sum 优势峰；它不是对所有预测时间窗都成立的、时间尺度无关的临界指标。短窗结果同时排除了一个较弱的替代解释：该峰并非仅由高 $K$ target 已完全同步所产生，因为在明确未强同步的 `tau=0.5,0.75` 条件下，曲线反而没有内部峰。

#### 更长时间窗：峰位穿过而非收敛于理论 $K_c$

为直接检验“继续增大 $\tau$ 后，峰是否会停在临界相变点”的假设，保持同一配对 protocol、`N=64`、3 个 seeds 和 full-sample TM estimator，将时间窗扩展为 $\tau\in\{4,6,8,10,12\}$。扫描在转变区加密到 $K=0.8,0.9,\ldots,2.6$，并保留 $K=0,0.4,3.2,4.0$ 锚点，以区分内部峰和扫描端点峰。

![Long-horizon paired large-N Kuramoto sweep](../../fig/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_long_horizon_refined.png)

结果不支持单调收敛后固定在理论 $K_c=1.596$ 的解释。随着 $\tau$ 从 4 增至 12，$\Phi^{EID}$ 的内部峰位依次为 $K_{\rm peak}=1.8,1.6,1.5,1.4,1.3$（峰值分别为 $278.92,274.52,272.83,271.65,269.61$ bits）。因此，`tau=6` 的 $K_{\rm peak}=1.6$ 只是在当前 $0.1$ 网格上恰好贴近 $K_c$；继续增加时间窗后，峰越过 $K_c$ 并持续移向更低的 $K$，而非在 $K_c$ 停留。所有这些峰都是加密区内部点，且其 $R_{0.99}$ 仅为 $0.644,0.561,0.523,0.492,0.492$，strong fraction 均为零；故该左移不是由峰落在高 $K$ 强同步 guard 失效区造成的。

更稳妥的结论是：$K_{\rm peak}(\tau)$ 是有限时间有效信息的时间尺度依赖 crossover，可能在某一中等时间窗掠过临界区，但不能把 $\tau\to\infty$ 的峰位等同于静态 Kuramoto 临界点。长窗极限还可能受相位混合和吸引子压缩控制；若要定义渐近临界指标，需要另行研究固定有限尺寸下的长时间衰减、再做 $N\to\infty$ 的有限尺寸标度，而不能从当前峰位外推。

#### 相变前检测：共同早期弛豫窗中的 $\Phi(\tau)$ 谱

前述长窗峰位不能直接用作预警器。为检验能否在 future target 尚未同步时识别系统的**最终动力学区间**，对全部 $K\in[0,4]$ 保留同一短时间窗，而不是为高 $K$ 自适应延长或截短 horizon。已有的 `tau=0.5,0.75` 结果与新增的 $\tau\in\{0.1,0.2,0.3,0.4,0.6\}$ 配对合并，得到共同谱 $\tau\in\{0.1,0.2,0.3,0.4,0.5,0.6,0.75\}$。所有 $(K,\tau)$ 条件都满足 $R_{0.99}<0.8$；即使在 $K=4$、$\tau=0.75$，$R_{0.99}\approx0.75$，因此该谱只观测初始相位分布向同步吸引子弛豫的早期，而没有把已同步 target 当作特征。

![Pre-transition Kuramoto Phi-tau phase detection](../../fig/classic_network_dynamics_benchmark/large_kuramoto_pretransition_phi_tau_phase_detection.png)

图 B 显示：超临界 $K>K_c$ 条件在整个共同早期窗内已有更陡、更高的 whole-state $\Phi^{EID}(\tau)$ 谱，而此时图 A 证明其 target 尚未发生强同步。以已知的 $K_c=1.596$ 作为模拟中的超临界参考标签，只输入 7 个早期 $\Phi(\tau)$ 值，使用 leave-one-$K$-out（完整留出该 $K$ 的 3 个 seed）逻辑回归，得到超临界识别 AUROC 为 $0.983$。将每一条谱除以自身最大值、仅保留形状后，AUROC 仍为 $0.972$；因此区分力不只是 $\Phi$ 的整体幅度，时间尺度上的增长形状也携带信息。图 C 展示了留出 $K$ 后的预测概率。

##### 识别算法与 AUROC 的计算

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

#### 系统规模边界：只有大系统提供临界峰参照

![Kuramoto oscillator-count appendix](../../fig/part1_kuramoto_size_phi_eid_appendix.png)

该对照只展示 Oracle $\Phi^{EID}$ 与 corrected order，不再混入学习模型读出。**小 $N=2$ classic Kuramoto。** 在相同 whole-state 口径下，Oracle $\Phi^{EID}$ 没有形成清楚的内部临界峰；它在当前扫描范围内主要随强耦合增强，到 `K=4.0` 约为 `0.96` bits。`N=2` 的 corrected order 也不是热力学意义下的相变曲线，而是有限二振子锁相读数。

**大 $N=64$ classic Kuramoto。** 在完全相同的方程、source partition、whole-state target 和 $\Phi^{EID}$ 公式下，corrected global order 从低 $K$ 的近零状态进入高 $K$ 同步饱和区。理论临界耦合为 $K_c\approx1.596$；有限时间读出下最大斜率出现在 `K=2.2`。对应 Oracle N-source $\Phi^{EID}$ 从 `K=0` 的 `0` bits 升高，在 `K=1.7` 达峰，约 `279.63` bits；随后进入强同步区后明显回落，`K=4.0` 约 `13.58` bits。

这个边界对照说明，在方程形式、source/target 和 EI 分解公式都固定后，是否出现临界内部峰主要取决于系统规模。`N=2` 没有经典 Kuramoto 的热力学同步相变，所以不能期待它给出与大系统相同的 $\Phi^{EID}$ 峰；`N=64` 才提供清晰的 order-parameter 转变参照。

因此，Kuramoto 临界相变实验的核心证据链是三步：order parameter 给出同步转变区，whole-state $\Phi^{EID}$ 在转变前沿形成峰值，determinism/degeneracy 分解说明该峰来自“联合相位构型仍可区分、单振子读出快速冗余化”的差异，而不是来自总 EI、determinism 或 degeneracy 任一单项的简单最大化。
