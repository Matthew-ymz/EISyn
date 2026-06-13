# 离散非线性迭代系统的四种协同读出比较

上图直接展示六个系统在多个参数点上的完整四方法实验曲线；下方只列方程、协同源和目标，以及 PEID 结果的简短解释。

![six-system discrete map comparison](../../fig/part1_synergy_comparison/six_system_discrete_iteration_synergy_panels.png)

## 读图口径

- 目标统一为一步映射 `s_t -> s_{t+1}`，不再预测 ODE 导数或 RK4 有限时间流。
- Standard Map、Ikeda、Nicholson-Bailey、Rulkov 和 Cournot 使用覆盖注册干预域的 broad one-step train/readout pools；WMS/SURD/SHAP、MLP+PEID 和 Oracle PEID 共享同一批 held-out broad readout states。
- 同一系统、参数和 seed 下，SHAP 与 PEID 使用同一个 fitted MLP；PEID states 与 WMS/SURD/SHAP readout states 的 digest 在 JSON 中一致。
- Replicator 仍保留 simplex 约束下的专用读出口径；histogram `score` 使用非负条件总相关，`signed_residual` 只作为源侧相关诊断。
- 对由扫描参数显式关闭的结构交互，展示曲线使用同一流程的估计值，不再替换成结构零；`raw_*` 字段保留为同值审计列。
- 曲线为 seeds 的算术均值，阴影为 population standard deviation。
- Standard map panel 使用 `symlog` 纵轴，以免 SURD 极端误差带压扁较小的 PEID 趋势；原始数值没有改变。
- Standard map panel 读取既有结果：`../../results/coupled_standard_map_method_comparison/part1_four_method_synergy.json`。

## 系统说明

### Coupled Standard Map

方程：

$$
I_{1,t}=K\sin q_{1,t}+J\sin(q_{2,t}-q_{1,t}),\quad I_{2,t}=K\sin q_{2,t}-J\sin(q_{2,t}-q_{1,t})
$$
$$
p_{i,t+1}=\operatorname{wrap}(p_{i,t}+I_{i,t}),\quad q_{i,t+1}=\operatorname{wrap}(q_{i,t}+p_{i,t+1})
$$

源和目标：`q1+q2->I1`。

PEID结果：MLP+PEID 从 `coupling=0.0000` 的 `0.0257` 到 `coupling=1.0000` 的 `0.1759`。整体随耦合增强而上升，和角度差耦合项带来的二源机制一致。

### Ikeda

方程：

$$
x_{t+1}=1+u(x_t\cos\theta_t-y_t\sin\theta_t),\;y_{t+1}=u(x_t\sin\theta_t+y_t\cos\theta_t),\;\theta_t=0.4-\frac{6}{1+x_t^2+y_t^2}
$$

源和目标：x+y->x_tau；x+y->y_tau。

PEID结果：MLP+PEID 从 `u=0.0000` 的 `0.0000` 到 `u=0.9000` 的 `0.1492`。读数持续为正是合理的，因为 Ikeda 相位项由 `x,y` 的联合半径和旋转共同决定；但它不必随 `u` 单调，因为信息读数不是幅值计。

### Nicholson-Bailey

方程：

$$
H_{t+1}=RH_t e^{-aP_t},\;P_{t+1}=H_t(1-e^{-aP_t}),\;R=1.6
$$

源和目标：H+P->H_tau。

PEID结果：MLP+PEID 从 `a=0.0000` 的 `0.0102` 到 `a=0.5000` 的 `2.9694`。高攻击率区间转正符合 `H_t e^{-aP_t}` 与 `H_t(1-e^{-aP_t})` 的乘性结构；低攻击率处的正值应主要看作有限样本和 MLP 读出的零点残差。

### Rulkov

方程：

$$
x_{t+1}=\frac{\alpha}{1+x_t^2}+y_t,\;y_{t+1}=y_t-\mu(x_t-\sigma)
$$

源和目标：x+y->x_tau。

PEID结果：MLP+PEID 从 `alpha=0.0000` 的 `0.0113` 到 `alpha=4.3000` 的 `0.4654`。`alpha=0` 时真实映射退化为 `x_tau=y`，Oracle TM Syn 近似为零；当前 broad one-step 训练与共享 broad readout 后，MLP+PEID 的零点残差也接近零。正 `alpha` 区间的读数与 Oracle 接近，对应快变量非线性项与慢变量 `y` 对 `x_tau` 的共同约束。

### Replicator

方程：

$$
x_{i,t+1}=\frac{x_{i,t}\exp(\gamma(Ax_t)_i)}{\sum_j x_{j,t}\exp(\gamma(Ax_t)_j)}
$$

源和目标：x1+x2->x1_tau；x2+x3->x2_tau；x1+x3->x3_tau。

PEID结果：MLP+PEID 从 `gamma=0.0000` 的 `0.9041` 到 `gamma=1.0000` 的 `1.0993`。当前 `score` 为非负条件总相关；旧的负 residual 来自 simplex 约束下源策略频率不独立，不应解释为 PEID 协同为负。

### Cournot

方程：

$$
q_{1,t+1}=q_{1,t}+\lambda q_{1,t}(a-c_1-2bq_{1,t}-bq_{2,t}),\;q_{2,t+1}=q_{2,t}+\lambda q_{2,t}(a-c_2-bq_{1,t}-2bq_{2,t})
$$

源和目标：q1+q2->q1_tau。

PEID结果：MLP+PEID 从 `lambda=0.0000` 的 `0.0094` 到 `lambda=0.2000` 的 `2.4276`。正 `lambda` 后 PEID 迅速升高是合理的，因为每个企业的下一产量都含有自身产量与对方产量共同调制的利润梯度。

## 解释边界

不同方法保留各自原生读数，不能把绝对值直接解释为同一个物理量；这里主要比较零点残差、参数趋势和跨 seed 稳定性。由于新五个系统都是离散一步映射，新报告不把它们与旧 ODE b-f 面板做数值等价声明。
