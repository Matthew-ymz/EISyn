# 模块化 Kuramoto 已知动力学中的 Greedy 层级恢复

## 1. 为什么需要这个例子

五源 XOR 例子具有闭式真值，适合检查 EI 计算、层级闭合和代码实现，但它只是算法 sanity check。为了检验 Greedy 层级分解能否对应更常见的动力学机制，这里进一步使用经典 Kuramoto 相位振子模型。Kuramoto 模型描述异质自持振子通过正弦相位差耦合产生同步，常用于研究神经群体、节律细胞、机械振子和电力网络等系统中的集体同步。

本实验不把 Kuramoto 数据离散成逻辑门，而是直接从连续相位动力学生成数据，使用 transport-map EI 估计全部源子集的信息大小，再调用真实数据分析所使用的共享 Greedy 层级实现。

## 2. 已知模块化动力学

考虑六个异质振子：

$$
\dot{\theta}_i
=\omega_i+\sum_{j\ne i}K_{ij}\sin(\theta_j-\theta_i),
\qquad i=1,\ldots,6.
\tag{1}
$$

预先植入两个动力学模块

$$
C_1=\{1,2,3\},\qquad C_2=\{4,5,6\}.
\tag{2}
$$

耦合矩阵定义为

$$
K_{ij}=
\begin{cases}
K_{\mathrm{in}}/2, & i\ne j,\ i,j\in C_m,\\
K_{\mathrm{out}}/3, & i\in C_1,j\in C_2\ \text{或反之},\\
0, & i=j.
\end{cases}
\tag{3}
$$

除数 2 和 3 分别补偿每个振子的群内与群外邻居数量，使 $K_{\mathrm{in}}$ 和 $K_{\mathrm{out}}$ 表示可比较的总耦合预算。固定

$$
K_{\mathrm{in}}=1.5,\qquad
\boldsymbol{\omega}=(-0.55,-0.05,0.42,-0.40,0.12,0.51),
\tag{4}
$$

只扫描

$$
K_{\mathrm{out}}\in\{0,0.25,0.75,1.5\}.
\tag{5}
$$

因此，该实验的处理因素只有跨模块耦合强度。当 $K_{\mathrm{out}}=0$ 时两个模块动力学独立；当 $K_{\mathrm{out}}$ 逐渐增大时出现越来越强的跨模块整合；当 $K_{\mathrm{out}}=K_{\mathrm{in}}$ 时，原先的二社区划分不再具有明显的动力学优势。

## 3. 从动力学生成数据并估计 EI

每个随机种子生成 1200 个独立最大熵相位干预：

$$
\theta_i\stackrel{\mathrm{i.i.d.}}{\sim}\mathrm{Uniform}(-\pi,\pi).
\tag{6}
$$

由式 (1) 直接计算短时相位速度，并加入独立高斯过程噪声：

$$
\mathbf{Y}=\dot{\boldsymbol{\theta}}+\boldsymbol{\varepsilon},
\qquad
\boldsymbol{\varepsilon}\sim\mathcal N(\mathbf 0,0.08^2\mathbf I).
\tag{7}
$$

在足够短的时间步下，$\mathbf{Y}$ 等价于 $\Delta\boldsymbol{\theta}/\Delta t$ 的生成机制读出，同时避免长时间同步吸引子改变干预支持。每个振子使用周期特征

$$
\mathbf{x}_i=(\cos\theta_i,\sin\theta_i)
\tag{8}
$$

作为一个源块。对全部 $2^6-1=63$ 个非空振子子集 $A$，用 degree-2 polynomial triangular transport map 估计

$$
F(A)=EI(\mathbf{x}_A\to\mathbf{Y}).
\tag{9}
$$

degree 2 是必要的，因为

$$
\sin(\theta_j-\theta_i)
=\sin\theta_j\cos\theta_i-\cos\theta_j\sin\theta_i
\tag{10}
$$

在周期特征中是二次交互。所有耦合条件固定样本数、频率、干预支持、transport-map degree、噪声尺度和 Greedy 参数；同一 seed 的相位样本和标准化噪声也在不同 $K_{\mathrm{out}}$ 间配对。

## 4. Greedy 恢复判据

从完整六振子集合开始，根层枚举所有非平凡二分，并选择最大化

$$
\Phi(L)+\Phi(R)
\tag{11}
$$

的分裂。主判据是根层是否精确恢复式 (2) 的 planted 分区：

$$
\{1,2,3\}\mid\{4,5,6\}.
\tag{12}
$$

辅助指标把全部正 Greedy atom 分为两类：

- **群内质量**：atom 的源集合完全落在 $C_1$ 或 $C_2$ 内；
- **跨模块质量**：atom 同时包含两个模块的振子。

这两个指标比要求每条 Kuramoto 边一一对应一个 atom 更合理。标准 Kuramoto 中多个耦合边共享振子，而 Greedy 输出是一棵二叉嵌套树，无法同时无损表示所有相互重叠的边集合。

## 5. 实验结果

![模块化 Kuramoto 动力学中的 Greedy 层级恢复](assets/greedy_hierarchy_kuramoto/validation.png)

*图 1. Greedy 层级分解恢复模块化 Kuramoto 动力学，并随跨模块耦合增强而改变。a，六振子 planted 耦合结构，蓝色为群内耦合，红色为跨群耦合。b，在 $K_{\mathrm{out}}=0$ 的代表种子中，两个最大的原子恰好为 $\{1,2,3\}$ 和 $\{4,5,6\}$；较小的群内 pair atom 反映模块内部的重叠边结构，六振子根残差很小。c，三个配对种子的均值与 SEM；跨群耦合增强时，原子质量由群内逐渐转移到跨模块残差。d，原 planted 根分裂在弱到中等跨群耦合下稳定恢复，但在 $K_{\mathrm{out}}=K_{\mathrm{in}}$ 时消失。图例位于数据区域外。*

结果汇总如下。

| $K_{\mathrm{out}}$ | planted 根分裂 | 群内原子质量 | 跨模块原子质量 | 根层跨模块残差 |
|---:|---:|---:|---:|---:|
| 0.00 | 3/3 | $0.960\pm0.002$ | $0.040\pm0.002$ | $0.145\pm0.009$ bits |
| 0.25 | 3/3 | $0.883\pm0.005$ | $0.117\pm0.005$ | $0.454\pm0.018$ bits |
| 0.75 | 3/3 | $0.777\pm0.006$ | $0.223\pm0.006$ | $0.864\pm0.029$ bits |
| 1.50 | 0/3 | $0.299\pm0.001$ | $0.701\pm0.001$ | $1.273\pm0.019$ bits |

表中误差为三个配对随机种子的 SEM。所有条件的层级闭合误差均低于 $5\times10^{-16}$ bits。

在 $K_{\mathrm{out}}=0$ 的代表种子中，系统级 $\Phi$ 为 3.663 bits，根层首先精确分成两个 planted 模块；96.1% 的正 atom 质量位于模块内部。两个最大 atom 分别为 $\{1,2,3\}$ 的 1.152 bits 和 $\{4,5,6\}$ 的 1.104 bits。随着跨模块耦合增强，根层残差和跨模块质量单调增加，而 planted 根分裂在 $K_{\mathrm{out}}\le0.75$ 时仍保持 3/3 稳定。

当 $K_{\mathrm{out}}=K_{\mathrm{in}}=1.5$ 时，原分区在 3/3 种子中均不再被选中，跨模块质量达到 70.1%。这不是恢复失败，而是式 (3) 中原两模块的动力学可分性已经基本消失；算法随生成机制改变而改变层级，而不是机械保留预设标签。

## 6. Target-shuffle 负对照

保持相位干预、样本量、source 维数、transport-map estimator 和 Greedy 规则不变，只随机打乱 $K_{\mathrm{out}}=0$ 代表数据的联合 target。观测数据的 $\Phi$ 为 3.663 bits，打乱后仅为 0.228 bits；打乱数据的根分裂为五振子对单振子，未恢复 planted 模块，群内 planted 质量为 0。剩余非零值反映有限样本与高维 TM 偏差，因而本实验主要解释观测与 null 的数量级差异和结构差异，而不把 0.228 bits 视为真实整合。

## 7. 能支持什么结论

这个实验支持的最窄结论是：**在经典连续正弦耦合动力学中，当系统确实存在两个弱耦合的振子模块时，基于生成数据估计的 EI 表与 Greedy 层级分解能够恢复 planted 模块；随着跨模块耦合增强，分解会把越来越多的信息质量移动到跨模块残差，并在原模块机制消失时停止恢复原分区。**

它同时给出三个边界：

1. Greedy 恢复的是不重叠的动力学模块，不是所有相互重叠 Kuramoto 边的唯一逐边分解；
2. 结果依赖最大熵相位干预、短时速度 target 和 degree-2 TM 估计口径；
3. 六振子受控系统验证的是机制可恢复性，不代表真实神经或电力网络中的模块必然可辨识。

可复现实验入口为 `scripts/validate_greedy_hierarchy_kuramoto.py`，结构化结果写入 `results/greedy_hierarchy_kuramoto/summary.json`。
