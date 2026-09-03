# Method

本文说明有效信息、三变量 PID 退化关系、联合有效信息增量及其层级分解方法。本文将这套递归层级分解算法正式命名为 **Synergy Partition Tree（SPT）**。这里仅保留适用于一般离散或连续动力系统的定义与算法，不涉及具体数据集、模型架构或实验配置。

## 1. 有效信息（EI）

考虑离散时间马尔可夫动力系统 [P1]：

$$
\mathbf{X}_{t+1}\sim p(\mathbf{x}_{t+1}\mid\mathbf{x}_t),
\qquad
\mathbf{X}_t=\bigl(X_t^{(1)},\ldots,X_t^{(n)}\bigr),
\tag{M.1}
$$

其中，$X_t^{(i)}$ 表示时刻 $t$ 的第 $i$ 个微观变量。有效信息不直接使用系统自然运行时的经验输入分布，而是在时刻 $t$ 对源变量施加最大熵干预，得到干预分布 $q^{\max}(\mathbf{x}_t)$。对于有限离散状态空间，最大熵分布就是输入状态空间上的均匀分布，因此干预可写为

$$
\operatorname{do}\!\left(
\mathbf{X}_t\sim\mathcal{U}(\Omega_{\mathbf{X}_t})
\right),
\tag{M.2}
$$

其中，$\Omega_{\mathbf{X}_t}$ 是源变量的输入状态空间。干预后的联合分布由最大熵输入分布与系统转移机制共同确定：

$$
q(\mathbf{x}_t,\mathbf{x}_{t+1})
=q^{\max}(\mathbf{x}_t)
p(\mathbf{x}_{t+1}\mid\mathbf{x}_t).
\tag{M.3}
$$

令 $A\subseteq\{1,\ldots,n\}$ 表示源变量的指标子集，对应的联合源状态记为 $\mathbf{X}_t^A$；目标固定为下一时刻的完整系统状态 $\mathbf{X}_{t+1}$。从 $\mathbf{X}_t^A$ 到 $\mathbf{X}_{t+1}$ 的有效信息定义为最大熵干预分布下的互信息：

$$
EI\!\left(\mathbf{X}_t^A\to\mathbf{X}_{t+1}\right)
\equiv
I_{q^{\max}}\!\left(\mathbf{X}_t^A;\mathbf{X}_{t+1}\right).
\tag{M.4}
$$

这一量衡量的是：当源状态被等概率地主动设置后，系统转移机制能够使未来目标状态保留多少关于干预源状态的信息。最大熵干预消除了自然输入频率不均衡带来的影响，使 EI 主要反映给定干预口径下的动力学转移机制，而不是观测数据中某些状态更常出现这一事实。

对于离散机制，EI 可以由转移概率矩阵显式计算。设源状态空间和目标状态空间分别为

$$
\Omega_{\mathbf{X}_t^A}=\{\mathbf{x}_1,\ldots,\mathbf{x}_M\},
\qquad
\Omega_{\mathbf{X}_{t+1}}=\{\mathbf{x}_1^+,\ldots,\mathbf{x}_L^+\},
$$

并定义

$$
\mathbf{P}_{\mathbf{X}_t^A\to\mathbf{X}_{t+1}}
=[p_{ij}]_{M\times L},
\qquad
p_{ij}=P\!\left(
\mathbf{X}_{t+1}=\mathbf{x}_j^+
\mid
\mathbf{X}_t^A=\mathbf{x}_i
\right).
\tag{M.5}
$$

在源侧均匀干预下，EI 为

$$
EI\!\left(\mathbf{P}_{\mathbf{X}_t^A\to\mathbf{X}_{t+1}}\right)
=\frac{1}{M}
\sum_{i=1}^{M}\sum_{j=1}^{L}
p_{ij}
\log\!\left(
\frac{M p_{ij}}{\sum_{k=1}^{M}p_{kj}}
\right).
\tag{M.6}
$$

当对数以 2 为底时，EI 的单位为 bit。对于连续系统，若变量没有支持集或矩约束，通常不存在无界空间上的均匀最大熵分布，因此必须明确干预约束。常见做法是在有界支持集上采用均匀干预，或在给定矩约束下选择相应的最大熵分布；无论采用哪种约束，均应保持“最大熵输入—机制转移—互信息”的定义链一致。

## 2. 三变量情形：从 PID 到联合 EI 增量

先考虑两个源变量 $X_t^{(1)}$、$X_t^{(2)}$ 和完整目标状态 $\mathbf{X}_{t+1}$ 构成的三变量情形。源侧最大熵独立干预使两个源变量的干预分布分解为

$$
q^{\max}\!\left(x_t^{(1)},x_t^{(2)}\right)
=q^{\max}\!\left(x_t^{(1)}\right)
q^{\max}\!\left(x_t^{(2)}\right),
\qquad
X_t^{(1)}\perp X_t^{(2)}.
\tag{M.7}
$$

在原论文 [P1] 采用的 PID 构造中，这一独立性使冗余信息项退化为零。于是，两个单源的唯一信息分别等于相应的单源 EI，而协同信息等于联合 EI 超出两个单源 EI 之和的部分。

**命题 1（三变量情形）.** 设源变量为 $\{X_t^{(1)},X_t^{(2)}\}$，目标为完整系统状态 $\mathbf{X}_{t+1}$。在使两个源变量相互独立的干预下，PID 冗余项为零，并且

$$
\begin{aligned}
Un\!\left(X_t^{(1)};\mathbf{X}_{t+1}\mid X_t^{(2)}\right)
&=EI\!\left(X_t^{(1)}\to\mathbf{X}_{t+1}\right),\\
Un\!\left(X_t^{(2)};\mathbf{X}_{t+1}\mid X_t^{(1)}\right)
&=EI\!\left(X_t^{(2)}\to\mathbf{X}_{t+1}\right),\\
Syn\!\left(X_t^{(1)},X_t^{(2)};\mathbf{X}_{t+1}\right)
&=EI\!\left((X_t^{(1)},X_t^{(2)})\to\mathbf{X}_{t+1}\right)
-EI\!\left(X_t^{(1)}\to\mathbf{X}_{t+1}\right)
-EI\!\left(X_t^{(2)}\to\mathbf{X}_{t+1}\right).
\end{aligned}
\tag{M.8}
$$

该结论同样适用于任意目标随机变量 $T$，完整证明见附录 A。特别地，把目标取为同一动力学系统在预测尺度 $\ell$ 下的完整状态 $\mathbf{X}_{t+\ell}$ 后，式（M.8）中的协同项正是二源联合 EI 增量。因此，三变量 PID 退化关系给出了下一节一般源集合定义的二源起点。

## 3. 联合有效信息增量

本文以 $\Xi$ 表示基于 PEID 定义的联合有效信息增量，并沿用式（M.1）的动力学记号。令 $C$ 表示从完整状态 $\mathbf{X}_t$ 中选取的一个联合源变量，目标固定为同一系统在预测尺度 $\ell$ 下的完整状态 $\mathbf{X}_{t+\ell}$。最细粒度划分把 $C$ 拆成其中包含的各个单源变量。在这一划分下，联合有效信息增量定义为

$$
\Xi_{\ell}(C;\mathbf{X}_{t+\ell})
=
EI(C\to\mathbf{X}_{t+\ell})
-\sum_{X_t^{(i)}\in C}EI(X_t^{(i)}\to\mathbf{X}_{t+\ell}).
\tag{M.9}
$$

该量刻画联合源变量的 EI 超出其中全部单源 EI 之和的部分。特别地，当 $C=(X_t^{(i)},X_t^{(j)})$ 只含两个源变量时，由第 2 节的三变量关系可得

$$
\begin{aligned}
\Xi_{\ell}\!\left((X_t^{(i)},X_t^{(j)});\mathbf{X}_{t+\ell}\right)
&=Syn^{\mathrm{EID}}\!\left(X_t^{(i)},X_t^{(j)};\mathbf{X}_{t+\ell}\right)\\
&=EI\!\left((X_t^{(i)},X_t^{(j)})\to\mathbf{X}_{t+\ell}\right)
-EI(X_t^{(i)}\to\mathbf{X}_{t+\ell})
-EI(X_t^{(j)}\to\mathbf{X}_{t+\ell}).
\end{aligned}
\tag{M.10}
$$

因此，二源变量的 $\Xi$ 恰好就是协同信息。以完整状态 $\mathbf{X}_t$ 为源时，系统整体的联合有效信息增量为 $\Xi_{\ell}(\mathbf{X}_t;\mathbf{X}_{t+\ell})$。当 $C=X_t^{(i)}$ 只含一个源变量时，式（M.9）中的联合 EI 与单源 EI 相互抵消，因此 $\Xi_{\ell}(X_t^{(i)};\mathbf{X}_{t+\ell})=0$。

除最细粒度划分外，层级分解只使用二分。设 $L$ 和 $R$ 是由 $C$ 拆出的两个非空联合源变量，二者不包含重复的单源变量，并满足 $C=(L,R)$。它们关于目标 $\mathbf{X}_{t+\ell}$ 的协同信息定义为

$$
Syn^{\mathrm{EID}}\!\left(L,R;\mathbf{X}_{t+\ell}\right)
=EI(C\to\mathbf{X}_{t+\ell})
-EI(L\to\mathbf{X}_{t+\ell})
-EI(R\to\mathbf{X}_{t+\ell}).
\tag{M.11}
$$

该量衡量一次二分后仍需联合读取 $L$ 和 $R$ 才能保留的信息。

### 3.1 定理 1：源侧协同的非负性

**定理 1（源侧协同的非负性）.** 在离散情形的源侧最大熵独立干预下，对于从 $\mathbf{X}_t$ 中选取的任意非空联合源变量 $C$ 和完整目标状态 $\mathbf{X}_{t+\ell}$，都有

$$
0\le
\Xi_{\ell}(C;\mathbf{X}_{t+\ell})
\le
EI(C\to\mathbf{X}_{t+\ell}).
\tag{M.12}
$$

对于 $C$ 的任意二分 $(L,R)$，还都有

$$
0\le
Syn^{\mathrm{EID}}\!\left(L,R;\mathbf{X}_{t+\ell}\right)
\le
EI(C\to\mathbf{X}_{t+\ell}).
\tag{M.13}
$$

**证明.** 源侧最大熵独立干预使 $C$ 中的各单源变量相互独立。根据式（M.9）展开互信息并消去无条件熵，可得

$$
\begin{aligned}
\Xi_{\ell}(C;\mathbf{X}_{t+\ell})
&=\sum_{X_t^{(i)}\in C}H_{q^{\max}}(X_t^{(i)}\mid\mathbf{X}_{t+\ell})
-H_{q^{\max}}(C\mid\mathbf{X}_{t+\ell})\\
&=TC_{q^{\max}}\!\left(C\mid\mathbf{X}_{t+\ell}\right)\ge0.
\end{aligned}
\tag{M.14}
$$

同理，二分协同可写为条件互信息：

$$
Syn^{\mathrm{EID}}\!\left(L,R;\mathbf{X}_{t+\ell}\right)
=I_{q^{\max}}\!\left(
L;R
\mid\mathbf{X}_{t+\ell}
\right)\ge0.
\tag{M.15}
$$

条件总相关和条件互信息均非负，因此两个下界成立。式（M.9）和式（M.11）分别从联合 EI 中减去若干非负的 EI，所以上界也成立。$\square$

### 3.2 定理 2：二分层级恒等式

**定理 2（二分层级恒等式）.** 对 $C$ 的任意二分 $(L,R)$，最细粒度增量满足

$$
\Xi_{\ell}(C;\mathbf{X}_{t+\ell})
=\Xi_{\ell}(L;\mathbf{X}_{t+\ell})
+\Xi_{\ell}(R;\mathbf{X}_{t+\ell})
+Syn^{\mathrm{EID}}\!\left(L,R;\mathbf{X}_{t+\ell}\right).
\tag{M.16}
$$

**证明.** 将式（M.9）分别用于 $C$、$L$ 和 $R$，再代入式（M.11）。由于 $L$ 和 $R$ 恰好二分 $C$，全部单源 EI 项相互抵消，直接得到式（M.16）。$\square$

## 4. Synergy Partition Tree（SPT）

**Synergy Partition Tree（SPT）** 是联合有效信息增量的递归划分树。它从完整源状态 $\mathbf{X}_t$ 开始，根节点取值为 $\Xi_{\ell}(\mathbf{X}_t;\mathbf{X}_{t+\ell})$；每个内部节点对应一个联合源集合，每条分支对应一次不相交二分，节点值记录该次划分后仍必须联合读取两个子块才能保留的局部协同。标准 SPT 在每个节点使用下述贪婪准则选择二分，直至到达单源叶节点。

SPT 不声称恢复唯一的高阶 PID 原子，而是回答：从全部源变量出发，每一步尽量拆出可由两个联合源变量解释的部分后，还有哪些源变量必须联合读取？若领域知识预先固定某些上层分组，或只允许一组受约束的候选二分，则得到**约束 SPT**。先验可以只规定变量块在若干层内必须保持完整，而不指定最终二分；此时仍应在全部合法块二分中按式（M.18）优化。例如三个完整块 $A,B,C$ 的根节点可比较 $A\mid(B,C)$、$B\mid(A,C)$ 和 $C\mid(A,B)$，由目标函数决定哪个块先分离。只有被先验唯一指定的切分才称为固定切分。报告结果时必须明确哪些结构来自候选约束，哪些来自数据驱动选择。若上层采用二分以外的固定多块划分，则 SPT 严格指各块内部的二叉子树；上层多块分解只使用同一层级闭合原理，不称为数据驱动的 SPT 切分。

### 4.1 SPT 的构造

SPT 把当前联合源变量 $C$ 拆成两个非空联合源变量：

$$
C=(L,R),\qquad L\neq\varnothing,\qquad R\neq\varnothing.
\tag{M.17}
$$

根据定理 2，父源变量的 $\Xi$ 等于两个子源变量的 $\Xi$ 与当前二分协同之和，并由定理 1 保证三者均非负。在所有允许的二分中，标准 SPT 选择两个子源变量 $\Xi_{\ell}$ 之和最大的一个：

$$
(L^\star,R^\star)
=\underset{C=(L,R)}{\arg\max}
\,[\Xi_{\ell}(L;\mathbf{X}_{t+\ell})+\Xi_{\ell}(R;\mathbf{X}_{t+\ell})].
\tag{M.18}
$$

理论优化不需要额外的非负约束。数值实现可以设置分裂容差 $\tau\ge0$，用于吸收有限样本估计和浮点计算造成的微小偏差；若候选值在容差内相同，则选择估计协同更小者。选定二分后，当前源集合留下的协同直接写为

$$
\begin{aligned}
Syn^{\mathrm{EID}}\!\left(L^\star,R^\star;\mathbf{X}_{t+\ell}\right)
&=\Xi_{\ell}(C;\mathbf{X}_{t+\ell})
-\Xi_{\ell}(L^\star;\mathbf{X}_{t+\ell})
-\Xi_{\ell}(R^\star;\mathbf{X}_{t+\ell})\\
&=EI(C\to\mathbf{X}_{t+\ell})
-EI(L^\star\to\mathbf{X}_{t+\ell})
-EI(R^\star\to\mathbf{X}_{t+\ell}).
\end{aligned}
\tag{M.19}
$$

第二个等号来自 $C=(L^\star,R^\star)$：将式（M.9）代入第一行后，$C$、$L^\star$ 和 $R^\star$ 中的全部单源 EI 两两抵消，只剩整体 EI 减去两个联合源变量各自对目标的 EI。这也说明该项为何记为 $Syn^{\mathrm{EID}}$。

随后对 $L^\star$ 和 $R^\star$ 继续二分，直到每个分支只剩一个源变量。当 $C=(X_t^{(i)},X_t^{(j)})$ 时，唯一的二分是 $L=X_t^{(i)}$、$R=X_t^{(j)}$；由于单源变量的 $\Xi$ 为 0，式（M.10）和式（M.19）给出 $\Xi_{\ell}((X_t^{(i)},X_t^{(j)});\mathbf{X}_{t+\ell})=Syn^{\mathrm{EID}}(X_t^{(i)},X_t^{(j)};\mathbf{X}_{t+\ell})$。协同项只出现在至少包含两个单源变量的内部节点上，单源叶节点不产生协同项。

### 4.2 闭合关系与实现细节

记 $\mathcal{A}_{\ell}$ 为当前 SPT 的内部联合源变量族，并将节点 $C$ 选中的二分记为 $(L_C^\star,R_C^\star)$。把全部内部节点的式（M.19）相加时，每个非根内部节点的 $\Xi_{\ell}$ 都出现一次正号和一次负号，因而两两抵消。叶节点的 $\Xi_{\ell}$ 为 0，所以求和后只留下完整源状态 $\mathbf{X}_t$ 的 $\Xi_{\ell}$。因此闭合关系为严格恒等式：

$$
\Xi_{\ell}(\mathbf{X}_t;\mathbf{X}_{t+\ell})
=\sum_{C\in\mathcal{A}_{\ell}}
Syn^{\mathrm{EID}}\!\left(
L_C^\star,R_C^\star;\mathbf{X}_{t+\ell}
\right).
\tag{M.20}
$$

实现直接按式（M.9）计算最细粒度划分下的 $\Xi$。由于 $L$ 和 $R$ 正好二分 $C$，其中的单源 EI 在式（M.19）中严格抵消；因此每个内部节点的协同也可按式（M.11）直接计算。核心树始终递归到单源叶节点，并保留每个内部节点的原始协同值；面向表格或绘图的扁平视图可以不显示严格为零的项，但不得借此改变树结构或闭合计算。

所有实验共用 `scripts/spt.py` 中的唯一构造入口。固定输入由有序源集合、返回任意联合源 $\Xi$ 的查询接口、候选二分策略和显式数值策略组成；固定输出为包含源集合、$\Xi$、原始节点协同、深度、切分类型与子节点的同一节点结构，同时返回非负性审计和闭合误差。完整 EI 表只通过适配器转换成同一 $\Xi$ 查询接口；大规模谱候选、领域先验约束和固定切分只替换候选二分策略，不另写递归或节点结构。树上的显著性、解剖分区等标记属于输出后的实验注释，不参与核心切分。

非负 SPT 必须在 Syn 的原生单位中声明容差。落在负容差以内的小负值保留原始数值并计入审计，低于负容差的候选必须显式失败并报告最小值、阈值和受影响数量；实现不得使用 `max(0, Syn)`、截断或静默投影。确需允许有符号残差的实验必须显式选择 signed 策略，其结果不得与非负 SPT 混称。

节点协同的阶数定义为 $|C|$。二阶表示两个源变量之间的协同，更高阶表示相应数量的源变量必须一起读取才能保留的协同，根节点表示完整源状态未被两个子源变量解释的全局协同。全部输出均对应 $|C|\ge2$，其中不包含任何单源 EI。SPT 输出不是严格的 Möbius 纯阶分解；结果依赖允许的候选划分、二分路径和数值阈值，并会受到估计误差与容差设置的影响。因此，它应解释为“沿当前 SPT 仍需联合读取的源变量”，而不是唯一的高阶信息分解。

## 附录 A：命题 1 的证明

本附录依据原论文 [P1] 的 Appendix B，说明源侧独立性为何使 PID 冗余项为零，并由此推出式（M.8）。证明使用 PID 的对称性、自冗余性、单调性、目标链式法则和恒等性公理。

**引理 A.1.** 对任意随机变量 $U$、$V$ 和 $T$，若 $U\perp V$，则

$$
Red(U,V;T)=0.
\tag{A.1}
$$

**证明.** 对增广目标 $(U,V,T)$ 两次使用目标链式法则，分别先分解 $(U,V)$ 和 $T$，以及先分解 $T$ 和 $(U,V)$，得到

$$
\begin{aligned}
Red(U,V;U,V,T)
&=Red(U,V;U,V)+Red(U,V;T\mid U,V),\\
Red(U,V;U,V,T)
&=Red(U,V;T)+Red(U,V;U,V\mid T).
\end{aligned}
\tag{A.2}
$$

比较式（A.2）的两个右端，可得

$$
Red(U,V;U,V)+Red(U,V;T\mid U,V)
=Red(U,V;T)+Red(U,V;U,V\mid T).
\tag{A.3}
$$

根据恒等性公理，

$$
Red(U,V;U,V)=I(U;V)=0,
\tag{A.4}
$$

其中最后一个等号来自 $U\perp V$。另一方面，由单调性和非负性，

$$
0\le Red(U,V;T\mid U,V)
\le I(U,V;T\mid U,V)=0,
\tag{A.5}
$$

所以 $Red(U,V;T\mid U,V)=0$。代入式（A.3）后得到

$$
0=Red(U,V;T)+Red(U,V;U,V\mid T).
\tag{A.6}
$$

式（A.6）右端两项均非负，因此 $Red(U,V;T)=0$，引理得证。$\square$

在命题 1 中，取

$$
U=X_t^{(1)},\qquad
V=X_t^{(2)},\qquad
T=\mathbf{X}_{t+1}.
\tag{A.7}
$$

源侧最大熵独立干预给出 $X_t^{(1)}\perp X_t^{(2)}$，由引理 A.1 立即得到

$$
Red\!\left(X_t^{(1)},X_t^{(2)};\mathbf{X}_{t+1}\right)=0.
\tag{A.8}
$$

PID 中的唯一信息满足

$$
Un(U;T\mid V)=I(U;T)-Red(U,V;T).
\tag{A.9}
$$

将式（A.8）代入式（A.9），并注意最大熵干预分布下的互信息就是 EI，可得

$$
\begin{aligned}
Un\!\left(X_t^{(1)};\mathbf{X}_{t+1}\mid X_t^{(2)}\right)
&=I_{q^{\max}}\!\left(X_t^{(1)};\mathbf{X}_{t+1}\right)
=EI\!\left(X_t^{(1)}\to\mathbf{X}_{t+1}\right),\\
Un\!\left(X_t^{(2)};\mathbf{X}_{t+1}\mid X_t^{(1)}\right)
&=I_{q^{\max}}\!\left(X_t^{(2)};\mathbf{X}_{t+1}\right)
=EI\!\left(X_t^{(2)}\to\mathbf{X}_{t+1}\right).
\end{aligned}
\tag{A.10}
$$

三变量 PID 的分解恒等式为

$$
\begin{aligned}
I_{q^{\max}}\!\left((X_t^{(1)},X_t^{(2)});\mathbf{X}_{t+1}\right)
={}&Red\!\left(X_t^{(1)},X_t^{(2)};\mathbf{X}_{t+1}\right)\\
&+Un\!\left(X_t^{(1)};\mathbf{X}_{t+1}\mid X_t^{(2)}\right)\\
&+Un\!\left(X_t^{(2)};\mathbf{X}_{t+1}\mid X_t^{(1)}\right)\\
&+Syn\!\left(X_t^{(1)},X_t^{(2)};\mathbf{X}_{t+1}\right).
\end{aligned}
\tag{A.11}
$$

把式（A.8）和式（A.10）代入式（A.11），得到

$$
\begin{aligned}
Syn\!\left(X_t^{(1)},X_t^{(2)};\mathbf{X}_{t+1}\right)
={}&EI\!\left((X_t^{(1)},X_t^{(2)})\to\mathbf{X}_{t+1}\right)\\
&-EI\!\left(X_t^{(1)}\to\mathbf{X}_{t+1}\right)
-EI\!\left(X_t^{(2)}\to\mathbf{X}_{t+1}\right).
\end{aligned}
\tag{A.12}
$$

由于引理 A.1 对任意目标随机变量 $T$ 均成立，同一结论也适用于任意预测尺度下的完整目标状态 $\mathbf{X}_{t+\ell}$。$\square$

## 参考文献

[P1] Yang, M., Wang, S., & Zhang, J. (2026). *Partial Effective Information Decomposition for Synergistic Causality*. arXiv:2605.03267.
