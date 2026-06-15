# WMS 相近的 \(2\to1\) 协同与整合信息指标

本文根据 Mediano, Seth, and Barrett (2018), *Measuring Integrated Information:
Comparison of Candidate Measures in Theory and Simulation*，整理能够用于二源一目标
关系

$$
(X_1,X_2)\to Y
$$

的协同或整合信息指标。这里的目标不是把论文中的整系统 \(\Phi\) 指标机械地改名为
二源协同，而是区分：

1. 可以直接用于共同目标 \(Y\) 的指标；
2. 需要额外定义受限模型后才能用于 \(2\to1\) 的指标；
3. 原本不回答二源协同问题的指标。

所有量都依赖联合分布 \(p(x_1,x_2,y)\)。若该分布来自自然轨迹，它们描述观测分布上的
信息结构；若来自独立最大熵干预，它们描述指定干预分布下的机制信息。两种口径不能混为
同一个量。

## 1. 指标总览

| 指标 | \(2\to1\) 形式 | 非负 | 是否直接衡量协同 | 建议用途 |
|---|---|---:|---:|---|
| Whole-minus-sum，WMS | \(I(X_1,X_2;Y)-I(X_1;Y)-I(X_2;Y)\) | 否 | 净协同减冗余 | 最简单的有符号筛选量 |
| PID integrated synergy，\(\psi\) | \(I(X_1,X_2;Y)-I_\cup(X_1,X_2;Y)\) | 是 | 是 | 需要非负协同原子时优先 |
| 条件随机交互，\(\widetilde{\phi}_{2\to1}\) | \(I(X_1;X_2\mid Y)\) | 是 | 仅在独立源下等于 WMS | 检查目标诱导的条件依赖 |
| Decoder-based loss，\(\phi^*_{2\to1}\) | \(I_p(X_1,X_2;Y)-I^*_{p,q}(X_1,X_2;Y)\) | 是 | 取决于受限解码器 \(q\) | 衡量忽略联合结构造成的解码损失 |
| Geometric integration，\(\phi^G_{2\to1}\) | \(\min_{q\in\mathcal Q}D_{\mathrm{KL}}(p\Vert q)\) | 是 | 取决于受限模型族 \(\mathcal Q\) | 衡量到无交互模型族的距离 |
| Causal density / 条件 TE | 无自然的共同目标协同形式 | 是 | 否 | 衡量有向单源传递，不作为协同量 |

## 2. WMS：有符号净协同

对共同目标 \(Y\)，WMS 定义为

$$
\operatorname{WMS}(X_1,X_2;Y)
=I(X_1,X_2;Y)-I(X_1;Y)-I(X_2;Y).
$$

它也等于三元交互信息的相反数：

$$
\operatorname{WMS}(X_1,X_2;Y)
=I(X_1;X_2\mid Y)-I(X_1;X_2).
$$

因此：

- \(\operatorname{WMS}>0\)：联合信息相对于单源信息之和占优；
- \(\operatorname{WMS}<0\)：两个单源对目标的冗余信息占优；
- \(\operatorname{WMS}=0\)：不能据此断言没有协同，因为正协同与冗余可能相互抵消。

在二源 PID 中，若

$$
I(X_1,X_2;Y)=R+U_1+U_2+S,
$$

则

$$
\operatorname{WMS}=S-R.
$$

所以 WMS 是“协同减冗余”的净量，不是非负协同原子。它的优点是只需估计三个互信息，
实现简单；缺点是负值和零值均存在解释歧义。

## 3. Integrated synergy \(\psi\)：PID 协同原子

Mediano et al. 将 integrated synergy 写成联合预测信息减去各部分信息的 union：

$$
\psi(X_1,X_2;Y)
=I(X_1,X_2;Y)-I_\cup(X_1,X_2;Y).
$$

对标准二源 PID，union information 为

$$
I_\cup(X_1,X_2;Y)
=I(X_1;Y)+I(X_2;Y)-R(X_1,X_2;Y),
$$

从而

$$
\psi(X_1,X_2;Y)=S(X_1,X_2;Y)\ge 0.
$$

它与 WMS 的关系为

$$
\psi=\operatorname{WMS}+R.
$$

因此，\(\psi\) 修正了 WMS 把冗余作为负项扣除的问题，但代价是必须选择一个冗余定义。
PID 本身不能仅由三个普通互信息唯一确定。

### 3.1 MMI-PID 的可计算形式

对 Mediano et al. 使用的线性高斯 MMI-PID，

$$
R_{\mathrm{MMI}}(X_1,X_2;Y)
=\min\{I(X_1;Y),I(X_2;Y)\}.
$$

于是

$$
\psi_{\mathrm{MMI}}
=I(X_1,X_2;Y)-\max\{I(X_1;Y),I(X_2;Y)\}.
$$

等价地，

$$
\psi_{\mathrm{MMI}}
=\min\{I(X_1;Y\mid X_2),I(X_2;Y\mid X_1)\}.
$$

该形式只需要普通互信息或条件互信息，适合线性高斯 benchmark。对一般离散或非高斯
系统，应明确使用的 PID 冗余函数，不能把 MMI-PID 默认当作唯一正确分解。

## 4. 条件随机交互：\(I(X_1;X_2\mid Y)\)

论文中的 integrated stochastic interaction 比较“整系统逆向不确定性”与“分块逆向
不确定性”。原定义要求每个过去分块都有对应的未来分块，因此不能原样用于两个源共享
一个目标的 \(2\to1\) 图。

若将同一个目标 \(Y\) 作为两个源的共同条件，可以定义共享目标版本

$$
\widetilde{\phi}_{2\to1}
=H(X_1\mid Y)+H(X_2\mid Y)-H(X_1,X_2\mid Y)
=I(X_1;X_2\mid Y).
$$

该量恒非负，表示观察 \(Y\) 后两个源之间仍有或新出现多少条件依赖。它与 WMS 满足

$$
\widetilde{\phi}_{2\to1}
=\operatorname{WMS}+I(X_1;X_2).
$$

特别地，当源分布独立时，

$$
I(X_1;X_2)=0
\quad\Longrightarrow\quad
\widetilde{\phi}_{2\to1}=\operatorname{WMS}\ge 0.
$$

这解释了为什么在独立最大熵源干预下，WMS 可以退化成非负的源侧联合约束；而在自然
轨迹上，源相关性会使 WMS 与条件随机交互明显不同。

需要注意，\(I(X_1;X_2\mid Y)\) 不是一般 PID 协同原子。若源本来相关，它会同时包含
源依赖、目标条件化和协同结构的混合影响。

## 5. Decoder-based \(\phi^*_{2\to1}\)：忽略联合结构的解码损失

论文中的 decoder-based integrated information 定义为真实解码信息与使用错误受限模型
时可提取信息之差：

$$
\phi^*_{2\to1}
=I_p(X_1,X_2;Y)-I^*_{p,q}(X_1,X_2;Y).
$$

其中 \(p(y\mid x_1,x_2)\) 是真实通道，\(q(y\mid x_1,x_2)\) 是刻意忽略某类联合结构的
受限解码器，\(I^*_{p,q}\) 是 mismatched decoding information。

整系统 \(\Phi^*\) 的标准受限模型通过切断分区之间的连接构造。对共同目标 \(Y\)，没有
唯一的标准切法；必须先指定 \(q\)，例如：

- 只允许最佳单源解码器；
- 允许两个单源证据的预注册组合，但禁止联合交互特征；
- 使用可加广义线性模型作为受限解码器。

因此，\(\phi^*_{2\to1}\) 可以回答“忽略联合结构会损失多少可解码信息”，但数值依赖
受限解码器的定义。它适合作为任务相关的预测损失指标，不应在未说明 \(q\) 时直接称为
通用协同信息。

## 6. Geometric \(\phi^G_{2\to1}\)：到无交互模型族的 KL 距离

geometric integrated information 的核心是：寻找与真实分布最近、但满足“切断连接”
约束的分布。对 \(2\to1\) 可写成

$$
\phi^G_{2\to1}
=\min_{q\in\mathcal Q}
D_{\mathrm{KL}}\!\left(
p(x_1,x_2,y)\,\Vert\,q(x_1,x_2,y)
\right),
$$

其中 \(\mathcal Q\) 是预先定义的无联合交互模型族。一个实用选择是保留源分布
\(p(x_1,x_2)\)，只限制目标条件模型：

$$
q(x_1,x_2,y)=p(x_1,x_2)q(y\mid x_1,x_2),
$$

并令 \(q(y\mid x_1,x_2)\) 属于可加条件模型族。此时

$$
\phi^G_{2\to1}
=\min_{q\in\mathcal Q}
\mathbb E_{p(x_1,x_2)}
\left[
D_{\mathrm{KL}}\!\left(
p(y\mid x_1,x_2)\Vert q(y\mid x_1,x_2)
\right)
\right].
$$

该量非负，并且直接衡量真实条件通道偏离预注册“无交互”模型族的程度。但它测量的是
相对于模型族的不可约性，不自动等于 PID 协同。若把 \(\mathcal Q\) 选成可加均值模型，
它主要识别函数非加性；若允许异方差或更复杂组合，结果会随模型族改变。

## 7. Causal density 为什么不应作为 \(2\to1\) 协同量

论文中的 causal density 是系统各分量之间条件 transfer entropy 的平均：

$$
\operatorname{TE}(X_i\to X_j\mid \mathbf X_{\setminus\{i,j\}}).
$$

对二源一目标，可以分别计算

$$
\operatorname{TE}(X_1\to Y\mid X_2),
\qquad
\operatorname{TE}(X_2\to Y\mid X_1).
$$

这些量回答“控制另一个源后，单源是否仍提供有向预测信息”，而不是“只有联合观察两个
源才能获得多少信息”。将两项相加仍然是条件单源传递总量，不是协同原子。因此 causal
density 可作为 pairwise/conditional-flow 对照，但不应替代 WMS、\(\psi\) 或 PEID
协同。

## 8. 估计与选择建议

### 8.1 离散变量

对离散联合概率表，可直接用熵和互信息计算 WMS、条件随机交互与选定 PID 下的
\(\psi\)。有限样本下应报告偏差修正、bootstrap 区间和分箱敏感性。

### 8.2 线性高斯变量

互信息可由协方差闭式计算：

$$
I(\mathbf X;Y)
=\frac12\log
\frac{\det\boldsymbol\Sigma_Y}
{\det\boldsymbol\Sigma_{Y\mid\mathbf X}}.
$$

此时 WMS、\(I(X_1;X_2\mid Y)\) 和 MMI-PID \(\psi\) 都容易计算。decoder-based 与
geometric 指标仍需要额外优化。

### 8.3 一般连续变量

可使用 kNN、transport map 或其他密度/互信息估计器，但同一比较中的联合互信息、单源
互信息和条件互信息应使用一致的估计协议。不要把小幅负估计残差自动解释为真实负协同。
关于 Continuous SxPID、Continuous BROJA、Flow-PID 与 PIRD 的方法边界和选择协议，
见 [一般连续动力学中的 PID 协同计算前沿](./一般连续动力学中的PID协同计算前沿.md)。

### 8.4 推荐顺序

1. 用 WMS 做最低成本的有符号筛选，并同时报告三个组成互信息。
2. 需要非负二源协同原子时，使用明确冗余定义的 PID \(\psi\)；线性高斯基准可使用
   MMI-PID。
3. 使用独立最大熵源干预时，可同时报告
   \(I(X_1;X_2\mid Y)\)，并验证它与 WMS 的理论等价关系。
4. 只有当研究问题明确涉及“受限解码器损失”或“到无交互模型族的距离”时，才使用
   \(\phi^*_{2\to1}\) 或 \(\phi^G_{2\to1}\)，并完整报告 \(q\) 或 \(\mathcal Q\)。
5. causal density 或条件 TE 只作为有向单源信息流对照，不作为二源协同指标。

## 9. 与当前 PEID 口径的关系

当前项目中的二源 PEID 协同写为

$$
\operatorname{Syn}^{\mathrm{EID}}(X_1,X_2\to Y)
=EI(X_1,X_2\to Y)-EI(X_1\to Y)-EI(X_2\to Y).
$$

它在代数形式上与 WMS 相同，但 EI 使用预注册的源侧独立最大熵干预分布，而 observational
WMS 使用自然轨迹经验分布。独立源使

$$
I(X_1;X_2)=0,
$$

因此

$$
\operatorname{Syn}^{\mathrm{EID}}
=I(X_1;X_2\mid Y)\ge 0.
$$

所以项目中的 PEID 协同不是新增一种任意的 WMS 变体，而是通过改变源分布，消除源侧
冗余项后得到的非负联合约束。比较 observational WMS 与 PEID 时，必须同时说明二者的
采样分布和因果语义不同。

## 参考文献

- Mediano, P. A. M., Seth, A. K., & Barrett, A. B. (2018). Measuring Integrated
  Information: Comparison of Candidate Measures in Theory and Simulation.
  *Entropy*, 21(1), 17. Zotero item: `X34436WI`.
- Barrett, A. B., & Seth, A. K. (2011). Practical measures of integrated
  information for time-series data. *PLoS Computational Biology*, 7,
  e1001052.
- Oizumi, M., Amari, S., Yanagawa, T., Fujii, N., & Tsuchiya, N. (2016).
  Measuring integrated information from the decoding perspective.
  *PLoS Computational Biology*, 12, e1004654.
