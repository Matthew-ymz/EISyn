# 高阶 PEID 协同的 Möbius 定义与非负性问题

本文记录一种用于分离精确高阶协同的候选定义，并检查它在源变量相互独立时是否仍然非负。结论是：该定义给出了唯一的加法分解和严格的阶数分离，但从三阶开始不保证非负。因此它更适合称为精确高阶 EI interaction，只有当数值为正时才可解释为该阶协同。

## 1. 设定

设源变量集合为

```math
S=\{1,\dots,n\},
```

目标变量为 \(T\)。在 PEID 语境中，所有有效信息均在源侧最大熵干预分布下计算；离散情形中这通常意味着源变量在干预分布下相互独立且均匀。对任意非空源子集 \(A\subseteq S\)，记

```math
F(A)=EI(X_A\to T),
\tag{1}
```

并约定

```math
F(\varnothing)=0.
\tag{2}
```

这里 \(X_A=(X_i)_{i\in A}\)。如果目标是下一期目标子系统，也可把 \(T\) 替换为 \(X^B_{t+1}\)，定义形式不变。

## 2. 精确 k 阶 EI interaction

对任意非空源集合 \(K\subseteq S\)，令 \(k=|K|\)。定义集合 \(K\) 关于目标 \(T\) 的精确 \(k\) 阶 EI interaction 为

```math
\Delta_K(T)
=\sum_{A\subseteq K}(-1)^{k-|A|}F(A).
\tag{3}
```

由于 \(F(\varnothing)=0\)，也可写成

```math
\Delta_K(T)
=\sum_{r=1}^{k}(-1)^{k-r}
  \sum_{\substack{A\subseteq K\\ |A|=r}} EI(X_A\to T).
\tag{4}
```

展开为

```math
\Delta_K(T)
=EI(X_K\to T)
-\sum_{\substack{A\subseteq K\\ |A|=k-1}}EI(X_A\to T)
+\sum_{\substack{A\subseteq K\\ |A|=k-2}}EI(X_A\to T)
-\cdots
+(-1)^{k-1}\sum_{i\in K}EI(X_i\to T).
\tag{5}
```

当 \(k=2\) 时，

```math
\Delta_{\{i,j\}}(T)
=EI(X_i,X_j\to T)-EI(X_i\to T)-EI(X_j\to T).
\tag{6}
```

当 \(k=3\) 时，

```math
\Delta_{\{i,j,\ell\}}(T)
=EI(X_i,X_j,X_\ell\to T)
-EI(X_i,X_j\to T)-EI(X_i,X_\ell\to T)-EI(X_j,X_\ell\to T)
+EI(X_i\to T)+EI(X_j\to T)+EI(X_\ell\to T).
\tag{7}
```

式 (7) 正是“三源整体 EI 减去所有二源 EI，再加回所有单源 EI”的定义。

## 3. Möbius 反演与唯一加法分解

命题 1。对任意非空 \(K\subseteq S\)，有

```math
EI(X_K\to T)=\sum_{\varnothing\ne A\subseteq K}\Delta_A(T).
\tag{8}
```

证明。式 (3) 是布尔子集格上的 Möbius 变换。对右侧展开：

```math
\sum_{\varnothing\ne A\subseteq K}\Delta_A(T)
=\sum_{\varnothing\ne A\subseteq K}
  \sum_{B\subseteq A}(-1)^{|A|-|B|}F(B).
\tag{9}
```

交换求和次序，某个固定 \(B\subseteq K\) 的系数为

```math
\sum_{A:\,B\subseteq A\subseteq K}(-1)^{|A|-|B|}.
\tag{10}
```

若 \(B=K\)，该系数为 \(1\)。若 \(B\subset K\)，令 \(m=|K|-|B|>0\)，则式 (10) 等于

```math
\sum_{q=0}^{m}\binom{m}{q}(-1)^q=(1-1)^m=0.
\tag{11}
```

因此右侧只剩 \(F(K)=EI(X_K\to T)\)，命题成立。

命题 1 说明，\(\Delta_K(T)\) 是由所有子集 EI 唯一诱导的精确阶数原子。相应地，当前 PEID 中基于 singleton finest partition 的 residual

```math
R_K(T)=EI(X_K\to T)-\sum_{i\in K}EI(X_i\to T)
\tag{12}
```

并不是纯 \(k\) 阶项，而是

```math
R_K(T)=\sum_{\substack{A\subseteq K\\ |A|\ge 2}}\Delta_A(T).
\tag{13}
```

因此 \(R_K(T)\) 会把所有二阶、三阶、一直到 \(k\) 阶的 interaction 混合在一起。

## 4. 非负性尝试

下面检查命题：“当源变量彼此独立时，\(\Delta_K(T)\ge 0\)”。这个命题对二阶成立，但对三阶及以上不成立。

### 4.1 二阶非负性成立

设 \(K=\{i,j\}\)。在源侧干预分布下若 \(X_i\perp X_j\)，则

```math
\Delta_{\{i,j\}}(T)
=I(X_i,X_j;T)-I(X_i;T)-I(X_j;T).
\tag{14}
```

利用链式法则，

```math
I(X_i,X_j;T)-I(X_i;T)-I(X_j;T)
=I(X_i;X_j\mid T)-I(X_i;X_j).
\tag{15}
```

由 \(X_i\perp X_j\) 得 \(I(X_i;X_j)=0\)，所以

```math
\Delta_{\{i,j\}}(T)=I(X_i;X_j\mid T)\ge 0.
\tag{16}
```

这就是 PEID 二源协同非负性的核心原因。

### 4.2 三阶非负性不成立

设 \(X_1,X_2,X_3\) 是相互独立的 Bernoulli\((1/2)\) 变量，并定义目标

```math
T=
\begin{cases}
X_1, & X_3=0,\\
X_2, & X_3=1.
\end{cases}
\tag{17}
```

这等价于一个二选一 multiplexer：第三个源 \(X_3\) 决定目标复制 \(X_1\) 还是复制 \(X_2\)。

对该分布可直接计算：

```math
I(X_1;T)=I(X_2;T)=0.1887218755,\qquad I(X_3;T)=0,
\tag{18}
```

```math
I(X_1,X_2;T)=I(X_1,X_3;T)=I(X_2,X_3;T)=0.5,
\tag{19}
```

```math
I(X_1,X_2,X_3;T)=1.
\tag{20}
```

代入式 (7) 得

```math
\Delta_{\{1,2,3\}}(T)
=1-0.5-0.5-0.5+0.1887218755+0.1887218755+0
=-0.1225562489<0.
\tag{21}
```

所以，即使源变量在干预分布下完全独立，精确三阶 EI interaction 也可能为负。

### 4.3 为什么证明会失败

对任意 \(K\) 且 \(|K|\ge 2\)，由

```math
I(X_A;T)=H(X_A)-H(X_A\mid T)
\tag{22}
```

代入式 (3)。如果源变量在干预分布下相互独立，则无条件熵项的 Möbius 变换为零：

```math
\sum_{A\subseteq K}(-1)^{|K|-|A|}H(X_A)=0.
\tag{23}
```

于是

```math
\Delta_K(T)
=-\sum_{A\subseteq K}(-1)^{|K|-|A|}H(X_A\mid T).
\tag{24}
```

当 \(|K|=2\) 时，式 (24) 退化为条件互信息 \(I(X_i;X_j\mid T)\)，因此非负。可是当 \(|K|=3\) 时，它等于条件三变量 co-information 的相反数：

```math
\Delta_{\{1,2,3\}}(T)
=-I(X_1;X_2;X_3\mid T).
\tag{25}
```

条件 co-information 是有符号量，既可正也可负，因此式 (25) 不可能提供一般非负性证明。更高阶情形同理，对应的是条件多变量 interaction information 的符号变换，也不是 KL 散度或条件总相关，因此没有一般非负性。

### 4.4 加两倍单源 EI 仍不保证非负

一种自然修正是认为三组二源项之间存在 overlap，因此把式 (7) 中的单源项系数从 \(1\) 改为 \(2\)：

```math
\widetilde{\Delta}_{\{1,2,3\}}(T)
=I(X_1,X_2,X_3;T)
-I(X_1,X_2;T)-I(X_1,X_3;T)-I(X_2,X_3;T)
+2I(X_1;T)+2I(X_2;T)+2I(X_3;T).
\tag{26}
```

这个修正确实会把上面的 multiplexer 例子变成正值，因为该例中单源 EI 不为零。但它仍然不能保证一般非负。考虑 \(X_1,X_2,X_3\) 相互独立且均为 Bernoulli\((1/2)\)，目标 \(T\) 由如下真值表确定：

```text
X1 X2 X3 | T
0  0  0  | 0
0  0  1  | 0
0  1  0  | 0
0  1  1  | 1
1  0  0  | 1
1  0  1  | 0
1  1  0  | 0
1  1  1  | 0
```

也就是说，\(T=1\) 当且仅当 \((X_1,X_2,X_3)=(0,1,1)\) 或 \((1,0,0)\)。该机制具有符号对称性，所以所有单源 EI 都为零：

```math
I(X_1;T)=I(X_2;T)=I(X_3;T)=0.
\tag{27}
```

直接计算得到

```math
I(X_1,X_2;T)=I(X_1,X_3;T)=I(X_2,X_3;T)=0.3112781245,
\tag{28}
```

```math
I(X_1,X_2,X_3;T)=0.8112781245.
\tag{29}
```

因此

```math
\widetilde{\Delta}_{\{1,2,3\}}(T)
=0.8112781245-3\times 0.3112781245+0
=-0.1225562489<0.
\tag{30}
```

这个反例说明，二源项之间的 overlap 不一定等于重复变量的单源 EI。即使所有单源 EI 都为零，二源信息之间仍可能存在高阶重叠，使得式 (26) 为负。

## 5. 与 PEID residual 的关系

PEID 文献中对一个源侧划分 \(P=\{M_1,\dots,M_m\}\) 定义的 residual 为

```math
Syn^{EID}_P(X_A\to T)
=EI(X_A\to T)-\sum_{q=1}^{m}EI(X_{M_q}\to T).
\tag{31}
```

在离散且源侧最大熵干预下，式 (31) 可写成条件 total correlation：

```math
Syn^{EID}_P(X_A\to T)
=TC(X_{M_1},\dots,X_{M_m}\mid T)\ge 0.
\tag{32}
```

因此，原 PEID residual 的非负性来自条件 total correlation。它是相对于划分 \(P\) 的非加性总量，而不是精确阶数原子。

特别地，对 singleton finest partition，

```math
Syn^{EID}_{fine}(X_K\to T)
=EI(X_K\to T)-\sum_{i\in K}EI(X_i\to T)
=\sum_{\substack{A\subseteq K\\ |A|\ge 2}}\Delta_A(T).
\tag{33}
```

式 (33) 的左侧非负，但右侧的各个 \(\Delta_A(T)\) 可以有正有负。也就是说，非负的是所有二阶及以上 interaction 的总和，不是每一个精确高阶 interaction。

## 6. 建议命名

为避免概念混淆，建议使用以下命名：

1. \(\Delta_K(T)\)：精确 \(k\) 阶 EI interaction，或 signed \(k\)-order PEID atom。它具有唯一加法分解，但可能为负。
2. \(\max\{0,\Delta_K(T)\}\)：正向精确 \(k\) 阶协同强度。它非负，但不再满足严格加法分解。
3. \(Syn^{EID}_{fine}(X_K\to T)\)：finest-partition source nonadditivity，或 order-mixed source synergy residual。它非负，但混合了所有二阶及以上 interaction。

因此，如果论文目标是“识别真正的 \(k\) 阶机制”，式 (3) 是合适的抽象定义；如果目标是“保证非负的协同强度”，则不能同时保留式 (8) 的精确阶数加法分解。
