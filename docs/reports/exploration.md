# 探索记录汇总

整理日期：2026-10-03。本文件集中保留暂未进入当前论文主线的旧探索及早期检查记录；其结论、文献证据和计算结果均保留原实验日期与适用边界，不自动升级为现行方法或执行指令。Brain 及 UniCM 的实际结果分别见[脑主报告](brain.md)和[Earth 主报告](earth.md)。Method、Part1 和纽约交通主报告继续独立维护。

- [A：RW-EI 原理、数值实验与高维失败边界](#rw-ei)（2026-08-24）
- [B：Sine 振幅—频率校准](#sine-frequency)（2026-08-25）
- [C：EI—Koopman—控制的理论边界与小实验](#ei-koopman-control)（2026-09-14）
- [D：控制方向的文献证据与阅读顺序](#ei-control-literature)（2026-09-14）
- [E：Part1 早期公平性检查约定](#part1-fairness-check)（历史检查记录）

各节内的“本次”“当前”按原报告日期理解。RW-EI 的密度比重加权不替代当前默认 TM 方法；探索中的动作分布、干预支持及控制成本与当前主稿 PEID 对象分别声明。原报告引用的 `control.md` 在本次整理前已被删除，本文只保留当时的结论记录及仍存在的实验入口，不补造已缺失报告。

<a id="rw-ei"></a>

## 附录 A：RW-EI 原理、数值实验与高维失败边界

### A.1 核心结论

Reweight EI（简称 **RW-EI**）提供了一条不拟合动力学模型的 EI 计算路径：它直接调整观测样本的权重，把原本相关的输入分布转换为目标干预分布，再从重加权后的联合分布中计算有效信息。

本实验得到四个直观结论：

1. **小样本时，优先考虑 RW-EI。** RW-EI 不需要先训练动力学模型，因此能够避免小样本条件下明显的模型拟合误差。
2. **样本充足时，两种方法都稳定在真值附近，但 RW-EI 更快。** 在最大样本规模下，RW-EI 的计算时间约为 MLP EI 的 $1/12.2$。
3. **输入相关性过强时，优先考虑 MLP EI。** 强相关会削弱观测分布与独立干预分布之间的重叠，使少量样本获得极大权重，进而造成 RW-EI 的明显偏差。
4. **ESS 是选择 RW-EI 的实用诊断指标。** ESS 越高，权重越均匀，RW-EI 通常越稳定；ESS 过低则意味着结果高度依赖少数样本。

一句话概括：**有重叠、看 ESS、优先 RW-EI；强相关、低 ESS、改用 MLP EI。**

### A.2 实验问题与数据生成过程

实验比较两种 EI 计算路径：

- **MLP EI**：先从观测数据拟合随机动力学，再在目标输入分布下生成干预样本，最后计算互信息；
- **RW-EI**：不拟合动力学，直接对观测样本进行重加权，再计算互信息。

此外，实验还设置了两个使用已知动力学的参照：**Oracle samples + TM** 和 **EI truth**。它们都知道真实的 $f$ 与噪声分布，但计算方式不同：前者先从真实动力学随机采样，再用 TM 估计互信息；后者直接对真实概率分布做确定性数值积分，不经过 TM。

令输入随机向量为 $\mathbf{x}=(X_1,X_2)^\mathsf{T}$，输出为标量 $Y$。观测输入的两个边缘分布均为 $\operatorname{Unif}[-1,1]$，但通过 Gaussian copula 引入相关系数 $\rho$。非线性随机动力学为

$$
Y=f(\mathbf{x})+\varepsilon,
\qquad
\varepsilon\sim\mathcal{N}(0,\sigma^2),
$$

其中

$$
f(\mathbf{x})
=0.8x_1-0.4x_2
+1.1\sin(\pi x_1x_2)
+0.35x_1^2,
\qquad
\sigma=0.3.
$$

目标干预把两个输入变量变为相互独立的均匀变量：

$$
q_{\mathrm{do}}(\mathbf{x})
=q_1(x_1)q_2(x_2),
\qquad
q_j(x_j)=\operatorname{Unif}[-1,1].
$$

干预只改变输入分布，不改变条件机制 $p(y\mid\mathbf{x})$，因此目标联合分布为

$$
q(\mathbf{x},y)
=q_{\mathrm{do}}(\mathbf{x})p(y\mid\mathbf{x}).
$$

实验中的 EI 就是这个干预联合分布下的互信息：

$$
\operatorname{EI}
=I_q(\mathbf{x};Y)
=\iint q(\mathbf{x},y)
\log_2
\frac{q(\mathbf{x},y)}
{q_{\mathrm{do}}(\mathbf{x})q(y)}
\,\mathrm{d}\mathbf{x}\,\mathrm{d}y.
$$

由于 $q(\mathbf{x},y)=q_{\mathrm{do}}(\mathbf{x})p(y\mid\mathbf{x})$，也可以写成

$$
\operatorname{EI}
=\iint q_{\mathrm{do}}(\mathbf{x})p(y\mid\mathbf{x})
\log_2
\frac{p(y\mid\mathbf{x})}{q(y)}
\,\mathrm{d}\mathbf{x}\,\mathrm{d}y.
$$

这个表达式说明：EI 衡量的是在指定输入干预下，知道当前输入 $\mathbf{x}$ 能够减少多少关于下一时刻输出 $Y$ 的不确定性。

#### A.2.1 EI truth：直接积分已知动力学

本实验的干预密度和条件动力学密度分别为

$$
q_{\mathrm{do}}(\mathbf{x})
=\frac{1}{4}\,\mathbb{I}\!\left(\mathbf{x}\in[-1,1]^2\right),
$$

$$
p(y\mid\mathbf{x})
=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left[
-\frac{\left(y-f(\mathbf{x})\right)^2}{2\sigma^2}
\right].
$$

因此，干预后的输出边缘密度是一个连续高斯混合：

$$
\begin{aligned}
q(y)
&=\int q_{\mathrm{do}}(\mathbf{x})p(y\mid\mathbf{x})\,\mathrm{d}\mathbf{x}\\
&=\frac{1}{4}
\int_{-1}^{1}\int_{-1}^{1}
\mathcal{N}\!\left(y;f(x_1,x_2),\sigma^2\right)
\,\mathrm{d}x_1\,\mathrm{d}x_2.
\end{aligned}
$$

因为 $Y=f(\mathbf{x})+\varepsilon$ 且噪声与输入独立，条件熵就是高斯噪声熵：

$$
H_q(Y\mid\mathbf{x})
=H(\varepsilon)
=\frac{1}{2}\log_2\!\left(2\pi e\sigma^2\right).
$$

输出边缘熵为

$$
H_q(Y)
=-\int q(y)\log_2 q(y)\,\mathrm{d}y.
$$

所以图中的 EI truth 定义为

$$
\operatorname{EI}_{\mathrm{truth}}
=H_q(Y)-\frac{1}{2}\log_2\!\left(2\pi e\sigma^2\right).
$$

实现中并不是用 Monte Carlo 样本近似 $q(y)$，而是使用二维 Gauss–Legendre 求积。令 $\{x_a,\omega_a\}_{a=1}^{L}$ 为区间 $[-1,1]$ 上的求积节点与权重，则

$$
\widehat q_L(y)
=\frac{1}{4}
\sum_{a=1}^{L}\sum_{b=1}^{L}
\omega_a\omega_b
\mathcal{N}\!\left(y;f(x_a,x_b),\sigma^2\right).
$$

由于实际输出积分只覆盖有限区间，代码先用梯形积分计算该区间内的质量并重新归一化：

$$
Z_L
=\int_{y_1}^{y_G}\widehat q_L(y)\,\mathrm{d}y,
\qquad
\widetilde q_L(y)
=\frac{\widehat q_L(y)}{Z_L}.
$$

随后用同一输出网格 $y_1,\ldots,y_G$ 上的梯形积分近似输出熵：

$$
\widehat H_L(Y)
\approx
-\sum_{g=1}^{G-1}\frac{y_{g+1}-y_g}{2}
\left[
\widetilde q_L(y_g)\log_2\widetilde q_L(y_g)
+\widetilde q_L(y_{g+1})\log_2\widetilde q_L(y_{g+1})
\right].
$$

最终使用

$$
\widehat{\operatorname{EI}}_{\mathrm{truth}}
=\widehat H_L(Y)
-\frac{1}{2}\log_2\!\left(2\pi e\sigma^2\right).
$$

全量实验取每个输入轴 $L=96$ 个求积节点和 $G=3000$ 个输出网格点；输出积分范围覆盖 $f(\mathbf{x})$ 的数值范围并向两端各扩展 $8\sigma$。因此，这里的“truth”是高精度数值真值，其剩余误差主要来自求积分辨率和有限输出积分区间，而不是样本波动或 TM 拟合。

#### A.2.2 Oracle samples + TM：真实动力学生成样本，TM 读取互信息

Oracle TM 同样知道真实动力学，但它不直接计算上面的积分。对每个随机种子，先生成 $M$ 个独立干预样本：

$$
\mathbf{x}^{(m)}\sim q_{\mathrm{do}}(\mathbf{x}),
\qquad
\varepsilon^{(m)}\sim\mathcal{N}(0,\sigma^2),
$$

$$
y^{(m)}
=f\!\left(\mathbf{x}^{(m)}\right)+\varepsilon^{(m)},
\qquad m=1,\ldots,M.
$$

于是

$$
\mathcal{D}_{\mathrm{oracle}}
=\left\{
\left(\mathbf{x}^{(m)},y^{(m)}\right)
\right\}_{m=1}^{M}
$$

是直接来自目标干预联合分布

$$
q(\mathbf{x},y)
=q_{\mathrm{do}}(\mathbf{x})p(y\mid\mathbf{x})
$$

的有限样本。本实验令 $M=N$，使 Oracle、MLP 和 RW-EI 路径使用相同数量级的互信息读取样本。

接着分别用五阶多项式三角传输映射拟合联合密度和两个边缘密度。以联合变量 $\mathbf{z}=(\mathbf{x}^\mathsf{T},y)^\mathsf{T}$ 为例，TM 密度写为

$$
\widehat q^{\mathrm{TM}}_{\mathbf{x}Y}(\mathbf{z})
=\phi\!\left(\mathbf{t}_{\mathbf{x}Y}(\mathbf{z})\right)
\left|
\det\nabla\mathbf{t}_{\mathbf{x}Y}(\mathbf{z})
\right|,
$$

其中 $\phi$ 是标准多元高斯密度。类似地得到 $\widehat q^{\mathrm{TM}}_{\mathbf{x}}(\mathbf{x})$ 和 $\widehat q^{\mathrm{TM}}_Y(y)$。Oracle TM 的 EI 估计为

$$
\widehat{\operatorname{EI}}_{\mathrm{Oracle\text{-}TM}}
=\frac{1}{M}\sum_{m=1}^{M}
\log_2
\frac{
\widehat q^{\mathrm{TM}}_{\mathbf{x}Y}
\!\left(\mathbf{x}^{(m)},y^{(m)}\right)
}{
\widehat q^{\mathrm{TM}}_{\mathbf{x}}
\!\left(\mathbf{x}^{(m)}\right)
\widehat q^{\mathrm{TM}}_Y\!\left(y^{(m)}\right)
}.
$$

Oracle TM 不需要观测数据、不需要 MLP，也不需要密度比重加权。它的作用是单独测量“有限干预样本 + TM 读取器”本身会带来多大误差。其误差包含 Monte Carlo 采样误差、有限样本密度估计误差和五阶 TM 的近似误差。

#### A.2.3 二者的核心区别

| 比较项 | Oracle samples + TM | EI truth |
|---|---|---|
| 是否使用已知 $f$ 和 $\sigma$ | 是 | 是 |
| 如何使用已知动力学 | 随机生成有限干预样本 | 直接构造 $p(y\mid\mathbf{x})$ 并积分 |
| 是否使用 TM | 是，五阶 TM | 否 |
| 是否随随机种子变化 | 是 | 否 |
| 主要误差来源 | Monte Carlo、有限样本、TM 近似 | 求积节点、输出网格和积分截断 |
| 在实验中的作用 | 测量共同 TM 读取器的误差基准 | 提供所有 EI 方法的数值参照 |

因此，二者虽然都使用已知动力学，但不能视为同一个量的重复画法。EI truth 回答“真实干预互信息是多少”；Oracle samples + TM 回答“即使动力学完全已知，只给 TM 有限干预样本时，最终能估计得多准”。二者之间的差距主要反映 TM 读取器与有限样本造成的误差，而不是动力学学习误差或重加权误差。

### A.3 RW-EI 为什么可以绕过动力学拟合

#### A.3.1 从分布替换到重要性权重

观测数据来自

$$
p_{\mathrm{obs}}(\mathbf{x},y)
=p_{\mathrm{obs}}(\mathbf{x})p(y\mid\mathbf{x}),
$$

而 EI 要求的目标数据分布是

$$
q(\mathbf{x},y)
=q_{\mathrm{do}}(\mathbf{x})p(y\mid\mathbf{x}).
$$

两者共享相同的条件机制 $p(y\mid\mathbf{x})$。因此，从观测联合分布转换到干预联合分布所需的密度比为

$$
w(\mathbf{x},y)
=\frac{q(\mathbf{x},y)}
{p_{\mathrm{obs}}(\mathbf{x},y)}
=\frac{
q_{\mathrm{do}}(\mathbf{x})p(y\mid\mathbf{x})
}{
p_{\mathrm{obs}}(\mathbf{x})p(y\mid\mathbf{x})
}
=\frac{q_{\mathrm{do}}(\mathbf{x})}
{p_{\mathrm{obs}}(\mathbf{x})}.
$$

关键点在于，未知的动力学项 $p(y\mid\mathbf{x})$ 在分子和分母中完全抵消。RW-EI 因而只需要估计输入分布比

$$
w(\mathbf{x})
=\frac{q_{\mathrm{do}}(\mathbf{x})}
{p_{\mathrm{obs}}(\mathbf{x})},
$$

而不需要先拟合 $f(\mathbf{x})$ 或完整的条件分布 $p(y\mid\mathbf{x})$。这正是 RW-EI 被称为“transition-model-free”估计方法的原因。

#### A.3.2 如何使用权重

对任意可积函数 $g(\mathbf{x},Y)$，目标干预分布下的期望可以改写为观测分布下的加权期望：

$$
\begin{aligned}
\mathbb{E}_{q}[g(\mathbf{x},Y)]
&=
\iint g(\mathbf{x},y)q(\mathbf{x},y)
\,\mathrm{d}\mathbf{x}\,\mathrm{d}y\\
&=
\iint
g(\mathbf{x},y)
\frac{q(\mathbf{x},y)}
{p_{\mathrm{obs}}(\mathbf{x},y)}
p_{\mathrm{obs}}(\mathbf{x},y)
\,\mathrm{d}\mathbf{x}\,\mathrm{d}y\\
&=
\mathbb{E}_{p_{\mathrm{obs}}}
\left[
w(\mathbf{x})g(\mathbf{x},Y)
\right].
\end{aligned}
$$

给定观测样本

$$
\mathcal{D}
=\{(\mathbf{x}_i,y_i)\}_{i=1}^{N},
$$

先计算原始权重 $w_i=w(\mathbf{x}_i)$，再将其归一化：

$$
\bar{w}_i
=\frac{w_i}{\sum_{j=1}^{N}w_j},
\qquad
\sum_{i=1}^{N}\bar{w}_i=1.
$$

目标期望即可用加权样本平均近似：

$$
\mathbb{E}_{q}[g(\mathbf{x},Y)]
\approx
\sum_{i=1}^{N}
\bar{w}_i g(\mathbf{x}_i,y_i).
$$

直观上，若某个输入状态在独立干预下应该更常出现、但在相关观测数据中较少出现，它就会获得更大的权重；反之则获得更小的权重。重加权后，原始观测样本整体上近似服从目标干预分布。

#### A.3.3 本实验如何估计密度比

本实验先对每一列输入独立随机置换，构造近似服从乘积边缘分布的参考样本

$$
\widetilde{\mathbf{x}}_i
=
\left(
x_{\pi_1(i),1},
\ldots,
x_{\pi_d(i),d}
\right)^\mathsf{T},
$$

其中 $\pi_1,\ldots,\pi_d$ 是相互独立的随机排列。独立置换保留每个输入维度的边缘分布，同时破坏维度之间的相关性，因此在本实验中近似生成

$$
q_{\mathrm{do}}(\mathbf{x})
=\prod_{j=1}^{d}p_{\mathrm{obs}}(x_j).
$$

随后使用两样本 kNN 密度比估计。对观测点 $\mathbf{x}_i$，记：

- $r_{p,i}$ 为它到观测样本中第 $k$ 个其他近邻的距离；
- $r_{q,i}$ 为它到独立置换样本中第 $k$ 个近邻的距离；
- $N$ 和 $M$ 分别为观测样本与置换样本数量；
- $d$ 为输入维度。

kNN 密度估计中的单位球体积和 $k$ 会在密度比中抵消，得到

$$
\widehat{w}_i
=
\frac{\widehat{q}_{\mathrm{do}}(\mathbf{x}_i)}
{\widehat{p}_{\mathrm{obs}}(\mathbf{x}_i)}
=
\frac{N-1}{M}
\left(
\frac{r_{p,i}}{r_{q,i}}
\right)^d.
$$

本实验取 $M=N$、$d=2$ 和 $k=20$。这一过程只使用输入样本 $\mathbf{x}_i$，完全不使用输出回归模型。

#### A.3.4 从重加权样本计算 EI

实验使用同一个五阶多项式三角传输映射（transport map，TM）作为所有方法的互信息读取器。令 $\mathbf{z}$ 表示联合变量，三角映射 $\mathbf{t}$ 将目标密度映射到标准高斯参考密度 $\phi$，则

$$
\widehat{q}(\mathbf{z})
=
\phi\!\left(\mathbf{t}(\mathbf{z})\right)
\left|
\det\nabla\mathbf{t}(\mathbf{z})
\right|.
$$

RW-EI 使用归一化权重 $\bar{w}_i$ 分别拟合加权联合密度 $\widehat{q}_{\mathbf{x}Y}$、输入边缘密度 $\widehat{q}_{\mathbf{x}}$ 和输出边缘密度 $\widehat{q}_Y$。最终估计量为

$$
\widehat{\operatorname{EI}}_{\mathrm{RW}}
=
\sum_{i=1}^{N}
\bar{w}_i
\log_2
\frac{
\widehat{q}_{\mathbf{x}Y}(\mathbf{x}_i,y_i)
}{
\widehat{q}_{\mathbf{x}}(\mathbf{x}_i)
\widehat{q}_Y(y_i)
}.
$$

因此，RW-EI 的完整计算流程可以概括为

$$
\text{观测样本}
\longrightarrow
\text{估计输入密度比}
\longrightarrow
\text{归一化权重}
\longrightarrow
\text{加权 TM 密度}
\longrightarrow
\widehat{\operatorname{EI}}_{\mathrm{RW}}.
$$

### A.4 ESS：RW-EI 是否可靠的快速诊断

重要性重加权的主要风险是权重过度集中。归一化权重对应的有效样本量定义为

$$
\operatorname{ESS}
=
\frac{\left(\sum_{i=1}^{N}w_i\right)^2}
{\sum_{i=1}^{N}w_i^2}
=
\frac{1}
{\sum_{i=1}^{N}\bar{w}_i^2}.
$$

它可以理解为：“当前这组不均匀加权样本，大约相当于多少个等权样本。”

- 若所有权重都相同，则 $\operatorname{ESS}=N$，说明观测分布与目标干预分布非常接近；
- 若权重集中在少量样本上，则 $\operatorname{ESS}\ll N$，说明目标干预依赖观测数据中的稀有区域；
- 实际比较中通常使用 $\operatorname{ESS}/N$，使不同样本规模之间可以直接比较。

ESS 主要反映权重集中造成的方差膨胀与分布重叠风险。它越低，RW-EI 越容易受到少数高权重点影响。因此，ESS 适合作为 RW-EI 的“预警灯”：高 ESS 通常可以放心使用，低 ESS 则应转向 MLP EI 或至少同时报告模型法结果。

不过，ESS 是风险指标，而不是准确性的证明。它不能单独发现密度比估计偏差、TM 近似误差或未观测混杂，因此仍应与权重分布和必要的模型诊断结合使用。

### A.5 合并结果图：准确性、稳健性与效率

下图把准确性、相关性、ESS 与运行时间放在同一个六面板视图中。图 a 与图 e 共用样本量横轴，图 b 与图 c 共用相关系数横轴。准确性实验在固定 $\rho=0.5$ 时扫描 7 个样本量（$N=1{,}000$ 至 $64{,}000$）；稳健性实验在固定 $N=8{,}000$ 时扫描 10 个相关系数（$\rho=0$ 至 $0.9$）。每个条件使用 30 个配对随机种子。运行时间实验对同一组 7 个样本量使用 10 个配对随机种子，并报告中位数及四分位区间。所有方法采用相同动力学、干预支持集、TM 阶数和信息单位，因而差异主要来自“如何得到干预联合分布”。

黑色虚线仅表示 EI 真值，用于检验两种 EI 估计是否接近已知动力学。

灰色的 **Oracle samples + TM** 曲线和黑色 EI truth 虚线都使用已知动力学，但含义不同：灰色曲线是“真实动力学采样后再由 TM 估计”的有限样本结果，黑色虚线是“不经过 TM、直接数值积分”的参照值。两者的计算公式与误差来源见第 A.2.1–A.2.3 节。

![RW-EI 准确性、稳健性与运行效率的合并结果](../../exp/reweighted_ei/results/rw_ei_combined_results.svg)

#### A.5.1 准确性、相关性与 ESS

**图 a：小样本时 RW-EI 更有优势。** 当观测样本数为 $N=1{,}000$ 时，MLP EI 的平均绝对误差（MAE）为 0.0592 bit，而 RW-EI 为 0.0353 bit。MLP 需要同时学习非线性条件均值和噪声尺度，小样本下更容易产生模型拟合误差；RW-EI 跳过这一步，因此表现更稳定。随着样本量增至 64,000，两者的 MAE 分别稳定到 0.0196 bit 和 0.0152 bit，均位于已知动力学真值附近。

**图 b：强相关是 RW-EI 的主要失效条件。** 在 $\rho\leq0.7$ 的大部分区间内，RW-EI 与 MLP EI 的误差处于相同量级，RW-EI 在 $\rho=0.4$ 至 $0.7$ 还略低于 MLP EI。但在 $\rho=0.8$ 时，RW-EI 的 MAE 突然上升到 0.0766 bit，明显高于 MLP EI 的 0.0357 bit；到 $\rho=0.9$ 时，两者分别增至 0.2428 bit 和 0.1402 bit。原因不是动力学发生变化，而是高度相关的观测输入难以覆盖独立干预所需要的状态组合。MLP EI 对这一问题更耐受，但在极端相关下也不是完全免疫。

**图 c：相关性较弱时，两种方法都能跟随 EI 真值。** 图 b 与图 c 共用相关系数 $\rho$ 横轴，便于同时观察“误差大小”和“估计值偏向哪里”。紫色曲线是普通互信息的 TM 估计；蓝色和红色曲线分别是 MLP EI 与 RW-EI。在 $\rho\leq0.7$ 时，两条 EI 估计曲线总体位于 EI 真值附近；从 $\rho=0.8$ 开始，RW-EI 明显向下偏离，而 MLP EI 到 $\rho=0.9$ 才出现更强的下偏。这说明 RW-EI 的误差主要由输入分布转换难度控制。

**图 d：ESS 能够识别 RW-EI 的风险。** 随着 $\operatorname{ESS}/N$ 降低，RW-EI 的误差幅度总体增大。在 $\rho=0.8$ 和 $0.9$ 时，平均 $\operatorname{ESS}/N$ 分别下降到约 0.284 和 0.236，并与明显的负偏差同时出现。因此，ESS 可以在不知道动力学真值的真实应用中，帮助判断当前 RW-EI 是否值得信任。

#### A.5.2 运行时间与计算开销

图 e 和图 f 比较四种计算路径在不同样本规模下的实际运行时间。计时使用相同进程和相同随机种子集合。

**图 e：RW-EI 始终明显快于 MLP EI。** 随样本量增加，两种方法的时间开销都会增长，但 MLP EI 还要承担网络训练与干预采样成本，其曲线始终远高于 RW-EI。RW-EI 的主要额外成本只是低维 kNN 密度比估计和加权 TM 拟合。

**图 f：最大样本规模下，RW-EI 约快 12.2 倍。** 当 $N=64{,}000$ 时，MLP EI 的中位运行时间为 11.63 秒，RW-EI 为 0.957 秒。RW-EI 只需要 MLP EI 约 $8.2\%$ 的时间，即获得约 12.2 倍的速度优势。

从渐近复杂度看，若 $E$ 为 MLP 训练轮数、$P$ 为网络参数运算规模，则 MLP 路径包含约

$$
\mathcal{O}(ENP)
$$

的训练成本。低维 kNN 密度比估计的平均成本约为

$$
\mathcal{O}(N\log N).
$$

两条路径随后都使用相同的 TM 读取器，因此 RW-EI 的时间优势主要来自省去了反复的神经网络训练。

### A.6 实际使用建议

可以按照下面的简单规则选择方法：

| 数据条件 | 推荐方法 | 原因 |
|---|---|---|
| 样本较少，ESS 较高 | **RW-EI** | 避免小样本下的动力学模型拟合误差 |
| 样本较多，ESS 较高 | **RW-EI** | 精度接近真值，同时计算速度明显更快 |
| 输入相关性很强，ESS 很低 | **MLP EI** | RW-EI 容易受到极端权重和支持不足影响 |
| 无法确定哪种方法可靠 | **先算 RW-EI 和 ESS，再决定是否补充 MLP EI** | ESS 可作为低成本风险诊断 |

需要强调的是，输入变量高度相关并不自动等同于存在未观测混杂。本实验中，强相关造成的核心问题是观测分布与独立干预分布之间缺少足够重叠。若真实系统还存在同时影响输入和输出的未观测因素，则 RW-EI 和 MLP EI 都需要额外的因果识别假设。

### A.7 总结

RW-EI 的核心价值在于：**把“学习动力学”转化为“修正样本分布”**。只要观测数据能够充分覆盖目标干预区域，它就能用一组输入密度比权重直接恢复干预联合分布，并以更低的计算开销获得接近动力学真值的 EI。

本实验给出的选择逻辑非常清楚：

- **小样本：RW-EI 更准；**
- **大样本：两者都稳定在真值附近，但 RW-EI 更快；**
- **强相关：RW-EI 更早失效，优先 MLP EI；极端相关下两者都要谨慎；**
- **是否适合 RW-EI：先看 ESS。**

### A.8 真实高维应用：SEEG 9/27 主成分敏感性检验

为检验“不拟合动力学是否会改善 Attend–Visible 的前后脑协同排序”，在一组 SEEG 注意与可见性实验中冻结前脑 9 个主成分、后脑 27 个主成分、0.25--0.55 s 时间窗、两步历史、条件划分和 block 配对，仅将拟合式 EI 替换为直接 RW-EI。目标输入分布设为前脑历史边缘与后脑历史边缘的乘积；联合、前脑部分和后脑部分 EI 由加权仿射 Gaussian/TM 互信息读取。

结果没有支持 Attend–Visible 的 synergy 显著最高。Evoked-residual 表示下，Attend–Visible 的直接条件协同均值表面排名第一，但相对 Attend–Invisible 只在 3/6 个留出 block 中更高，Holm 校正后 $p=0.25$；raw 表示下则由 Attend–Invisible 排名第一。联合 RW-EI 在两种表示下都由 Attend–Visible 表面排名第一，但相对三个竞争条件的配对比较没有全部通过校正。

该结果的决定性信息来自权重诊断，而不是表面排序。每个估计原有 1,120 个时序对，但 288 个估计的平均 $\operatorname{ESS}/N$ 仅为 0.00589，最低为 0.000893；最极端单个样本占总权重 99.982%。重加权后前脑与后脑历史仍保留平均 22.259 bits 的依赖，说明目标乘积干预没有实现。

因此，这次真实数据尝试进一步限定了第 A.6 节的建议：RW-EI 的速度优势不能抵消高维支持不重叠。对于 72 维历史输入，即使原始样本数超过一千，实际有效样本也可能只剩个位数。此时不能把表面较高的条件互信息解释为可靠的 PEID synergy；应降低到由独立预测或解剖规则预先确定的宏变量维度，并重新要求 ESS、最大单点权重与加权后残余源依赖同时通过诊断。

### A.9 SURD 与 PEID 为什么没有给出同一排序

#### A.9.1 公平比较的设置

为了判断差异是否来自实现细节，PEID 复核固定了与 SURD 相同的前脑 9 / 后脑 27 主成分、0.25--0.55 s 时间窗、VAR(2) 历史和 Present--Absent 差值。Animal、object 和 face 没有分别拟合模型：每个 attention × visibility × stimulus-presence 单元只拟合一个转移模型，三类刺激合并后共有 60 个训练 trial 和 12 个留出 trial。类别标签只用于训练折内减去类别平均诱发波形；raw 表示不使用这一步。

这次比较只改变 PEID 的干预支持，其余数据、模型和统计单位保持不变：

| PEID 干预支持 | Attend--Visible 的 Present--Absent synergy（bit） | 四条件排名 | 最高条件 |
| --- | ---: | ---: | --- |
| 各 PC × lag 独立、单位协方差 | 0.01036 | 3 | Unattend--Visible |
| 前后脑块独立、块内保留相关矩阵 | 0.01004 | 3 | Unattend--Visible |
| 前后脑块独立、块内保留协方差矩阵 | 0.00819 | 3 | Unattend--Visible |

三种支持下，Attend--Visible 都只在 4/6 个 block 中为正，单侧精确检验均未达到 0.05；改变独立性的粒度没有恢复 SURD 的条件排序。块协方差的最大条件数约为 11.5，也没有出现明显数值病态。因此，当前差异不能主要归因于“单位高斯干预过度打散了前脑或后脑内部结构”。

#### A.9.2 高斯不是 PEID 的定义

PEID 的必要结构是源侧最大熵干预及源变量独立，而不是高斯分布本身。对前脑历史向量 $\mathbf{f}$、后脑历史向量 $\mathbf{p}$ 和未来状态 $\mathbf{y}$，干预联合分布写为

$$
q(\mathbf{f},\mathbf{p},\mathbf{y})
=q_F(\mathbf{f})q_P(\mathbf{p})
p(\mathbf{y}\mid\mathbf{f},\mathbf{p}).
$$

离散变量通常在有限状态空间上采用均匀分布；连续 PEID 也可以在预先规定的有界支持上采用独立均匀分布。PEID 论文的连续实验正是这样采样。不过，该论文随后使用仿射高斯 transport map 读取互信息。对于当前线性转移模型，这种读取器只依赖二阶协方差。因此，具有相同协方差的独立均匀干预与高斯干预会得到同一个仿射闭式结果。高斯闭式是当前估计器的性质，不是 PEID 的定义。

若要真正比较均匀与高斯的分布形状，必须同时换成能够识别非高斯密度的读取器。这里不能只把高斯随机数换成均匀随机数，却继续使用协方差 log-det 公式，然后把相同结果解释为“均匀干预也失败”。此外，连续均匀干预必须先规定边界；不同单元分别使用样本最小值和最大值，会让干预支持随条件和 Present/Absent 改变，从而把机制差异与支持差异混在一起。

#### A.9.3 两种方法估计的不是同一个量

SURD 使用观测联合分布 $p(\mathbf{f},\mathbf{p},\mathbf{y})$。当前实现先计算逐目标事件的特异信息，然后定义

$$
R_{\mathrm{SURD}}
=\mathbb{E}\!\left[\min\{i_F(\mathbf{y}),i_P(\mathbf{y})\}\right],
$$

$$
S_{\mathrm{SURD}}
=\mathbb{E}\!\left[i_{FP}(\mathbf{y})-
\max\{i_F(\mathbf{y}),i_P(\mathbf{y})\}\right].
$$

它保留了真实数据中前后脑状态的出现频率、相关性和共同驱动。PEID 则先把源侧分布替换为 $q_Fq_P$，再通过拟合的条件机制产生未来状态。对任意满足 $\mathbf{f}\perp\mathbf{p}$ 的干预分布，PEID 的两源 synergy 为

$$
\begin{aligned}
S_{\mathrm{PEID}}
&=I_q(\mathbf{f},\mathbf{p};\mathbf{y})
-I_q(\mathbf{f};\mathbf{y})
-I_q(\mathbf{p};\mathbf{y})\\
&=I_q(\mathbf{f};\mathbf{p}\mid\mathbf{y})\geq 0.
\end{aligned}
$$

这两个量只有在使用同一个联合分布、且源侧冗余为零等额外条件下才可能一致。因此，“SURD 得到预期排序”并不推出“PEID 必须得到同一排序”。前者回答观测条件下信息怎样被分解，后者回答在指定独立干预下拟合机制产生多少不可加的因果信息。

真实结果直接显示了这一区别。在 Attend--Visible 条件下，evoked-residual 的 SURD synergy 从 Absent 到 Present 增加 0.06987 bit，但 SURD redundancy 同时增加 0.27839 bit。因此

$$
\Delta\mathrm{WMS}
=\Delta S_{\mathrm{SURD}}-\Delta R_{\mathrm{SURD}}
=-0.20858\ \text{bit}.
$$

raw 表示得到同一模式：synergy 增加 0.08420 bit，redundancy 增加 0.25504 bit，而 WMS 降低 0.17089 bit。换言之，Attend--Visible 的观测 SURD 阳性结果伴随着更强、而不是更弱的冗余增长。SURD 的 Attend--Visible 平均值确实为四条件最高，并在对三个竞争条件的配对比较中通过 Holm 校正；但它不是“去除冗余后的净交互”。PEID 和 WMS 没有复现该结果，正是因为它们对冗余和观测源分布的处理不同。

#### A.9.4 更适合本问题的冻结参数

后续若继续检验 PEID，建议预先冻结以下设置：

1. Animal、object 和 face 始终合并拟合；不报告类别特异模型作为主结果。
2. 主效应始终是每个条件内的 Present--Absent synergy，而不是四个条件的绝对 synergy 大小。
3. 源分区固定为两个多维宏变量：前脑历史 $\mathbf{f}$ 与后脑历史 $\mathbf{p}$；不要在看到结果后改变 PC 分组。
4. 干预分布采用所有训练条件共同的支持，且留出 block 不参与边界或尺度估计。对已经标准化的 PC，一个无需按结果调节的候选是每个 PC × lag 独立服从 $\operatorname{Uniform}[-\sqrt{3},\sqrt{3}]$，使每维方差固定为 1。
5. 将高斯与均匀作为预注册的 estimator sensitivity，而不是选择 Attend--Visible 排名更高者。均匀干预必须使用真正的非高斯密度读取器；仿射高斯读取只能作为协方差基线。
6. Ridge 强度、lag 和噪声收缩继续只按留出预测与似然选择，不按 synergy 排名选择。现有 alpha 300、lag 2、noise shrinkage 0.1 都有独立预测筛选依据，暂时没有证据说明它们是 SURD--PEID 差异的主因。

当前证据支持的最窄结论是：统一类别合并、Present--Absent 对比和前后脑分区后，PEID 仍未复现 SURD 的 Attend--Visible 排序；进一步保留块内协方差也没有改变结论。下一项有意义的检验不是继续搜索 Ridge 或 PCA 参数，而是在固定的 $\operatorname{Uniform}[-\sqrt{3},\sqrt{3}]$ 干预下，使用能识别非高斯输出密度且通过模拟校准的连续互信息估计器。考虑到源历史有 72 维，这一步必须同时报告有限样本偏差与重复稳定性，否则仍可能只是把 RW-EI 的高维不稳定换成另一种密度估计不稳定。

#### A.9.5 能否用前后脑相关直接替代冗余

普通前后脑相关不能复现 SURD 的 Attend--Visible 最高。为避免不同 PC 的正负相关相互抵消，使用所有前脑--后脑 Pearson 相关平方的均方根作为块相关强度。该指标不拟合转移模型，也不使用未来目标。Evoked-residual 表示下，历史块相关的 Present--Absent 增量为：

| 条件 | 历史 RMS 相关增量 | 排名 |
| --- | ---: | ---: |
| Attend--Invisible | 0.001940 | 3 |
| Attend--Visible | 0.003708 | 2 |
| Unattend--Invisible | 0.001938 | 4 |
| Unattend--Visible | 0.005045 | 1 |

不含 lag 的同时刻相关得到同一排序：Attend--Visible 增加 0.003627，而 Unattend--Visible 增加 0.005490。Raw 表示也完全一致。Attend--Visible 虽然在 6/6 个 block 中均为正，却显著低于 Unattend--Visible，因此不能称为四条件显著最高。

多变量 Gaussian source MI 也支持这一结论。Evoked-residual 下，前后脑源 MI 的 Present--Absent 增量在 Unattend--Visible 和 Attend--Visible 中分别为 0.5830 与 0.2967 bit；raw 下分别为 0.5660 与 0.2083 bit。因此，SURD 冗余的 Attend--Visible 特异性不是简单的源间相关或同步增强。

一个更简单且概念上正确的替代量是 MMI redundancy：

$$
R_{\mathrm{MMI}}
=\min\left\{
I(\mathbf{f};\mathbf{y}),
I(\mathbf{p};\mathbf{y})
\right\}.
$$

它仍然依赖未来目标 $\mathbf{y}$，但不需要 SURD 的逐事件特异信息分解。该指标的 Attend--Visible Present--Absent 增量在 evoked-residual 和 raw 下分别为 0.27851 和 0.25523 bit，都是四条件最高。相对另外三个条件的单侧配对检验在 Holm 校正后均为 $p=0.046875$。它与当前 SURD redundancy 的 192 个单元估计相关系数为 0.9984，平均绝对差仅 0.00669 bit。

因此，若研究问题只是“图片呈现是否让前后脑对未来状态具有更多重叠信息”，$R_{\mathrm{MMI}}$ 是当前最简单、最贴近结果的主指标。Pearson 相关或前后脑 source MI 回答的是“两个源彼此是否更相关”，没有目标变量，不能等同于信息冗余。

<!-- report-section:sine-frequency:start -->
<a id="sine-frequency"></a>

## 附录 B：固定干预支持下的 Sine 校准

本实验检验：当只改变振幅参数 $\alpha$ 或响应面的空间频率 $k$ 时，MLP+PEID 的 ${x,y}\rightarrow z$ 协同读数如何变化。这里的 $k$ 是 $x_ty_t$ 响应面上的空间振荡频率，不是时间采样频率。

$$
\begin{aligned}
x_{t+1} &= 0.42x_t + \eta^x_t,\\
y_{t+1} &= 0.38y_t + \eta^y_t,\\
z_{t+1} &= 0.22z_t + \alpha\sin(kx_ty_t) + \eta^z_t.
\end{aligned}
$$

### 受控比较协议

- treatment：$\alpha\in[0.25, 0.5, 1.0, 1.5, 2.0]$ 与 $k\in[1.0, 2.0, 4.0, 6.0, 8.0, 10.0]$ 的全因子扫描；
- pairing：每个 seed 的同一批 `2048` 个干预状态复用于全部 $(\alpha,k)$ 条件；
- support：$x,y\in[-1.8, 1.8]$，$z\in[-1.25, 1.25]$，在全部条件中固定；
- readout：learned MLP 与 known dynamics 使用完全相同的干预状态和 TM 估计协议；
- estimator：先从统一的 $1,\ldots,10$ 阶固定谐波字典中自动选择响应最强的谐波，再在未参与选择的另一半样本上运行 affine triangular TM；交换两半样本后取平均。该 cross-fitting 协议不读取当前条件的真实 $k$；
- diagnostic：$R^2$ 在固定干预支持上针对无噪声条件均值计算，而不是训练集 $R^2$；
- nonnegativity：原生 Syn 单位中的容差为 `0.01` bits；显著违规数为 `0`。

训练协议固定为 `1100` 个轨迹样本、noise `0.05`、`90` epochs 和 seeds `[0, 1, 2, 3]`。系统没有共同驱动或隐藏变量。

![无 confounder sine frequency sweep](../../fig/granger_peid_mlp_comparison/sine_frequency_mlp_peid_sweep.png)

*图｜固定干预支持下的振幅—空间频率校准。a，learned MLP 从 1–10 阶候选字典中选择各谐波的频率；每个真实 $k$ 汇总 5 个 $\alpha$ 条件和 4 个 seeds。b，learned MLP 与 known-dynamics TM 的 Syn；点为跨 $\alpha$ 和 seed 的均值，阴影为相应标准差。c，各 $k$ 下 learned Syn 随 $\alpha$ 的变化；点和误差棒分别为 4 个 seeds 的均值和标准差。d，固定干预支持上的条件均值预测 $R^2$；点和误差棒分别为 4 个 seeds 的均值和标准差。*

### 结果判断

1. **Known-dynamics 基准能够识别空间频率。** 在不读取真实 $k$ 的 cross-fitted 选频协议下，真实谐波恢复率为 `100.0%`。
2. **当前 learned MLP 没有复现高频识别。** 总体真实谐波恢复率为 `16.7%`，成功条件主要集中在 $k=1$。固定支持 $R^2$ 在 $k=1$ 时为 `0.884`，而 $k>1$ 时各条件均值仅为 `0.157`–`0.256`。
3. **振幅不变性没有复现。** Known-dynamics Syn 随 $\alpha$ 稳定增加，其各 $k$ 条件的斜率约为 `0.975`–`0.997` bits / unit $\alpha$。旧结果中近似水平的振幅曲线主要来自低阶 TM 特征饱和，不能解释为 PEID 对物理振幅严格不敏感。

### 汇总结果

| $k$ | Learned Syn | Known-dynamics Syn | Fixed-support $R^2$ | Learned Syn range across $\alpha$ |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 1.212 | 1.267 | 0.884 | 1.796 |
| 2.0 | 0.8693 | 1.318 | 0.2393 | 0.8394 |
| 4.0 | 0.394 | 1.341 | 0.2559 | 0.3102 |
| 6.0 | 0.2696 | 1.355 | 0.2051 | 0.2079 |
| 8.0 | 0.18 | 1.362 | 0.2179 | 0.1568 |
| 10.0 | 0.146 | 1.368 | 0.1574 | 0.1302 |

固定 $k$ 时沿振幅方向的线性斜率：

| fixed $k$ | Learned Syn slope / $\alpha$ | Known-dynamics slope / $\alpha$ | Fixed-support $R^2$ slope / $\alpha$ |
| ---: | ---: | ---: | ---: |
| 1.0 | 1.008 | 0.9749 | 0.08551 |
| 2.0 | 0.4318 | 0.9888 | -0.1243 |
| 4.0 | 0.1711 | 0.9939 | -0.1096 |
| 6.0 | 0.1087 | 0.9962 | -0.08068 |
| 8.0 | 0.08291 | 0.9967 | -0.102 |
| 10.0 | 0.07027 | 0.9934 | -0.101 |

### 解释边界

“Known dynamics” 是在已知条件均值函数上运行相同 TM 估计器所得的机制基准，不是解析真值。Learned 与 known-dynamics 曲线的差异同时反映有限轨迹学习误差与有限样本 TM 误差。只有在固定支持 $R^2$ 保持良好时，才可把 Syn 随 $k$ 的变化主要解释为对响应面几何的敏感性；若二者同时下降，则应解释为 surrogate 分辨率边界。
<!-- report-section:sine-frequency:end -->

<a id="ei-koopman-control"></a>

## 附录 C：EI—Koopman—控制的理论边界与初步验证

日期：2026-09-14。本文区分标准信息论/控制论恒等式、本次可复现实验结果和后续研究假设。文献证据及阅读顺序见 [文献对照](exploration.md#ei-control-literature)。

### C.1 结论与已有基础

**建议继续，但主问题应当是“哪些多尺度表示保留了完成干预任务所需的信息”，而不是“最大化 EI 或 Syn 就能最小化控制能量”。** Koopman 提供可计算的动力学表示，NIS+ 提供表示选择方法，PEID 提供联合输入结构；控制目标、执行器约束与成本仍须显式加入。

现有仓库已经做过相当直接的桥接，不能把下一步描述成从零开始：

| 已有证据 | 对本方向的意义 | 尚未证明的部分 |
| --- | --- | --- |
| 联合点火报告 §1–4（原报告已缺失）：最终 basin 的 Syn 能识别有效组合，但部分基线同样有效；Syn 与最小幅度成本的相关弱 | 机制组合识别可以作为控制入口 | 组合识别不等于成本排序；原成本也不等于二次能量 |
| 同报告 §7（原报告已缺失）：廉价短时代理和 TM 初态代理未稳定恢复最终点火效果 | 时间尺度和目标读出不能任意替换 | 不能默认短时机制指标足够指导跨 basin 控制 |
| 同报告 §9（原报告已缺失）：人工层级锁存器上，按节点数归一化的预干预联合增量与总能量负相关 | 存在与任务对齐的正对照 | 单实例、受控门结构、执行成本与增益基本对齐 |
| 同报告 §9 的自然网络压力测试（原报告已缺失）：20 节点 Neural/Eco、ER/WS 网络未稳定复现上述负相关 | 已存在明确失败证据 | 下一步应解释或修复失败，而非只扩展成功例子 |

本次复用 [联合锁存器实现](../../exp/network_revival/joint_required_ignition.py)，新增两个小实验，不重新训练 NIS+，也不把点火搜索称为完整闭环控制。研究框架旧路径在 README 中仍有引用，但当前文件不存在，因此本报告以可读取的论文、代码和控制报告为依据。

### C.2 先区分三个问题

设真实受控系统为

$$
\mathbf{x}_{t+1}=f(\mathbf{x}_t,\mathbf{u}_t,\boldsymbol{\xi}_t),
\qquad \mathbf{z}_t=\psi(\mathbf{x}_t).
\tag{C.2.1}
$$

向量用粗体小写，矩阵用粗体大写。本文理论公式使用自然对数，单位为 nat；离散点火实验使用 bit，结果单独标明。

**状态机制 EI。** 固定状态干预分布 $q_x$ 后，计算 $I_{q_x}(\mathbf{x}_t;\mathbf{x}_{t+1})$。只有明确支持集、参考测度或约束后，“最大熵干预”才有确定意义。连续全空间不存在均匀概率分布。

**动作到结果的信息。** 给定初始状态，随机化真实可施加的动作序列 $\mathbf{u}=(\mathbf{u}_0,\ldots,\mathbf{u}_{T-1})$，定义

$$
\mathcal I_{q_u}^{T}(\mathbf{x}_0)
=I_{q_u}\!\left(\mathbf{u};g(\mathbf{x}_T)\mid\mathbf{x}_0=\mathbf{x}_0\right).
\tag{C.2.2}
$$

$g$ 是任务读出，例如脑区群活动或鸟群方向。式 (C.2.2) 是与控制直接相关的干预信息；它与对全部状态做均匀重置的 EI 不是同一实验。若进一步在相同可行动作和预算约束下优化 $q_u$，就接近既有的 empowerment/通道容量问题，而非全新指标。[Empowerment 综述](https://arxiv.org/abs/1310.1863)

**指定目标的控制成本。** 最小能量问题则是

$$
J^\star(\mathbf{x}_0,\mathcal G)
=\inf_{\pi}\;\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}\mathbf{u}_t^\top\mathbf{R}\mathbf{u}_t\right],
\quad
\Pr_\pi\{g(\mathbf{x}_T)\in\mathcal G\}\ge 1-\delta,
\tag{C.2.3}
$$

并满足执行器、幅度和状态约束；$\mathbf{R}\succ0$ 表示操作成本。动作产生的结果容易区分，并不保证容易到达某个指定结果。EI 不携带“希望输出是什么”这一偏好，也不会自动知道电流、药物或通信成本。

### C.3 Koopman 表示与原系统的 EI：可以先证明什么

#### C.3.1 同一干预下：可逆变换保持信息，粗粒化只能丢失信息

**命题 1（标准信息论恒等式）。** 固定真实干预联合分布 $q_x(\mathbf{x})p(\mathbf{y}\mid\mathbf{x})$，令 $\mathbf{z}=\psi(\mathbf{x})$、$\mathbf{w}=\psi(\mathbf{y})$ 为确定性可测编码，且相关互信息有限。则

$$
I(\mathbf{x};\mathbf{y})-I(\mathbf{z};\mathbf{w})
=I(\mathbf{x};\mathbf{y}\mid\mathbf{z})
+I(\mathbf{z};\mathbf{y}\mid\mathbf{w})\ge0.
\tag{C.3.1}
$$

证明：分别用链式法则展开 $I(\mathbf{x};\mathbf{y})=I(\mathbf{z};\mathbf{y})+I(\mathbf{x};\mathbf{y}\mid\mathbf{z})$ 和 $I(\mathbf{z};\mathbf{y})=I(\mathbf{z};\mathbf{w})+I(\mathbf{z};\mathbf{y}\mid\mathbf{w})$，相减即可。若编码在输入、输出分布的支持上均有可测逆，两个条件项均为零，因此 EI 完全相等。即使编码升维，只要它单射并保留真实联合分布，这个结论仍成立。

该等式与 Koopman 是否线性无关；Koopman 的价值在于让右侧的动态信息可计算，而非通过坐标变化创造信息。原系统确定、连续且无测量噪声时，互信息可能为无穷，不能直接用两个无穷相减；必须先规定噪声或有限分辨率。

#### C.3.2 每个尺度重新做均匀干预：比较的分布已经变了

若 $\psi$ 可逆且光滑，原干预在潜在空间的推前密度为

$$
q_z(\mathbf{z})
=q_x(\psi^{-1}(\mathbf{z}))\left|\det D\psi^{-1}(\mathbf{z})\right|.
\tag{C.3.2}
$$

它一般不是均匀分布。因此“原空间均匀 EI”与“潜在空间重新均匀 EI”可能不同，不违反式 (C.3.1)。应分别报告：同一物理干预的编码前后信息、每个尺度重新定义干预后的 EI、以及维度归一化 EI；三者不能混用。

若编码降维，还须指定实现宏观干预的微观条件分布 $\ell(\mathbf{x}\mid\mathbf{z})$。同一宏观值可能对应多个不同的微观操作和成本。若编码升维，合法潜在状态往往位于低维流形上；在整个潜在长方体独立均匀抽样会生成没有物理对应的状态。

#### C.3.3 线性条件均值不等于线性高斯转移核

随机 Koopman 关系 $\mathbb E[\psi(\mathbf{x}_{t+1})\mid\mathbf{x}_t]=\mathbf{K}\psi(\mathbf{x}_t)$ 只约束条件均值。EI 依赖完整转移分布，还需要噪声的条件协方差、形状及其随状态的变化。原空间的加性高斯噪声经过非线性编码后，通常不再是加性、同方差高斯噪声。

受控情况下还要验证所有相关输入下的闭合，而不只是无控制轨迹。一般控制仿射系统经非线性 lifting 后可出现状态与输入的乘积；有限维双线性模型、局部模型或近似线性预测器都可能比强行假设全局线性更合适。[Koopman MPC](https://arxiv.org/abs/1611.03537)、[2026 控制综述](https://doi.org/10.1016/j.arcontrol.2025.101035)

#### C.3.4 一个可以解析计算的基准

若真正具有

$$
\mathbf{z}_{t+1}=\mathbf{K}\mathbf{z}_t+\boldsymbol{\epsilon}_t,
\quad \mathbf{z}_t\sim\mathcal N(\mathbf0,\mathbf{S}),
\quad \boldsymbol{\epsilon}_t\sim\mathcal N(\mathbf0,\boldsymbol{\Sigma}),
\tag{C.3.3}
$$

且两者独立、$\boldsymbol{\Sigma}\succ0$，则由高斯熵公式直接得到

$$
I(\mathbf{z}_t;\mathbf{z}_{t+1})
=\frac12\log\det\!\left(\mathbf{I}+\boldsymbol{\Sigma}^{-1/2}\mathbf{K}\mathbf{S}\mathbf{K}^{\top}\boldsymbol{\Sigma}^{-1/2}\right).
\tag{C.3.4}
$$

这是固定协方差高斯干预下的精确值，不是任意有界均匀干预的精确公式。相同协方差、任意非高斯输入时，式 (C.3.4) 是上界，因为高斯分布在固定协方差下最大化输出熵。

Liu 等的[线性随机 EI 论文](https://arxiv.org/abs/2405.09207)是直接相关的起点，但使用其中的行列式表达式前需核对干预与边界处理。对有限均匀输入，输出是均匀分布经过线性变换再与高斯卷积，通常既不均匀也不高斯。形如 $\log(|\det\mathbf{K}|L^d)-h(\boldsymbol{\epsilon})$ 的表达不能无条件当作真实 MI：它可能为负。本文不将该式作为通用精确基准；在非奇异、小噪声条件下才考虑其作为近似的适用性。

数值路线很直接：先固定物理干预，成对模拟原空间与编码空间，再用同一 TM 协议估计；低维已知转移核用积分作为参考；只有满足式 (C.3.3) 时才使用式 (C.3.4) 的闭式。特征间的代数依赖会使升维样本协方差奇异，不能把密度估计器硬套在整个环境空间。

### C.4 EI 与控制能量的解析桥梁，以及它的边界

#### C.4.1 二者可以共享一个矩阵，但使用矩阵的方式不同

固定线性受控动力学、初态、时域和执行器：

$$
\mathbf{z}_{t+1}=\mathbf{A}\mathbf{z}_t+\mathbf{B}\mathbf{u}_t,
\qquad \mathbf{y}_T=\mathbf{C}\mathbf{z}_T.
\tag{C.4.1}
$$

令 $\mathbf{H}_T=[\mathbf{C}\mathbf{A}^{T-1}\mathbf{B},\ldots,\mathbf{C}\mathbf{B}]$，$\mathbf{R}_T=\mathbf{I}_T\otimes\mathbf{R}$，目标位移为 $\mathbf{d}=\mathbf{y}^\star-\mathbf{C}\mathbf{A}^T\mathbf{z}_0$。定义输出可控性 Gramian

$$
\mathbf{G}_T=\mathbf{H}_T\mathbf{R}_T^{-1}\mathbf{H}_T^\top.
\tag{C.4.2}
$$

**命题 2（标准最小范数解）。** 无幅度/路径约束、确定性终端等式约束下，若 $\mathbf{d}\in\operatorname{range}(\mathbf{H}_T)$，则

$$
J^\star=\mathbf{d}^{\top}\mathbf{G}_T^{\dagger}\mathbf{d};
\tag{C.4.3}
$$

否则不可达。证明：令 $\mathbf{v}=\mathbf{R}_T^{1/2}\mathbf{u}$，在 $\mathbf{H}_T\mathbf{R}_T^{-1/2}\mathbf{v}=\mathbf{d}$ 下求最小欧氏范数即可。

为测量动作信息，在同一终端映射上加入独立观测噪声 $\boldsymbol{\eta}\sim\mathcal N(\mathbf0,\boldsymbol{\Sigma}_y)$，并用动作干预 $\mathbf{u}\sim\mathcal N(\mathbf0,\alpha\mathbf{R}_T^{-1})$。于是

$$
\mathcal I_{q_u}^{T}
=\frac12\log\det\!\left(\mathbf{I}+\alpha\boldsymbol{\Sigma}_y^{-1/2}\mathbf{G}_T\boldsymbol{\Sigma}_y^{-1/2}\right).
\tag{C.4.4}
$$

这就是连接：**信息衡量噪声尺度下可分辨输出的整体扩展，能量衡量到指定方向的难度。** 这是控制 Gramian 与高斯通道的组合推导，不能宣称为本项目首次发现。若共有 $mT$ 个动作自由度，则该干预的平均能量为 $\alpha mT$；比较不同输入维度时应固定总预算 $E_0$，取 $\alpha=E_0/(mT)$。式 (C.4.3) 不是有过程噪声和概率约束时的完整随机最优控制解。

可挽救的充分条件是矩阵偏序：相同噪声和预算下，若 $\mathbf{G}_1\succeq\mathbf{G}_2\succ0$，则信息不减且每个目标方向的能量不增。但一个标量 EI 的大小不足以推出这个矩阵偏序。

#### C.4.2 高 EI 不保证低能量：二维反例

取 $\alpha=1$、$\boldsymbol{\Sigma}_y=\mathbf{I}$、$\mathbf{d}=(0,1)^\top$，令

$$
\mathbf{G}_1=\operatorname{diag}(100,0.01),\qquad
\mathbf{G}_2=\operatorname{diag}(1,1).
\tag{C.4.5}
$$

则 $\mathcal I_1=2.312535>\mathcal I_2=0.693147$ nat，却有 $J_1^\star=100>J_2^\star=1$。甚至交换同一个 Gramian 的两个对角元，EI 完全不变，到第二坐标的成本仍能相差一万倍。因此仅凭谱的聚合量，通常不能确定具体目标的控制难度。

同样，固定无控制机制 $\mathbf{A}$ 和噪声但改变 $\mathbf{B}$，状态 EI 不变，可控性却会改变；固定动力学和干预分布但改变 $\mathbf{R}$，EI 不变，最优操作也可改变。

#### C.4.3 Syn 的操作含义不等于“非线性协作降低成本”

按 PEID 论文的二源定义，在同一独立干预联合分布下

$$
\operatorname{Syn}(U_1,U_2;Y)
=I(U_1,U_2;Y)-I(U_1;Y)-I(U_2;Y)
=I(U_1;U_2\mid Y)\ge0.
\tag{C.4.6}
$$

即使纯线性通道 $Y=U_1+U_2+\epsilon$，其中独立高斯输入方差均为 $v$、噪声方差为 $\sigma^2$，也有

$$
\operatorname{Syn}=\frac12\log\frac{(1+s)^2}{1+2s}>0,
\qquad s=v/\sigma^2>0.
\tag{C.4.7}
$$

式 (C.4.7) 是将式 (C.4.6) 的信息恒等式应用于连续有限 MI 通道；高斯输入是明确的协方差约束干预，非原论文离散均匀实验。正 Syn 本身不证明乘法机制、单点不可控或能量优势。

此外，若确定性二值目标为 $Y=\mathbf1\{\text{成功}}$，joint EI 恰为 $H(Y)=h_2(p_{\mathrm{success}})$，其最大值位于成功率 $1/2$。成功率接近 1 时，控制很可靠，但该 EI 接近 0。这是任务效用与信息区分能力的具体差别。

应把 PEID 的源保留为有物理含义的独立执行器或动作块。混合多个执行器的可逆潜在变换虽保持总 MI，却不一定保持各分块 Syn；在混合后的坐标上重新独立干预，又改变了原干预协议。闭环策略产生相关动作时，也不能直接套独立源的式 (C.4.6)，需另设条件随机干预协议。

### C.5 本次小实验

#### C.5.1 非线性原系统与精确随机 Koopman 表示

取标量系统 $X_{t+1}=\operatorname{asinh}(0.8\sinh X_t+\epsilon_t)$，其中 $\epsilon_t\sim\mathcal N(0,0.25^2)$。特征 $Z_t=\sinh X_t$ 满足精确线性高斯转移。先令 $Z_t\sim\mathcal N(0,1)$，并将此分布拉回原空间。式 (C.3.4) 给出两种坐标共有的真值 **1.209739 nat**。

3 个固定种子、每个 4096 个成对样本，degree-3 TM 的均值分别为潜在空间 **1.204713**、原空间 **1.196393 nat**。原/潜在坐标的估计偏差不同，不是 EI 在编码后真实增加；3 个种子也不足以证明估计器一致性。

再分别采用 $X_t\sim U[-1,1]$ 和 $Z_t\sim U[-\sinh1,\sinh1]$，噪声与基础随机分位数成对共享。它们支持对应，但概率权重不同。已知通道积分参考分别为 **0.800975** 和 **0.838970 nat**，差为 **0.037995 nat**；TM 均值分别为 **0.808803** 和 **0.859886 nat**。结果明确表明，重新均匀化的收益不能归因于单纯坐标线性化。

![原空间与 Koopman 空间的信息比较](../../fig/koopman_ei_invariance_pilot.png)

*图 1。a，同一高斯潜在干预及其原空间拉回分布；每条灰线连接同种子的两种坐标估计，虚线为共有解析真值。b，改变均匀干预的位置；虚线分别为已知通道积分参考。种子为 11、12、13，横坐标 0–2 是其顺序。两面板纵轴独立，均为 nat。图中点是 TM 估计，不是精确 EI。*

积分仅作已知一维通道的参考，主估计沿用仓库 TM。积分先对真实均值 $0.8\sinh x$ 或 $0.8z$ 做输入求积，再计算卷积输出熵；64/128/256 个输入节点结果差约 $10^{-14}$。恒等变换版本还与高斯 CDF 差给出的均匀输入卷积密度核对，MI 差约 $2.2\times10^{-14}$。输出积分采用有限的 8 倍噪声标准差尾部范围和 50001 点网格；数值一致性不应被表述为任意参数下的严格误差界。

#### C.5.2 固定预算的联合执行器筛选

复用原仓库的 8 源双稳态锁存器，20 个连续预设种子 20260615–20260634，每例 28 个二节点组合。所有方法先获得完全相同的每对 $5\times5$ 幅度网格，再按 Syn、joint EI、成功率、最小已观察成功能量或随机顺序选出 3 对，使用相同 $21\times21$ 网格细化。并列名次用共同随机排列打破。

每个方法都保留全部粗网格中已知成功方案作为兜底，不能人为丢弃信息来削弱随机或直接基线。每种方法的独立查询预算均为 $28\times25+3\times(441-25)=1948$；全对细网格参考需要 12348 个独立查询，只用于评估。实现共享模拟结果以节省计算，方法排名不读取其他对的细网格或参考最优值。

固定脉冲时长为 8、幅度范围为 $[0,4]$。能量为 $J=8(r_i u_i^2+r_j u_j^2)$。两种成本条件分别取全部 $r_i=1$，以及每个实例固定抽取的 $r_i\sim\operatorname{LogUniform}[0.25,4]$；动力学、样本、标签和随机并列规则完全不变，仅成本及使用成本的排序改变。

| 筛选方法 | 等成本：平均相对能量超额 | 异质成本：平均相对能量超额 | 异质成本：平均绝对超额 |
| --- | ---: | ---: | ---: |
| Syn | 0.00% | 23.49% | 5.287 |
| joint EI | 2.29% | 20.03% | 4.004 |
| 成功率 | 0.00% | 31.77% | 6.780 |
| 最小已观察成功能量 | 1.53% | 0.00% | 0.000 |
| 随机 | 42.35% | 68.02% | 15.440 |

相对超额逐实例计算 $(J_{\mathrm{method}}-J_{\mathrm{grid}}^\star)/J_{\mathrm{grid}}^\star$ 后取均值；所有方法在两种成本条件下的失败率均为 0。这里“最优”仅指固定时长、两节点、有限幅度网格中的最佳方案，不是连续动作、可变时长或反馈控制的全局最优。

![相同预算下的控制成本比较](../../fig/ei_control_screening_pilot.png)

*图 2。点为 20 个实例的平均绝对能量超额，误差线为实例间标准差，不是置信区间。左右分别为等成本和异质成本，纵轴范围独立。标准差区间可延伸到零以下，但各实例实际超额均非负；所有方法均成功。*

配对结果比均值更能限定结论：等成本下 Syn 与成功率在 20/20 例持平；相对直接成本筛选，Syn 在 2 例更好、18 例持平。异质成本下，直接成本筛选在 10 例优于 Syn、10 例持平、没有更差实例。joint EI 与 Syn 在异质成本下各赢 2 例、16 例持平，所以两者平均差不能解释成稳定优劣。

Syn 采用离散笛卡尔均匀干预的精确频数，单位为 bit；不存在连续密度估计，故此实验不适用 TM。非负容差为 $10^{-10}$ bit，560 个独立组合表的原始最小值为 0、负值计数为 0。程序对显著负值显式失败并报告最小值、阈值和数量。前 3 个实例的所有 28 对均通过步长减半标签核对，每例包含 700 个粗网格点和 12348 个细网格点；全试验最接近 basin 阈值的细网格距离为 $7.53\times10^{-5}$。

**这组证据的解释：** Syn 可以在受控、成本与增益对齐的模型中成为有效筛选分数，但不包含执行器价格。异质成本反例说明不能将其直接解释为普遍的能量代理。该试验用最终 basin 发现数据计算分数，因此不是“完全未见控制结果”的机制预筛选验证，更不是对现有自然网络负结果的修复。发现网格已包含成功样本，也使本试验主要考查细化后的能量，不考查稀有成功的首次发现。

#### C.5.3 复现

在仓库根目录运行：

```bash
.venv/bin/python scripts/koopman_ei_invariance_pilot.py --n 4096 --seeds 3
.venv/bin/python scripts/ei_control_screening_pilot.py
```

两个脚本分别输出精简结果并重建对应 PNG；快速 smoke 模式均为 `--smoke`，会使用更少种子，建议用 `--output /tmp/对应图.png` 避免覆盖正式图。小实验没有创建长期数据缓存或 CSV。控制脚本输出实例配对差值；完整采样和积分参数可从固定代码重建。

独立核对命令为 `.venv/bin/python -m unittest discover -s tests -p test_ei_control_pilots.py -v`，3 项检查全部通过：用潜在推前密度重新积分核对非线性均匀输入参考、显著负 Syn 的完整错误报告、以及配对控制的预算和已知成功方案复用。正文的 15 个公式编号、11 处显式公式引用和本地文件链接均已检查；两张导出图已视觉检查。

### C.6 建议的具体切入点：保留控制任务的宏观信息

#### C.6.1 第一篇工作可以收窄成什么

建议问题表述：**在相同真实动作预算下，哪些 NIS+/Koopman 宏观表示能够压缩状态而保留目标转移通道？在此表示上，PEID 能否减少联合执行器的搜索成本？**

这里 EI 的主线仍然清楚，但需要保留整个动作条件机制，而不只一个分数。对每个候选表示，先检验任务读出是否可由 $\mathbf{z}$ 表示、动作是否在物理上可实施、干预分布是否具有覆盖，再检验受控动态的闭合。

一个直接的诊断是，固定干预设计分布后定义

$$
D_\psi=I(\mathbf{x}_t;\mathbf{z}_{t+1}\mid\mathbf{z}_t,\mathbf{u}_t)
=\mathbb E\!\left[
D_{\mathrm{KL}}\!\left(p(\mathbf{z}'\mid\mathbf{x},\mathbf{u})\Vert
p(\mathbf{z}'\mid\psi(\mathbf{x}),\mathbf{u})\right)\right].
\tag{C.6.1}
$$

右侧是条件 MI 的标准表达，不是新定理。$D_\psi=0$ 表示在所覆盖状态和动作上，同一宏观状态中的微观差异不再影响下一步宏观分布。若同时即时成本、目标和约束也只依赖宏观状态，并且上述相同性对所有相关状态和动作成立，就满足构造精确控制抽象的核心条件，动态规划可在宏观空间进行。这与 MDP abstraction/bisimulation 紧密相关，必须作为直接理论对照。[DBC](https://arxiv.org/abs/2006.10742)

**重要限制：** 平均 $D_\psi$ 小，不保证稀有跨 basin 轨迹上的误差小。要导出策略性能界，还需覆盖、逐状态动作误差或分布转移控制；不能把自由演化下的平均 EI 差直接称为控制性能保证。

#### C.6.2 最小可证伪的下一阶段

先在小型随机受控系统上比较三种同维表示：预测误差选出的表示、原有状态 EI 选出的表示、加入动作条件信息与任务读出约束的表示。固定数据、特征字典或网络容量、训练和规划预算；所有方法使用相同 MPC 求解器。低维已知模型可用枚举/动态规划或可核验的优化解提供评估参考。

主指标是达到相同成功概率所需的能量；次指标是模型查询数、约束违规率、规划时间和跨初态/目标/执行器成本的迁移。不要用 EI 提高本身充当控制成功。最有价值的对照包括：高 EI 但执行器不可达的模式、低频但与目标无关的模式、相同动力学不同操作成本，以及短时预测准确但跨 basin 失败。

随后才加入 PEID：在固定物理动作分块上识别候选组合，作为搜索先验或 warm start，保留探索其他组合的通道。必须比较不加 PEID 的同一控制器、joint EI、直接成本/成功率筛选、随机筛选；线性适用时加入 Gramian 执行器选择，非线性小系统加入直接受限优化。筛选使用的仿真或实验查询必须计入总预算。

若在固定目标上 EI/PEID 不能赢过直接任务读出，可检验它作为跨任务可复用先验的价值，但需把预训练成本计入，并给出摊销到多少任务后才划算。这是待检验假设，不能用来解释掉所有负结果。

#### C.6.3 脑网络、鸟群与强化学习的衔接

| 场景 | 先定义可操作的任务 | EI/PEID 可放的位置 | 最直接对照 |
| --- | --- | --- | --- |
| 模拟脑网络 | 指定脑区群从低活动到目标活动/吸引域，限制刺激位置、电流与持续时间 | 学习目标充分的宏观状态；筛选联合刺激组 | 输出可控性、局部线性 MPC、非线性 MPC/直接优化 |
| 鸟群 | 改变质心速度方向，同时约束凝聚度、碰撞距离与少数受控个体 | 宏观表示选择与领航个体组合 | 相同任务下的领航/钉扎控制、MPC、任务训练 RL |
| 层级 RL | 将可实施动作序列封装为宏观技能，完成指定目标转移 | NIS+ 学状态抽象；干预式信息学习技能效果；PEID 分析组合 | DIAYN、DADS、METRA，以及同容量任务模型 |
| 世界模型控制 | 已有仿真任务上固定数据和规划预算 | 给动作条件模型加入信息保持约束 | Koopman MPC、TD-MPC2；更大预算再用 Dreamer |

这些是对应关系与后续设计，不是本次完成的算法。尤其 DADS 已研究“可预测技能的动力学与规划”，METRA 已指出纯 MI 技能学习的探索局限；若只把 latent skill 的互信息改名为 EI，贡献并不充分。[DADS](https://arxiv.org/abs/1907.01657)、[METRA](https://arxiv.org/abs/2310.08887)

应追求在有限数据、有限计算、模型误差或执行器选择约束下改善效果。对于已知线性系统的无约束最小能量问题，式 (C.4.3) 已给出全局最优值；任何 EI 方法都不可能在相同问题设定下比这个真最优更低。降低宏观任务的约束后得到更低能量，也不能算击败原始全状态控制，因为任务已变。

### C.7 当前决策

建议继续的优先级是：**干预对齐与控制抽象 > 目标相关的执行器组合筛选 > 大规模脑网络或高级 RL。**

值得形成第一阶段成果的是：清楚的坐标/干预区分、明确的 EI—Gramian 联系和反例、一个预算公平且可复现的表示选择验证。标准恒等式和已有 empowerment/Koopman 公式只能作为理论基础；可发表的新贡献仍需来自新的受控表示条件、可验证误差界或稳定的算法增益。

<a id="ei-control-literature"></a>

## 附录 D：EI—Koopman—干预控制的文献证据

检索日期：2026-09-14。配套结论、推导及实验见 [探索报告](exploration.md#ei-koopman-control)。

### D.1 范围与证据边界

问题是：多尺度 EI 和 PEID 能否帮助学习可控制的表示、选择联合执行器，并降低指定宏观状态转移的成本？检索覆盖 EI/NIS+、信息论 Koopman 表示、empowerment、网络控制、控制抽象及技能学习，时间范围为基础工作至 2026-09-14。

首先只读检索本地 Zotero；以 `effective information Koopman causal emergence control empowerment NIS` 排序得到 35 个候选，另精确检索 PEID。选取 7 篇直接相关论文读取本地索引全文，并以公开学术页面核对关键题名、版本或发表信息。外部定向检索补齐控制和 RL 文献。下表保留 34 篇有明确关联的文献；不声称全部精读，也不将检索结果数量当作证据强度。

证据标签：**全文**表示读取了可访问正文中的相关章节，不代表逐页验证所有定理；**摘要**表示只据摘要或学术页面的摘要描述其关联；**仅元数据**不用于推断结果。未逐篇排除后续勘误，本文也不是穷尽式系统综述。

本地论文主键与附件键：

| 文献 | Zotero item key | 全文 attachment key | 本次使用部分 |
| --- | --- | --- | --- |
| PEID | MYATYWAJ | Y8GH3W58 | 二源定义、独立干预、分区 Syn 非负性及层级关系 |
| NIS+ | T3TKPLLG | JKTP95PB | 宏观均匀干预、表示学习目标、方法和鸟群实验 |
| 线性随机 EI | FTUE5FKU | GAGTIZ4M | 干预域、线性粗粒化、EI 表达式 |
| Information Shapes Koopman Representation | ZNEHTSSU | APVZWFXQ | 概率生成模型、信息目标、谱分配与表示塌缩 |
| Koopman 有限维不变子空间 | Y2EFJY5S | JL4329JZ | 不变子空间限制、控制动机 |
| Koopman 控制综述 | 5S7TB4JI | QQK736JS | 受控模型、双线性表示、误差与闭环保证 |
| 生物宏观尺度与控制 | AELDZBBX | WHXVBYZP | EI 的生物尺度选择动机 |

本地排名中其余相关论文只使用摘要，键在下表注明。没有向 Zotero 导入或修改记录。

### D.2 直接相关文献及各自作用

#### EI、多尺度与分解

| 编号 | 文献及入口 | 类别；证据 | 本项目应如何使用 |
| --- | --- | --- | --- |
| 1 | Yang、Wang、Zhang，2026，[Partial Effective Information Decomposition for Synergistic Causality](https://arxiv.org/abs/2605.03267) | Emerging，预印本；全文；MYATYWAJ | 沿用干预和 Syn 定义；它不是控制性能定理 |
| 2 | Yang 等，2025，[Finding emergence in data by maximizing effective information](https://pmc.ncbi.nlm.nih.gov/articles/PMC11697982/) | Recent frontier；全文；T3TKPLLG | NIS+ 表示选择基础；需补显式动作条件机制 |
| 3 | Zhang、Liu，2022 在线发表，[Neural Information Squeezer for Causal Emergence](https://doi.org/10.3390/e25010026) | Foundational；摘要；K8NRVTXW | 可逆转换和信息丢弃的结构；卷期属于 2023 年 |
| 4 | Liu、Yuan、Zhang，2024，[An Exact Theory of Causal Emergence for Linear Stochastic Iteration Systems](https://arxiv.org/abs/2405.09207) | Recent frontier；全文；FTUE5FKU | 最直接解析先例；引用公式前核对均匀输入与噪声卷积 |
| 5 | Liu 等，2025，[SVD-based Causal Emergence for Gaussian Iterative Systems](https://arxiv.org/abs/2502.08261) | Recent frontier；摘要；C7479JCN | 谱、噪声协方差与 EI 的桥梁；采用此预印本题名 |
| 6 | Hoel、Albantakis、Tononi，2013，[Quantifying causal emergence shows that macro can beat micro](https://doi.org/10.1073/pnas.1314922110) | Foundational；摘要；G9D838F7 | 多尺度重新干预的基础，避免误用数据处理不等式 |
| 7 | Hoel、Levin，2020，[Emergence of informative higher scales in biological systems: a computational toolkit for optimal prediction and control](https://doi.org/10.1080/19420889.2020.1802914) | Foundational；全文；AELDZBBX | “EI 支持生物调控”已有明确动机，不能当全新想法 |
| 8 | Rosas 等，2020，[Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data](https://doi.org/10.1371/journal.pcbi.1008289) | Foundational；摘要；7LWNIPLK | 脑活动和鸟群的多尺度信息先例；不等于真实最小能量控制 |
| 9 | Varley、Hoel，2022，[Emergence as the conversion of information: a unifying theory](https://doi.org/10.1098/rsta.2021.0150) | Foundational；摘要；5V5LJEMM | 总信息与分解组成可有不同变化，支持区分 MI 和 Syn |

#### Koopman 表示与控制

| 编号 | 文献及入口 | 类别；证据 | 本项目应如何使用 |
| --- | --- | --- | --- |
| 10 | Cheng 等，ICLR 2026，[Information Shapes Koopman Representation](https://arxiv.org/abs/2510.13025) | Recent frontier；全文；ZNEHTSSU | 最接近的信息论表示学习对照；已有 MI 与谱多样性的联合目标 |
| 11 | Brunton 等，2016，[Koopman Invariant Subspaces and Finite Linear Representations of Nonlinear Dynamical Systems for Control](https://pmc.ncbi.nlm.nih.gov/articles/PMC4769143/) | Foundational；全文；Y2EFJY5S | 区分有限维精确表示、近似表示和可重建原状态的要求 |
| 12 | Proctor、Brunton、Kutz，2016 预印本，[Generalizing Koopman Theory to allow for inputs and control](https://arxiv.org/abs/1602.07647) | Foundational；摘要 | 自主动力学表示不足以识别输入输出作用 |
| 13 | Korda、Mezić，2018，[Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control](https://arxiv.org/abs/1611.03537) | Foundational；全文相关章节 | 最合适的低成本控制基线：相同预测器容量和 MPC 预算 |
| 14 | Li 等，ICLR 2020，[Learning Compositional Koopman Operators for Model-Based Control](https://arxiv.org/abs/1910.08264) | Foundational；摘要 | 图结构/可组合表示已经用于控制，适合网络任务对照 |
| 15 | Strässer 等，2026，[An overview of Koopman-based control: From error bounds to closed-loop guarantees](https://doi.org/10.1016/j.arcontrol.2025.101035) | Recent frontier；全文；5S7TB4JI | 误差与稳定性条件是现代 Koopman 控制的重要部分；不能只比预测误差 |

#### 信息论控制、网络控制和脑网络

| 编号 | 文献及入口 | 类别；证据 | 本项目应如何使用 |
| --- | --- | --- | --- |
| 16 | Klyubin、Polani、Nehaniv，2008，[Keep Your Options Open: An Information-Based Driving Principle for Sensorimotor Systems](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0004018) | Foundational；摘要 | 动作到感知结果的通道容量，是 EI 控制方向的直接邻域 |
| 17 | Salge、Glackin、Polani，2012，[Approximation of Empowerment in the Continuous Domain](https://doi.org/10.1142/S0219525912500798) | Foundational；摘要 | 连续 empowerment 的线性高斯近似，避免重复已有信道计算 |
| 18 | Salge、Glackin、Polani，2013 预印本，[Empowerment—An Introduction](https://arxiv.org/html/1310.1863) | Foundational；全文相关章节 | 任务无关的控制能力与指定目标效用不同；开放/闭环语义需区分 |
| 19 | Summers、Cortesi、Lygeros，2016，[On Submodularity and Controllability in Complex Dynamical Networks](https://arxiv.org/abs/1404.7665) | Foundational；摘要 | 执行器组合的强基线；具体子模结论必须逐指标核对，不能一概而论 |
| 20 | Baggio、Bassett、Pasqualetti，2021，[Data-Driven Control of Complex Networks](https://arxiv.org/abs/2003.12189) | Foundational；摘要 | 已有从实验输入输出直接求控制的路线；定理适用线性系统 |
| 21 | Gu 等，2015，[Controllability of structural brain networks](https://www.nature.com/articles/ncomms9414) | Foundational；摘要及页面方法片段 | 脑网络线性控制的基础参照，拓扑与真实非线性刺激机制需区分 |
| 22 | Betzel 等，2016，[Optimally controlling the human connectome: the role of network topology](https://www.nature.com/articles/srep30770) | Foundational；摘要 | 将起终状态与控制节点纳入问题定义 |
| 23 | Karrer 等，2019 预印本，[A practical guide to methodological considerations in the controllability of structural brain networks](https://arxiv.org/abs/1908.03514) | Foundational；摘要 | 控制时域、归一化、输入矩阵等方法选择；此处不混用正式发表年份 |
| 24 | 2024，[A network control theory pipeline for studying the dynamics of the structural connectome](https://www.nature.com/articles/s41596-024-01023-w) | Recent frontier；摘要 | 可复用控制能量和平均可控性的计算流程 |
| 25 | Ben Messaoud 等，2025，[Low-dimensional controllability of brain networks](https://doi.org/10.1371/journal.pcbi.1012691) | Recent frontier；摘要 | 极直接的宏观控制对照：谱投影与输出可控性；全文访问遇到验证码，未据全文推断细节 |

#### 强化学习与集体调控

| 编号 | 文献及入口 | 类别；证据 | 本项目应如何使用 |
| --- | --- | --- | --- |
| 26 | Zhang 等，ICLR 2021，[Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/abs/2006.10742) | Foundational；摘要 | 用 bisimulation 保留任务相关信息，是控制抽象的直接理论对照 |
| 27 | Eysenbach 等，ICLR 2019，[Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/abs/1802.06070) | Foundational；摘要 | 信息论技能发现；技能可区分性不等于低成本目标转移 |
| 28 | Sharma 等，ICLR 2020，[Dynamics-Aware Unsupervised Discovery of Skills](https://arxiv.org/abs/1907.01657) | Foundational；摘要 | 技能的可预测结果和宏观规划，最贴近分层控制构想 |
| 29 | Park、Rybkin、Levine，ICLR 2024，[METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://arxiv.org/abs/2310.08887) | Recent frontier；摘要 | 时间距离约束的技能表示；纯 MI 目标的局限已有讨论 |
| 30 | Hansen、Su、Wang，ICLR 2024，[TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828) | Recent frontier；摘要 | 适合第二阶段的潜在世界模型与规划基线，非本轮重训练对象 |
| 31 | Hafner 等，2025，[Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2) | Recent frontier；摘要 | Dreamer 的跨任务世界模型路线；规模较大，应在小实验成立后比较 |
| 32 | 2019 预印本，[Learning to flock through reinforcement](https://arxiv.org/abs/1911.01697) | Foundational；摘要 | 已有学习集体运动先例；“形成鸟群”与“低成本宏观转向”应分开 |
| 33 | 2021，[Learning to control active matter](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.033291) | Foundational；摘要 | 用 RL 调控集体现象，适合作为复杂系统调控场景参照 |
| 34 | Chen 等，2023，[Optimal synchronization in pulse-coupled oscillator networks using reinforcement learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10109446/) | Recent frontier；摘要 | 振子网络的任务驱动调控，适合脑/同步任务的邻近基线 |

### D.3 综合判断与阅读顺序

**已有共识与先例。** 信息理论指导表示学习、动作到结果的容量、Koopman 预测器用于控制、低维脑网络输出控制，都已有直接文献。新工作不能仅将这几个词放在同一框架里。

**需认真区分的假设。** NIS+ 比较的宏观最大熵干预、Koopman 表示学习中的观测/生成分布、empowerment 优化的动作分布、最小能量控制的目标约束，彼此不相同。“预测更好”“EI 更高”“控制更便宜”因而不是可互换结论。

**值得探索但尚未证实的缺口。** 将物理可实施干预、受控动态闭合和成本保持同时纳入多尺度 EI 表示选择，再检验 PEID 是否能在相同预算下缩小执行器组合搜索。当前检索没有确立这一具体方案的首创性，更没有证明它优于现有算法。

建议阅读顺序：

1. 先对读 **NIS+、线性随机 EI、PEID**：明确自己的干预和分解对象。
2. 再读 **Information Shapes Koopman Representation、Koopman MPC、2026 Koopman 控制综述**：明确已经解决的表示问题，以及未解决的受控误差问题。
3. 接着读 **empowerment 综述、低维脑网络控制、执行器 Gramian 选择**：区分整体控制能力与指定目标成本。
4. 最后看 **DBC、DADS、METRA**，选择控制抽象或技能学习的连接点；TD-MPC2/Dreamer 留作后续预算允许时的算法对照。

### D.4 检索可复核性与访问限制

公共学术 API 的四个原始查询为：

- `Koopman effective information causal emergence control`，起始年份 2013。
- `empowerment controllability information energy`，起始年份 2005。
- `causal emergence neural information squeezer control`，起始年份 2013。
- `Koopman model predictive control reinforcement learning abstraction`，起始年份 2023。

每查询每源上限 12、超时 8 秒。返回的跨领域噪声较多，部分元数据年份甚至超过当前日期，因此没有直接使用自动排名的“最新”结果。后续采用题名和主题定向检索，并优先使用 arXiv、出版社、作者机构页面以及本地全文。

arXiv export API 与部分 Semantic Scholar 查询返回 HTTP 429 或超时；遇到这些结果后未重试同一访问路径。Baggio 论文的实验 HTML 入口失败，但摘要页可读；低维脑网络论文 PMC 正文访问触发验证码，降级为摘要证据。另一个非核心候选 Cornelius 等的 Nature 入口跳转认证失败，未将其作为已核验文献纳入上表。没有下载远程 PDF、绕过登录或导入 Zotero。

原始 API 记录暂存于 `/tmp/eisyn_control_search_1.json` 至 `/tmp/eisyn_control_search_4.json`，仅为本次检索审计；下方保存持久化的查询与失败链接，避免依赖临时文件重建检索。

#### 公共 API 查询及失败记录

**查询 1**：Koopman effective information causal emergence control

- [openalex](https://api.openalex.org/works?search=Koopman+effective+information+causal+emergence+control&filter=has_abstract%3Atrue%2Cfrom_publication_date%3A2013-01-01&sort=publication_date%3Adesc&per-page=12)
- [semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=Koopman+effective+information+causal+emergence+control&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2013-)
- [arxiv](https://export.arxiv.org/api/query?search_query=all%3AKoopman+effective+information+causal+emergence+control&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending)
- [crossref](https://api.crossref.org/works?query=Koopman+effective+information+causal+emergence+control&rows=12&sort=published&order=desc&select=DOI%2Ctitle%2Cauthor%2Cpublished%2Cpublished-online%2Cpublished-print%2Ccontainer-title%2CURL%2Cis-referenced-by-count&filter=from-pub-date%3A2013-01-01)
- 失败：[arxiv](https://export.arxiv.org/api/query?search_query=all%3AKoopman+effective+information+causal+emergence+control&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending) — permission gate: HTTP 429

**查询 2**：empowerment controllability information energy

- [openalex](https://api.openalex.org/works?search=empowerment+controllability+information+energy&filter=has_abstract%3Atrue%2Cfrom_publication_date%3A2005-01-01&sort=publication_date%3Adesc&per-page=12)
- [semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=empowerment+controllability+information+energy&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2005-)
- [arxiv](https://export.arxiv.org/api/query?search_query=all%3Aempowerment+controllability+information+energy&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending)
- [crossref](https://api.crossref.org/works?query=empowerment+controllability+information+energy&rows=12&sort=published&order=desc&select=DOI%2Ctitle%2Cauthor%2Cpublished%2Cpublished-online%2Cpublished-print%2Ccontainer-title%2CURL%2Cis-referenced-by-count&filter=from-pub-date%3A2005-01-01)
- 失败：[semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=empowerment+controllability+information+energy&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2005-) — permission gate: HTTP 429
- 失败：[arxiv](https://export.arxiv.org/api/query?search_query=all%3Aempowerment+controllability+information+energy&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending) — permission gate: HTTP 429

**查询 3**：causal emergence neural information squeezer control

- [openalex](https://api.openalex.org/works?search=causal+emergence+neural+information+squeezer+control&filter=has_abstract%3Atrue%2Cfrom_publication_date%3A2013-01-01&sort=publication_date%3Adesc&per-page=12)
- [semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=causal+emergence+neural+information+squeezer+control&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2013-)
- [arxiv](https://export.arxiv.org/api/query?search_query=all%3Acausal+emergence+neural+information+squeezer+control&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending)
- [crossref](https://api.crossref.org/works?query=causal+emergence+neural+information+squeezer+control&rows=12&sort=published&order=desc&select=DOI%2Ctitle%2Cauthor%2Cpublished%2Cpublished-online%2Cpublished-print%2Ccontainer-title%2CURL%2Cis-referenced-by-count&filter=from-pub-date%3A2013-01-01)
- 失败：[semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=causal+emergence+neural+information+squeezer+control&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2013-) — permission gate: HTTP 429
- 失败：[arxiv](https://export.arxiv.org/api/query?search_query=all%3Acausal+emergence+neural+information+squeezer+control&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending) — network failure: The read operation timed out

**查询 4**：Koopman model predictive control reinforcement learning abstraction

- [openalex](https://api.openalex.org/works?search=Koopman+model+predictive+control+reinforcement+learning+abstraction&filter=has_abstract%3Atrue%2Cfrom_publication_date%3A2023-01-01&sort=publication_date%3Adesc&per-page=12)
- [semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=Koopman+model+predictive+control+reinforcement+learning+abstraction&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2023-)
- [arxiv](https://export.arxiv.org/api/query?search_query=all%3AKoopman+model+predictive+control+reinforcement+learning+abstraction&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending)
- [crossref](https://api.crossref.org/works?query=Koopman+model+predictive+control+reinforcement+learning+abstraction&rows=12&sort=published&order=desc&select=DOI%2Ctitle%2Cauthor%2Cpublished%2Cpublished-online%2Cpublished-print%2Ccontainer-title%2CURL%2Cis-referenced-by-count&filter=from-pub-date%3A2023-01-01)
- 失败：[semantic-scholar](https://api.semanticscholar.org/graph/v1/paper/search?query=Koopman+model+predictive+control+reinforcement+learning+abstraction&limit=12&fields=title%2Cauthors%2Cyear%2CpublicationDate%2Cvenue%2CexternalIds%2Curl%2CcitationCount&year=2023-) — permission gate: HTTP 429
- 失败：[arxiv](https://export.arxiv.org/api/query?search_query=all%3AKoopman+model+predictive+control+reinforcement+learning+abstraction&start=0&max_results=12&sortBy=submittedDate&sortOrder=descending) — network failure: The read operation timed out

<a id="part1-fairness-check"></a>

## 附录 E：Part1 早期公平性检查约定

历史检查清单，保留用于追溯原实验约定；实际执行与现行图表以 [Part1](Part1.md) 为准。

本文档用于检查 `Part1.md` 文件里的所有实验。

检查以下几点确保实验的公平性：

1. 同一 panel 内，横轴参数在零点时，从生成数据到计算流程都和其他参数点一致。
2. 同一 panel 内，MLP 训练、WMS、SURD 以及其他直接读取观测数据的方法必须使用同一套数据口径。默认优先使用同一条自然轨迹及其对齐的 source-target 样本，不得为某个方法单独更换状态分布。MLP+SHAP 和 MLP+PEID 必须复用同一个 fitted MLP。
3. PEID 在 MLP 训练完成后，可以独立均匀采样 intervention states，再将这些样本输入 fitted MLP 计算 PEID。这属于 PEID 方法内部的干预读出流程，不要求 intervention states 与 WMS、SURD 等观测方法使用的自然轨迹样本相同，也不应把 `readout_state_digest == peid_readout_state_digest` 作为普遍公平性条件。
4. 计算信息指标时，同一 panel 内要么都使用离散分箱方式估计，要么都使用 TM 方法估计。如果使用 TM，在估计概率密度和互信息时，映射函数的多项式阶数必须一致。

#### 公平性审计

公平性审计必须按以下口径执行。旧版 `fairness_audit.passed` 若仍要求 PEID intervention states 与观测方法共享 readout states，则不能作为本口径下的通过证据，需要先更新审计条件再重新检查。

1. **零点流程一致**：同一 panel 和 seed 下，各横轴参数点使用参数匹配的 train/readout 输入状态池，即输入状态 digest 在参数扫描中保持一致；参数只进入真实映射和由此产生的目标。零点与正参数点调用相同的训练、SHAP、WMS、SURD 和 PEID 流程，图中零点直接报告估计 residual，且 `raw_*` 与展示值相同。
2. **观测数据与模型一致**：对每个参数和 seed，MLP 训练、WMS、SURD 及其他观测方法使用同一套自然轨迹数据口径和相同的 source-target 对齐方式。默认使用完全相同的自然轨迹样本；若预先注册 train/readout split，则所有观测方法必须遵守同一 split 规则，且不得为不同方法改变数据分布。MLP+SHAP 与 MLP+PEID 复用同一个 fitted MLP；`shap_mlp_model_digest == peid_mlp_model_digest`。
3. **PEID 干预采样独立**：MLP 拟合完成后，MLP+PEID 可以从注册干预域独立均匀采样 intervention states，并在该 fitted MLP 上读出 PEID。审计应检查干预域、采样方式、样本量和随机种子是否记录完整，但不要求 PEID intervention-state digest 与 WMS/SURD 的自然轨迹 digest 相同。Oracle PEID 若用于机制参照，应与 MLP+PEID 使用相同的干预协议。
4. **信息估计器一致**：同一 panel 的 WMS、SURD、MLP+PEID 与 Oracle PEID 使用相同类别的信息估计器；使用 TM 时统一多项式阶数，使用分箱时统一分箱规则。SHAP interaction、PCMCI 和 Neural Granger 等原生读数不是互信息量，不受此条约束，但必须明确标注其单位和语义。
5. **持续检查**：自动审计应检查零点流程、观测数据口径、source-target 对齐、共享 fitted MLP、PEID 干预协议、估计器一致性和零点原始值。不得仅因 PEID 使用独立均匀 intervention states 而将公平性判为失败。


统一协议如下：

- 同一 panel 中，MLP、WMS、SURD 和其他观测方法默认使用同一条自然轨迹及相同的 source-target 对齐样本。自然轨迹是观测方法公平比较的首选数据口径。
- 如果某个 panel 因实验设计必须使用 broad one-step samples，而不是自然轨迹，则 MLP、WMS、SURD 和其他观测方法必须共同使用这套 broad one-step 数据口径，并在报告中明确说明原因。不得让 MLP 使用 broad states、WMS/SURD 使用自然轨迹，或反向混用。
- MLP+PEID 的数据流程分为两个阶段：先使用与 WMS/SURD 同口径的数据训练 MLP；再从注册干预域独立均匀采样 intervention states，将其输入同一个 fitted MLP 计算 PEID。第二阶段样本属于 PEID 方法内部过程，不参与观测数据一致性比较。
- 在同一系统、参数和 seed 下，SHAP 与 MLP+PEID 使用同一个 fitted MLP；JSON 中记录 MLP digest，用来审计二者是否确实共享模型。JSON 还应分别记录观测数据 digest 和 PEID intervention-state digest，但两者无需相等。
- Oracle PEID 仅作为事后机制一致性诊断，并与 MLP+PEID 复用相同的干预域、采样 states 和目标噪声；Oracle PEID 不参与 MLP 训练数据与观测方法数据是否一致的判断。
- Standard Map、Wilson-Cowan refractory、Kuramoto、Ikeda y_tau 和 Nicholson-Bailey 的正式信息量数值使用三阶 transport map。Coupled Hénon 的 WMS、SURD、MLP+PEID 与 Oracle PEID 统一使用每变量 `6` 个等宽 bins；SHAP interaction 仍读取连续 MLP 响应。估计器一致性不意味着样本状态必须相同：PEID 仍可使用独立均匀 intervention states。
- Syn 按定义非负。每个数值实验必须用 Syn 原生单位声明零容差：容差内的小负值记为数值零，并在 JSON 中记录容差、数量和最小原始值；小于负容差的值必须显式报错。不得使用未声明的 `max(0, Syn)`。
- panel a 使用 `symlog` 纵轴，并在图内明确标注；这是为了同时保留 Standard Map 上 SURD 的极端退化估计和约 `0.03-0.18` bits 的 PEID 趋势，不改变任何原始数值。
- 对由扫描参数显式关闭的结构交互，主图仍显示同一套生成数据和同一 fitted MLP 经配置 estimator 得到的零点 residual；`raw_*` 字段保留为同值审计列。若 MLP residual 明显大于同 estimator 的 Oracle 零点 residual，则说明 surrogate 在 broad readout 上仍有形状误差，而不是说明真实机制存在协同。
