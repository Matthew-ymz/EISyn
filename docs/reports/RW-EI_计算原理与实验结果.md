# Reweight EI（RW-EI）：计算原理与实验结果

## 1. 核心结论

Reweight EI（简称 **RW-EI**）提供了一条不拟合动力学模型的 EI 计算路径：它直接调整观测样本的权重，把原本相关的输入分布转换为目标干预分布，再从重加权后的联合分布中计算有效信息。

本实验得到四个直观结论：

1. **小样本时，优先考虑 RW-EI。** RW-EI 不需要先训练动力学模型，因此能够避免小样本条件下明显的模型拟合误差。
2. **样本充足时，两种方法都稳定在真值附近，但 RW-EI 更快。** 在最大样本规模下，RW-EI 的计算时间约为 MLP EI 的 $1/12.2$。
3. **输入相关性过强时，优先考虑 MLP EI。** 强相关会削弱观测分布与独立干预分布之间的重叠，使少量样本获得极大权重，进而造成 RW-EI 的明显偏差。
4. **ESS 是选择 RW-EI 的实用诊断指标。** ESS 越高，权重越均匀，RW-EI 通常越稳定；ESS 过低则意味着结果高度依赖少数样本。

一句话概括：**有重叠、看 ESS、优先 RW-EI；强相关、低 ESS、改用 MLP EI。**

## 2. 实验问题与数据生成过程

实验比较两种 EI 计算路径：

- **MLP EI**：先从观测数据拟合随机动力学，再在目标输入分布下生成干预样本，最后计算互信息；
- **RW-EI**：不拟合动力学，直接对观测样本进行重加权，再计算互信息。

此外，实验还设置了两个使用已知动力学的参照：**Oracle samples + TM** 和 **EI truth**。它们都知道真实的 $f$ 与噪声分布，但计算方式不同：前者先从真实动力学随机采样，再用 TM 估计互信息；后者直接对真实概率分布做确定性数值积分，不经过 TM。

令输入随机向量为 $\mathbf{X}=(X_1,X_2)^\mathsf{T}$，输出为标量 $Y$。观测输入的两个边缘分布均为 $\operatorname{Unif}[-1,1]$，但通过 Gaussian copula 引入相关系数 $\rho$。非线性随机动力学为

$$
Y=f(\mathbf{X})+\varepsilon,
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
=I_q(\mathbf{X};Y)
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

这个表达式说明：EI 衡量的是在指定输入干预下，知道当前输入 $\mathbf{X}$ 能够减少多少关于下一时刻输出 $Y$ 的不确定性。

### 2.1 EI truth：直接积分已知动力学

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

因为 $Y=f(\mathbf{X})+\varepsilon$ 且噪声与输入独立，条件熵就是高斯噪声熵：

$$
H_q(Y\mid\mathbf{X})
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

### 2.2 Oracle samples + TM：真实动力学生成样本，TM 读取互信息

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
\widehat q^{\mathrm{TM}}_{\mathbf{X}Y}(\mathbf{z})
=\phi\!\left(\mathbf{T}_{\mathbf{X}Y}(\mathbf{z})\right)
\left|
\det\nabla\mathbf{T}_{\mathbf{X}Y}(\mathbf{z})
\right|,
$$

其中 $\phi$ 是标准多元高斯密度。类似地得到 $\widehat q^{\mathrm{TM}}_{\mathbf{X}}(\mathbf{x})$ 和 $\widehat q^{\mathrm{TM}}_Y(y)$。Oracle TM 的 EI 估计为

$$
\widehat{\operatorname{EI}}_{\mathrm{Oracle\text{-}TM}}
=\frac{1}{M}\sum_{m=1}^{M}
\log_2
\frac{
\widehat q^{\mathrm{TM}}_{\mathbf{X}Y}
\!\left(\mathbf{x}^{(m)},y^{(m)}\right)
}{
\widehat q^{\mathrm{TM}}_{\mathbf{X}}
\!\left(\mathbf{x}^{(m)}\right)
\widehat q^{\mathrm{TM}}_Y\!\left(y^{(m)}\right)
}.
$$

Oracle TM 不需要观测数据、不需要 MLP，也不需要密度比重加权。它的作用是单独测量“有限干预样本 + TM 读取器”本身会带来多大误差。其误差包含 Monte Carlo 采样误差、有限样本密度估计误差和五阶 TM 的近似误差。

### 2.3 二者的核心区别

| 比较项 | Oracle samples + TM | EI truth |
|---|---|---|
| 是否使用已知 $f$ 和 $\sigma$ | 是 | 是 |
| 如何使用已知动力学 | 随机生成有限干预样本 | 直接构造 $p(y\mid\mathbf{x})$ 并积分 |
| 是否使用 TM | 是，五阶 TM | 否 |
| 是否随随机种子变化 | 是 | 否 |
| 主要误差来源 | Monte Carlo、有限样本、TM 近似 | 求积节点、输出网格和积分截断 |
| 在实验中的作用 | 测量共同 TM 读取器的误差基准 | 提供所有 EI 方法的数值参照 |

因此，二者虽然都使用已知动力学，但不能视为同一个量的重复画法。EI truth 回答“真实干预互信息是多少”；Oracle samples + TM 回答“即使动力学完全已知，只给 TM 有限干预样本时，最终能估计得多准”。二者之间的差距主要反映 TM 读取器与有限样本造成的误差，而不是动力学学习误差或重加权误差。

## 3. RW-EI 为什么可以绕过动力学拟合

### 3.1 从分布替换到重要性权重

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

### 3.2 如何使用权重

对任意可积函数 $g(\mathbf{X},Y)$，目标干预分布下的期望可以改写为观测分布下的加权期望：

$$
\begin{aligned}
\mathbb{E}_{q}[g(\mathbf{X},Y)]
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
w(\mathbf{X})g(\mathbf{X},Y)
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
\mathbb{E}_{q}[g(\mathbf{X},Y)]
\approx
\sum_{i=1}^{N}
\bar{w}_i g(\mathbf{x}_i,y_i).
$$

直观上，若某个输入状态在独立干预下应该更常出现、但在相关观测数据中较少出现，它就会获得更大的权重；反之则获得更小的权重。重加权后，原始观测样本整体上近似服从目标干预分布。

### 3.3 本实验如何估计密度比

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

### 3.4 从重加权样本计算 EI

实验使用同一个五阶多项式三角传输映射（transport map，TM）作为所有方法的互信息读取器。令 $\mathbf{z}$ 表示联合变量，三角映射 $\mathbf{T}$ 将目标密度映射到标准高斯参考密度 $\phi$，则

$$
\widehat{q}(\mathbf{z})
=
\phi\!\left(\mathbf{T}(\mathbf{z})\right)
\left|
\det\nabla\mathbf{T}(\mathbf{z})
\right|.
$$

RW-EI 使用归一化权重 $\bar{w}_i$ 分别拟合加权联合密度 $\widehat{q}_{\mathbf{X}Y}$、输入边缘密度 $\widehat{q}_{\mathbf{X}}$ 和输出边缘密度 $\widehat{q}_Y$。最终估计量为

$$
\widehat{\operatorname{EI}}_{\mathrm{RW}}
=
\sum_{i=1}^{N}
\bar{w}_i
\log_2
\frac{
\widehat{q}_{\mathbf{X}Y}(\mathbf{x}_i,y_i)
}{
\widehat{q}_{\mathbf{X}}(\mathbf{x}_i)
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

## 4. ESS：RW-EI 是否可靠的快速诊断

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

## 5. 合并结果图：准确性、稳健性与效率

下图把准确性、相关性、ESS 与运行时间放在同一个六面板视图中。图 a 与图 e 共用样本量横轴，图 b 与图 c 共用相关系数横轴。准确性实验在固定 $\rho=0.5$ 时扫描 7 个样本量（$N=1{,}000$ 至 $64{,}000$）；稳健性实验在固定 $N=8{,}000$ 时扫描 10 个相关系数（$\rho=0$ 至 $0.9$）。每个条件使用 30 个配对随机种子。运行时间实验对同一组 7 个样本量使用 10 个配对随机种子，并报告中位数及四分位区间。所有方法采用相同动力学、干预支持集、TM 阶数和信息单位，因而差异主要来自“如何得到干预联合分布”。

黑色虚线仅表示 EI 真值，用于检验两种 EI 估计是否接近已知动力学。

灰色的 **Oracle samples + TM** 曲线和黑色 EI truth 虚线都使用已知动力学，但含义不同：灰色曲线是“真实动力学采样后再由 TM 估计”的有限样本结果，黑色虚线是“不经过 TM、直接数值积分”的参照值。两者的计算公式与误差来源见第 2.1--2.3 节。

![RW-EI 准确性、稳健性与运行效率的合并结果](../../exp/reweighted_ei/results/rw_ei_combined_results.svg)

### 5.1 准确性、相关性与 ESS

**图 a：小样本时 RW-EI 更有优势。** 当观测样本数为 $N=1{,}000$ 时，MLP EI 的平均绝对误差（MAE）为 0.0592 bit，而 RW-EI 为 0.0353 bit。MLP 需要同时学习非线性条件均值和噪声尺度，小样本下更容易产生模型拟合误差；RW-EI 跳过这一步，因此表现更稳定。随着样本量增至 64,000，两者的 MAE 分别稳定到 0.0196 bit 和 0.0152 bit，均位于已知动力学真值附近。

**图 b：强相关是 RW-EI 的主要失效条件。** 在 $\rho\leq0.7$ 的大部分区间内，RW-EI 与 MLP EI 的误差处于相同量级，RW-EI 在 $\rho=0.4$ 至 $0.7$ 还略低于 MLP EI。但在 $\rho=0.8$ 时，RW-EI 的 MAE 突然上升到 0.0766 bit，明显高于 MLP EI 的 0.0357 bit；到 $\rho=0.9$ 时，两者分别增至 0.2428 bit 和 0.1402 bit。原因不是动力学发生变化，而是高度相关的观测输入难以覆盖独立干预所需要的状态组合。MLP EI 对这一问题更耐受，但在极端相关下也不是完全免疫。

**图 c：相关性较弱时，两种方法都能跟随 EI 真值。** 图 b 与图 c 共用相关系数 $\rho$ 横轴，便于同时观察“误差大小”和“估计值偏向哪里”。紫色曲线是普通互信息的 TM 估计；蓝色和红色曲线分别是 MLP EI 与 RW-EI。在 $\rho\leq0.7$ 时，两条 EI 估计曲线总体位于 EI 真值附近；从 $\rho=0.8$ 开始，RW-EI 明显向下偏离，而 MLP EI 到 $\rho=0.9$ 才出现更强的下偏。这说明 RW-EI 的误差主要由输入分布转换难度控制。

**图 d：ESS 能够识别 RW-EI 的风险。** 随着 $\operatorname{ESS}/N$ 降低，RW-EI 的误差幅度总体增大。在 $\rho=0.8$ 和 $0.9$ 时，平均 $\operatorname{ESS}/N$ 分别下降到约 0.284 和 0.236，并与明显的负偏差同时出现。因此，ESS 可以在不知道动力学真值的真实应用中，帮助判断当前 RW-EI 是否值得信任。

### 5.2 运行时间与计算开销

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

## 6. 实际使用建议

可以按照下面的简单规则选择方法：

| 数据条件 | 推荐方法 | 原因 |
|---|---|---|
| 样本较少，ESS 较高 | **RW-EI** | 避免小样本下的动力学模型拟合误差 |
| 样本较多，ESS 较高 | **RW-EI** | 精度接近真值，同时计算速度明显更快 |
| 输入相关性很强，ESS 很低 | **MLP EI** | RW-EI 容易受到极端权重和支持不足影响 |
| 无法确定哪种方法可靠 | **先算 RW-EI 和 ESS，再决定是否补充 MLP EI** | ESS 可作为低成本风险诊断 |

需要强调的是，输入变量高度相关并不自动等同于存在未观测混杂。本实验中，强相关造成的核心问题是观测分布与独立干预分布之间缺少足够重叠。若真实系统还存在同时影响输入和输出的未观测因素，则 RW-EI 和 MLP EI 都需要额外的因果识别假设。

## 7. 总结

RW-EI 的核心价值在于：**把“学习动力学”转化为“修正样本分布”**。只要观测数据能够充分覆盖目标干预区域，它就能用一组输入密度比权重直接恢复干预联合分布，并以更低的计算开销获得接近动力学真值的 EI。

本实验给出的选择逻辑非常清楚：

- **小样本：RW-EI 更准；**
- **大样本：两者都稳定在真值附近，但 RW-EI 更快；**
- **强相关：RW-EI 更早失效，优先 MLP EI；极端相关下两者都要谨慎；**
- **是否适合 RW-EI：先看 ESS。**
