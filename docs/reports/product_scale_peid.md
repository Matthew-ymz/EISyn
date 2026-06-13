# 纯乘积映射中 PEID Syn 随 \(\alpha\) 的变化

## 问题

考虑确定性纯乘积映射

$$
z=\alpha xy.
$$

本实验检验：当乘积系数 \(\alpha\) 改变时，独立最大熵干预下 \(x,y\) 对 \(z\) 的 PEID Syn 如何变化。

PEID 使用二源 finest partition residual：

$$
\operatorname{Syn}^{\mathrm{EID}}(x,y\to z)
=I(\{x,y\};z)-I(x;z)-I(y;z).
$$

## 实验协议

- \(x,y\overset{\mathrm{iid}}{\sim}\operatorname{Uniform}[-1,1]\)。
- 扫描
  \(\alpha\in\{-5,-1,-0.1,-0.01,-0.001,0,0.001,0.01,0.1,1,5\}\)。
- 每个 seed 使用同一批 \(x,y\) 干预样本，只改变目标缩放系数。
- `12` 个独立 seed，每个 seed 使用 `50,000` 个样本。
- 使用 `12` 个分位数箱的 histogram PEID；图中为跨 seed 的 mean \(\pm\) std。

![纯乘积映射的 alpha 扫描 PEID](../../fig/product_scale_peid/product_scale_peid.png)

## 结果

| \(\alpha\) 类别 | \(I(x;z)\) | \(I(y;z)\) | \(I(\{x,y\};z)\) | PEID Syn |
| --- | ---: | ---: | ---: | ---: |
| \(\alpha=0\) | 0 | 0 | 0 | 0 |
| 任意扫描到的 \(\alpha\neq0\) | \(0.5155\pm0.0040\) | \(0.5171\pm0.0049\) | \(2.7994\pm0.0092\) | \(1.7667\pm0.0050\) |

结果表现为严格的“零点跳变 + 非零平台”：

1. 当 \(\alpha=0\) 时，\(z\) 为常数，所有 EI 与 Syn 都为零。
2. 当 \(\alpha\neq0\) 时，所有扫描系数得到相同的 PEID 分解；增大 \(|\alpha|\) 不会继续增大 Syn。
3. 改变 \(\alpha\) 的符号同样不改变 Syn。

## 原因

对任意 \(\alpha\neq0\)，映射

$$
xy\mapsto \alpha xy
$$

是目标变量上的可逆线性变换。互信息对可逆目标变换不变，因此

$$
I(S;\alpha xy)=I(S;xy),\qquad \alpha\neq0,
$$

其中 \(S\) 可以是 \(x\)、\(y\) 或联合源 \(\{x,y\}\)。三个 EI 项均不随非零 \(\alpha\) 改变，它们的差 Syn 也保持不变。

当前分位数 histogram 估计器也保留这一性质：正缩放不改变目标分位数标签，负缩放只反转标签顺序，不改变互信息。因此同一 seed 内的非零结果逐点一致。

## 结论与边界

对于纯映射 \(z=\alpha xy\)，PEID Syn 检测的是不可约的联合乘积几何是否存在，而不是系数 \(|\alpha|\) 有多大。它能区分 \(\alpha=0\) 与 \(\alpha\neq0\)，但不能作为非零乘积系数的幅值计。

表中的绝对 bit 数值依赖干预支持、样本量和分箱数。若目标中加入未随 \(\alpha\) 同步缩放的噪声、记忆项或其他竞争项，改变 \(\alpha\) 会改变相对信噪比，此时 Syn 可以随 \(\alpha\) 变化并最终饱和。

## 复现

```bash
python scripts/product_scale_peid.py
```

机器可读结果保存于 `results/product_scale_peid/product_scale_peid.json`。

## Transport-map 弱噪声对照

进一步使用仓库中的连续 transport-map PEID 估计器，并在目标动力学中加入固定尺度的弱高斯噪声：

$$
z=\alpha xy+\epsilon,\qquad \epsilon\sim\mathcal{N}(0,0.05^2).
$$

源分布、扫描点、seed 数和每个 seed 的样本数与 histogram 实验相同。每个 seed 内所有 \(\alpha\) 共享同一批 \(x,y,\epsilon\) 样本，以减少扫描点之间的 Monte Carlo 差异。

![Transport-map 弱噪声乘积映射 alpha 扫描](../../fig/product_scale_peid/product_scale_transport_peid.png)

| \(|\alpha|\) | Transport-map Syn |
| ---: | ---: |
| 0 | \(0.000050\pm0.000035\) |
| 0.001 | 约 \(0.00008\) |
| 0.01 | 约 \(0.00325\) |
| 0.1 | 约 \(0.205\) |
| 1 | 约 \(0.575\) |
| 5 | 约 \(0.586\) |

transport-map 结果与无噪 histogram 结果**定性一致，但不数值一致**：

1. \(\alpha=0\) 时，Syn 仅保留约 \(5\times10^{-5}\) bits 的有限样本估计残差。
2. 正负同幅系数结果近似对称，说明协同由 \(|\alpha|\) 控制，而不由符号控制。
3. 当 \(|\alpha|\) 较大时，Syn 趋于约 `0.586` bits 的平台，仍然不能作为高信噪比区间中的线性幅值计。
4. 与无噪纯映射不同，低 \(|\alpha|\) 区域不再立即进入平台。固定噪声没有随 \(\alpha\) 同步缩放，因此 \(|\alpha|\) 增大会提高乘积信号相对噪声的强度，Syn 随之上升。

单源 EI 在全部扫描点都接近零，高 \(|\alpha|\) 时 joint EI 几乎全部表现为 Syn。这与对称独立干预下纯乘积机制的结构相符：单独观察 \(x\) 或 \(y\) 几乎不能确定带符号的乘积目标，联合观察二者才提供主要信息。

transport-map 与 histogram 的绝对 bit 数值不能直接比较。前者是连续密度模型估计，后者是有限分位数离散估计；二者支持的共同结论是：

> PEID Syn 能识别乘积协同是否在噪声之上可分辨；固定噪声存在时，它主要随信噪比上升，并在高信噪比区间饱和。

复现 transport-map 对照：

```bash
python scripts/product_scale_peid.py --transport-map
```

机器可读结果保存于 `results/product_scale_peid/product_scale_transport_peid.json`。
