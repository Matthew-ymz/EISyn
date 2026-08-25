# 固定干预支持下的 Sine 振幅—空间频率校准实验

本实验检验：当只改变振幅参数 $\alpha$ 或响应面的空间频率 $k$ 时，MLP+PEID 的 ${x,y}\rightarrow z$ 协同读数如何变化。这里的 $k$ 是 $x_ty_t$ 响应面上的空间振荡频率，不是时间采样频率。

$$
\begin{aligned}
x_{t+1} &= 0.42x_t + \eta^x_t,\\
y_{t+1} &= 0.38y_t + \eta^y_t,\\
z_{t+1} &= 0.22z_t + \alpha\sin(kx_ty_t) + \eta^z_t.
\end{aligned}
$$

## 受控比较协议

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

## 结果判断

1. **Known-dynamics 基准能够识别空间频率。** 在不读取真实 $k$ 的 cross-fitted 选频协议下，真实谐波恢复率为 `100.0%`。
2. **当前 learned MLP 没有复现高频识别。** 总体真实谐波恢复率为 `16.7%`，成功条件主要集中在 $k=1$。固定支持 $R^2$ 在 $k=1$ 时为 `0.884`，而 $k>1$ 时各条件均值仅为 `0.157`–`0.256`。
3. **振幅不变性没有复现。** Known-dynamics Syn 随 $\alpha$ 稳定增加，其各 $k$ 条件的斜率约为 `0.975`–`0.997` bits / unit $\alpha$。旧结果中近似水平的振幅曲线主要来自低阶 TM 特征饱和，不能解释为 PEID 对物理振幅严格不敏感。

## 汇总结果

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

## 解释边界

“Known dynamics” 是在已知条件均值函数上运行相同 TM 估计器所得的机制基准，不是解析真值。Learned 与 known-dynamics 曲线的差异同时反映有限轨迹学习误差与有限样本 TM 误差。只有在固定支持 $R^2$ 保持良好时，才可把 Syn 随 $k$ 的变化主要解释为对响应面几何的敏感性；若二者同时下降，则应解释为 surrogate 分辨率边界。
