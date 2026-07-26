# 混合阶 Kuramoto：从随机转移数据恢复 Greedy 层级

## 实验问题

该正对照检验：从有限的随机动力学转移中学习模型后，使用完整五振子未来响应作为 target，Greedy 层级能否在过程噪声和弱跨模块连接下恢复已知的 `2+3` 分区。

## 已知动力学

五个相位振子包含对称 pairwise 模块

$$
C_2=\{\theta_1,\theta_2\},
$$

以及对称三体模块

$$
C_3=\{\theta_3,\theta_4,\theta_5\}.
$$

模块内部动力学为

$$
\begin{aligned}
\dot{\theta}_1
&=\omega_1+K_1\sin(\theta_2-\theta_1),\\
\dot{\theta}_2
&=\omega_2+K_1\sin(\theta_1-\theta_2),\\
\dot{\theta}_i
&=\omega_i+K_2\sin(\theta_j+\theta_k-2\theta_i),
\qquad
i,j,k\in\{3,4,5\},\ i\ne j\ne k,
\end{aligned}
$$

其中 $K_1=1.8$，$K_2=2.0$。三体项在全局相位平移下不变，并同时依赖三个振子。两个模块之间使用 degree-normalized all-to-all pairwise coupling；弱连接条件取 $K_{\mathrm{out}}=0.04$。

## 配对的 $2\times2$ 对照

两个 treatment factor 为：

- 过程噪声强度：$0$ 或 $0.08$；
- 跨模块耦合：$0$ 或 $0.04$。

四个条件使用相同 seeds、初始训练相位、训练/测试划分、干预相位、模型结构、训练预算和 EI 估计器。实验单位为 seed，共使用三个 seeds。

每个 seed 和条件先生成 4800 个随机有限时转移。初始相位独立服从

$$
\theta_i\sim\mathrm{Uniform}(-\pi,\pi).
$$

积分步长为 $0.01$，预测跨度为 $\tau=0.2$。过程噪声在每个积分步加入，标准差按 $\sqrt{dt}$ 缩放。

## 动力学识别与 target

非线性动力学模型接收五个振子的圆周特征，并直接预测完整五维未来相位变化

$$
\mathbf{Y}
=\Delta_\tau\boldsymbol{\theta}_{1:5}
=\operatorname{wrap}\!\left(
\boldsymbol{\theta}_{t+\tau}-\boldsymbol{\theta}_t
\right).
$$

target 使用所有五个振子的未来变化，而不是只读取振子 1 和 5，也不是使用标量汇总。使用相位增量可去除绝对 next state 中平凡的 identity/persistence，同时仍保留整个五维系统的有限时响应。此前用不一致的 subset 估计器审计绝对未来相位时曾出现负 $\Phi$；该现象不能归因于动力学本身，因而不作为选择 target 或解释结果的理论证据。

模型在训练转移上拟合后，从 held-out 残差协方差估计随机响应。PEID readout 使用另外 6000 个独立均匀相位干预，并对学习模型采样。真实条件三个 seeds 的 held-out 圆周 MAE 为 $0.066$–$0.074$ rad。

## 同一联合 TM 的 EI 估计

每个 seed 只拟合一个连续条件 transport map

$$
p_{\mathrm{TM}}(\mathbf{y}\mid\boldsymbol{\theta}),
$$

其中 source context 为 16 维圆周 Fourier 特征，target 为完整五维 $\mathbf{Y}$。已知的独立均匀干预分布 $q(\boldsymbol{\theta})$ 与该条件 TM 共同定义唯一联合分布

$$
p_{\mathrm{joint}}(\boldsymbol{\theta},\mathbf{y})
=q(\boldsymbol{\theta})p_{\mathrm{TM}}(\mathbf{y}\mid\boldsymbol{\theta}).
$$

所有 31 个非空 source subset 的边缘

$$
p_{\mathrm{joint}}(\mathbf{y}\mid\boldsymbol{\theta}_S)
=\int q(\boldsymbol{\theta}_{\bar S})
p_{\mathrm{TM}}(\mathbf{y}\mid\boldsymbol{\theta}_S,
\boldsymbol{\theta}_{\bar S})
\,d\boldsymbol{\theta}_{\bar S}
$$

都从这个联合分布用共同 scrambled Sobol 点积分得到；没有为不同 subset 重新拟合模型。target-only triangular TM 是嵌套空模型：只有当条件 TM 的 held-out 对数似然增益超过两个 paired SEM 时才保留 source context。三个 target-shuffle 对照均选择空模型，因此在所选独立联合分布下 EI 解析地为 0；这不是对估计值截断。

## 非负性与数值收敛

由于 $q(\boldsymbol{\theta})=\prod_iq_i(\theta_i)$，同一联合分布下

$$
\Phi(S)
=I(\boldsymbol{\theta}_S;\mathbf{Y})
-\sum_{i\in S}I(\theta_i;\mathbf{Y})
=\operatorname{TC}(\boldsymbol{\theta}_S\mid\mathbf{Y})
\ge 0.
$$

同理，对任意不交并 $S=L\mathbin{\dot\cup}R$，

$$
\Phi(S)-\Phi(L)-\Phi(R)
=I(\boldsymbol{\theta}_L;\boldsymbol{\theta}_R\mid\mathbf{Y})
\ge0.
$$

实现中没有 `max(0,\cdot)`、非负截断或单调投影。先前的小负值来自有限内层粒子对
$\log\mathbb{E}[p_{\mathrm{TM}}]$ 的 Jensen 偏差，而不是上述恒等式失效。现在对共同 Sobol 边缘积分使用 two-block jackknife 消去一阶 `log-mean` 偏差；若任一原始 $\Phi$ 或 partition residual 为负，则保持同一个已拟合 TM，依次把积分规模从 $2048\times512$ 增至 $2048\times1024$ 和 $4096\times2048$。12 个正式重复最终的最小原始 $\Phi$ 为 $0.0043$–$0.0537$ bits。

通过检查后，Greedy 对每个非平凡二分计算

$$
C(L,R)=\Phi(L)+\Phi(R)
$$

并选择最大值；停止阈值为 $10^{-5}$ bits。
当递归块只剩两个节点时，唯一二分必然落到两个 singleton；图中不再显示无信息量的 $C=0$，而直接显示该终端原子
$\Xi_S\equiv\Phi(S)$。

## 代表 seed

![混合阶 Kuramoto 的 learned-dynamics Greedy 层级](assets/mixed_order_kuramoto_hierarchy/validation.png)

图 a 上半部分给出 pairwise、triadic 与弱跨模块连接；下半部分直接展示带过程噪声的五维相位信号
$\sin\theta_i(t)$。五条曲线仅为避免遮挡而纵向错开，不改变各通道的时间变化。

代表 seed 的真实条件满足

$$
\Phi(\{1,2,3,4,5\})=4.230\ \text{bits}.
$$

15 个根候选中，正确切分

$$
\{1,2\}\mid\{3,4,5\}
$$

捕获 $3.778$ bits；第二名为 $2.481$ bits，根层保留 $0.452$ bits 的整合残差。下一层有

$$
\Phi(\{1,2\})=1.830\ \text{bits},
\qquad
\Phi(\{3,4,5\})=1.947\ \text{bits}.
$$

`{1,2}` 的唯一下一步切分是两个 singleton，因此直接显示
$\Xi_{\{1,2\}}=1.830$ bits。`{3,4,5}` 的三个候选均为正；最大候选
`{3}|{4,5}` 捕获 $0.410$ bits，留下 $1.538$ bits 的三体原子。随后继续展示二节点终点
$\Xi_{\{4,5\}}=0.410$ bits。这构成图 b 中从根排序到终端原子的完整递归 Greedy 过程。

## 跨 seed 结果

| 条件 | 根分区恢复 | 根残差（bits） | pairwise atom（bits） | triadic atom（bits） | 其他正原子（bits） |
|---|---:|---:|---:|---:|---:|
| 无噪声、无跨模块连接 | 3/3 | $0.719\pm0.090$ | $1.892\pm0.051$ | $1.841\pm0.225$ | $0.578\pm0.089$ |
| 仅过程噪声 | 3/3 | $0.587\pm0.055$ | $1.805\pm0.023$ | $1.783\pm0.121$ | $0.490\pm0.042$ |
| 仅弱跨模块连接 | 3/3 | $0.688\pm0.094$ | $1.879\pm0.013$ | $1.731\pm0.230$ | $0.593\pm0.099$ |
| 过程噪声与弱连接同时存在 | 3/3 | $0.641\pm0.096$ | $1.801\pm0.020$ | $1.902\pm0.200$ | $0.536\pm0.064$ |

四个条件共 $12/12$ 个 seed-condition 组合恢复 planted 根切分。噪声、弱连接和有限模型误差使根残差及其他原子合理地非零。真实条件的三个 target-shuffle 均未通过 source-context 增益门槛，根 $\Phi$ 为 0。

## 解释边界

该实验支持：在当前噪声、弱连接、模型容量和干预支持下，learned-dynamics Greedy 能稳定恢复已知 `2+3` 根分区。

该实验不支持以下更强结论：

- 当前有限 TM 容量与积分精度可无条件推广到更高维系统；
- jackknife 已消除所有高阶有限粒子偏差；
- Greedy 树能唯一表示共享节点或重叠超边。

## 复现

```bash
python scripts/validate_mixed_order_kuramoto_hierarchy.py --mode full --seeds 3
```

机器可读结果保存在
`results/mixed_order_kuramoto_hierarchy/summary.json`，其中包含每个条件和 seed 的动力学拟合误差、同一联合 TM 的 EI 表、边缘积分收敛轨迹、完整 Greedy trace 和 target-shuffle 负控。图保存在
`docs/ref/assets/mixed_order_kuramoto_hierarchy/validation.{png,svg,pdf}`。
