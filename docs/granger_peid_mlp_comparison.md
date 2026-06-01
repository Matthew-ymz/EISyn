# 统一动力系统：共同驱动 + sine 协同

这个例子把两种容易混淆的结构放进同一个动力系统：一方面，`w` 是 `x`、`y` 背后的共同原因；另一方面，`x`、`y` 对 `z` 的作用不是两条可分离的 pairwise 边，而是一个二源协同项。系统为

$$
\begin{aligned}
w_{t+1} &= 0.78w_t + \eta^w_t,\\
x_{t+1} &= 0.42x_t + 0.82w_t + \eta^x_t,\\
y_{t+1} &= 0.38y_t + 0.76w_t + \eta^y_t,\\
z_{t+1} &= 0.22z_t + \alpha\sin\left(x_t y_t\right) + \eta^z_t.
\end{aligned}
$$

其中 `w` 不直接进入 `z` 的结构方程。真实机制应读成两层：

- pairwise 层面：`w -> x`、`w -> y`；
- 高阶层面：`{x, y} -> z`；
- 非结构边：`w -> z` 不是直接机制边，单独的 `x -> z`、`y -> z` 也只是 sine 协同项的 pairwise 投影。

## 读出方式

同一条模拟时间序列先用于训练一个 MLP 一步转移模型，输入为 `[x_t, y_t, z_t, w_t]`，输出为 `[x_{t+1}, y_{t+1}, z_{t+1}, w_{t+1}]`。随后在固定 MLP 上读出四类量：

- Granger/ablation：把某个 source 的输入列替换为均值，记录目标预测 MSE 的增量。它回答“去掉这个变量会不会损害预测”。
- SHAP 类归因：在同一 fitted MLP 上做 interventional Shapley 读出，用经验背景替换未给定特征。单特征 SHAP 报告 mean absolute attribution；二阶 SHAP interaction 报告 `x:y` 的 mean absolute interaction。前者回答“某个特征分到多少预测贡献”，后者回答“两个特征的非加性预测贡献有多大”。
- 交互项 probe：在同一 fitted MLP 的最大熵干预预测面上，用标准化主效应加一个二阶乘积项拟合目标输出，并记录该乘积项相对于主效应模型的 incremental `R^2`。它回答“固定这个预测器时，响应面是否含有可由 `x:y` 近似的二阶非加性形状”。
- PEID：先做最大熵独立干预，再计算 single-source EI、joint EI 和 synergy：

$$
\mathrm{Syn}(\{x,y\}\to z)
= \max\left(0,\; EI(\{x,y\}\to z)-EI(x\to z)-EI(y\to z)\right).
$$

## 代表性结果

| quantity | value |
| --- | ---: |
| fitted MLP final training loss | 0.1546 |
| Granger/ablation `w -> x` | 0.3356 |
| Granger/ablation `w -> y` | 0.3002 |
| Granger/ablation `w -> z` | 0.006681 |
| Granger/ablation `x -> z` | 0.2121 |
| Granger/ablation `y -> z` | 0.2078 |
| SHAP mean abs `w -> x` | 0.4859 |
| SHAP mean abs `w -> y` | 0.4569 |
| SHAP mean abs `w -> z` | 0.01567 |
| SHAP mean abs `x -> z` | 0.1393 |
| SHAP mean abs `y -> z` | 0.1357 |
| SHAP interaction mean abs `x:y -> z` | 0.3625 |
| product interaction `x:y -> z` incremental `R^2` | 0.8669 |
| product interaction `x:y -> z` coefficient | 0.3758 |
| product interaction `w:x -> z` incremental `R^2` | 0.001375 |
| product interaction `w:y -> z` incremental `R^2` | 0.001088 |
| PEID pairwise EI `w -> x` | 0.6961 |
| PEID pairwise EI `w -> y` | 0.6918 |
| PEID pairwise EI `w -> z` | 0.01494 |
| PEID pairwise EI `x -> z` | 0.1475 |
| PEID pairwise EI `y -> z` | 0.135 |
| PEID joint EI `{x, y} -> z` | 1.081 |
| PEID synergy `{x, y} -> z` | 0.7989 |

![同一 MLP 上的二维读出对照](../fig/granger_peid_mlp_comparison/sine_readout_2d_summary.png)

图中左侧热图把 Granger、SHAP 和 PEID 的单源读出放在同一组边上比较。因为三行的单位不同，颜色只在每一行内部归一化，格子里的数字才是原始读数。右上角显示标准化乘积项对 `z` 的增量解释度：`x:y` 明显高于 `w:x` 与 `w:y`。右下角显示 PEID 对 `z` 的信息分解，联合 EI 与 synergy 高于单源 EI。

## alpha 扫描：SHAP 交互与 PEID 协同

![alpha 扫描下的 SHAP 与 PEID 对照](../fig/granger_peid_mlp_comparison/sine_alpha_shap_peid_sweep.png)

| alpha | SHAP `x->z` | SHAP `y->z` | SHAP `w->z` | SHAP interaction `|x:y|` | product probe incremental `R^2` | PEID joint EI `{x,y}->z` | PEID synergy `{x,y}->z` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.004416 | 0.002329 | 0.003175 | 0.002055 | 0.00873 | 0.1859 | 0.03884 |
| 0.20 | 0.03171 | 0.03216 | 0.009361 | 0.06448 | 0.7161 | 0.8641 | 0.6433 |
| 0.40 | 0.06578 | 0.06662 | 0.01125 | 0.1258 | 0.7124 | 0.9199 | 0.7038 |
| 0.60 | 0.09785 | 0.1008 | 0.01287 | 0.2028 | 0.7314 | 0.9471 | 0.7141 |
| 0.80 | 0.1307 | 0.1353 | 0.01678 | 0.2791 | 0.7365 | 0.978 | 0.7339 |
| 1.00 | 0.1637 | 0.1687 | 0.01961 | 0.3523 | 0.7371 | 0.9757 | 0.7297 |

这里的 `alpha` 是 sine 项前面的强度系数。`alpha=0` 时，`z` 只剩自身记忆与噪声，SHAP 二阶交互和产品项增量解释度接近零；PEID 仍保留少量估计底噪和分箱残差。随着 `alpha` 增大，SHAP 单源 `x->z`、`y->z` 与 SHAP interaction 同时上升，但单源项是对协同响应的归因分摊，不是结构边；SHAP interaction 与产品项 probe 反映的是 fitted MLP 响应面的二阶非加性形状；PEID joint EI 与 synergy 反映的是在最大熵联合干预下 `{x,y}` 对目标分布施加的机制信息约束。



## 解释

`w -> x` 和 `w -> y` 在 Granger/ablation 与 PEID pairwise EI 中都很强，说明 fitted MLP 学到了共同驱动结构。`w -> z` 很小，符合结构方程中 `w` 不直接进入 `z` 的设定；若某些归因方法给出非零 `w -> z`，应解释为 `w` 通过诱导 `x,y` 相关性形成的代理贡献，而不是直接结构边。

对 `z` 来说，Granger/ablation 会给出明显的 `x -> z` 与 `y -> z`，SHAP 类单特征归因也会倾向把 sine 项拆成单变量贡献。交互项 probe 则能进一步指出 fitted MLP 的响应面中确实存在强 `x:y` 二阶非加性项，因此它比纯单特征 SHAP 更接近“有交互”的诊断；但它仍然是响应面形状分析，不是源侧最大熵干预语义下的机制信息分解。这些读出有预测解释价值，但它们把

$$
\alpha\sin(x_t y_t)
$$

投影成了 pairwise 贡献或低阶乘积项，不能单独表达“只有联合给定 `x_t` 和 `y_t` 时才稳定确定目标响应”的机制事实。

PEID 的关键读数是 `EI({x, y} -> z)` 与 `Syn({x, y} -> z)` 均显著高于单源投影。它说明联合干预 `{x,y}` 后，目标分布的约束远超过两个单源 EI 的加和。因此这个例子的结论不是“PEID 消除了所有代理效应”，而是：在同一个 learned transition surrogate 上，PEID 可以同时保留 `w -> x,y` 的共同驱动边，以及 `{x,y} -> z` 的协同超边；Granger 和 SHAP 单特征方法主要给出预测贡献的 pairwise 投影，交互项 probe 可以提示 `x:y` 非加性存在，但 PEID 才把这个非加性读成源集合到目标的协同有效信息。
