# `tm_vs_nis.ipynb` 单参数 `TM` 协同验证设计

## 1. 目标

把 `exp/tm_vs_nis.ipynb` 从 “`NIS` 与 `TM` 两种方法对比” 改成 “只保留 `TM` 路线，在已知动力学下验证用 `TM` 估计 `EI` 并计算 `syn` 是合理的”。

这次 notebook 需要直接回答两个问题：

- 当 `alpha = 0` 时，目标动力学退化为单源加性结构，`syn` 是否接近 `0`；
- 当 `alpha` 从 `0` 增大到 `1` 时，目标中联合项占比逐渐提高，`TM` 估计得到的 `syn` 是否随之上升。

## 2. 案例设计

保留统一的输入分布，固定两个源变量：

\[
Q_2^n, Q_3^n \sim \mathrm{Uniform}[-L/2, L/2].
\]

目标动力学改成单参数家族：

\[
Q_1^{n+1} = \alpha (Q_2^n Q_3^n) + (1-\alpha) Q_2^n + \varepsilon,
\quad \alpha \in [0, 1].
\]

其中噪声项 `\varepsilon` 保持较小且固定，用来避免完全退化的确定性映射。

这个设计的解释很直接：

- `alpha = 0` 时，`Q_1^{n+1}` 只依赖 `Q_2^n`，因此协同应接近 `0`；
- `alpha` 增大时，乘积项逐步取代单源加性项，目标越来越依赖 `(Q_2^n, Q_3^n)` 的联合状态，因此 `syn` 应上升；
- 当 `alpha` 接近 `1` 时，单源 `EI` 应显著下降，而整体 `EI` 主要由联合结构支撑。

## 3. 计算口径

notebook 只保留 `TM` 估计：

- `EI^tm`
- `Syn^tm`
- `EI^tm(Q_2 \to Q_1)`
- `EI^tm(Q_3 \to Q_1)`

不再计算或展示 `NIS` 结果，也不再保留 `nis/tm` 对比图。

## 4. 输出

notebook 输出三部分：

- `alpha` 扫描后的汇总表；
- `EI^tm`、两个单源 `EI`、`Syn^tm` 随 `alpha` 变化的分解图；
- `Syn^tm / EI^tm` 以及单源占比随 `alpha` 变化的比例图。

图中重点强调：

- `alpha = 0` 时 `Syn^tm \approx 0`；
- `alpha` 递增时 `Syn^tm` 上升；
- `EI^tm(Q_2 \to Q_1)` 随 `alpha` 下降，说明原本可由单源解释的信息逐步让位给联合协同项。

## 5. 持久化

继续把图、汇总表和 manifest 写到 `fig/transport_map_mutual_information/`，并把原来的 `nis_transport_*` 命名改成新的 `tm_alpha_*` 命名，避免语义冲突。
