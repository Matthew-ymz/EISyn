# Runge 多步 EI 路径方案说明

本设计说明已经合并到 [Runge_SLP_1948_2026_PEID_Result.md](Runge_SLP_1948_2026_PEID_Result.md)。当前主线以该文档为准。

当前采用方案 A：

- 用平均条件 EI 替换二阶协同时的直接边权；
- 不做非负性截断；
- 不做 top-k 稀疏化；
- 不引入谱缩放参数；
- 在 signed dense 条件 EI 矩阵上计算有限长度 graph-walk path score。

方案 A 的核心定义如下。令 $n=60$，单源 EI 为

$$
E^{(0)}_{ij}=EI(X_i\to X_j).
$$

对源 $i$、目标 $j$ 和背景源 $r\ne i,j$，定义

$$
E_{i\to j\mid r}
=
EI(X_{\{i,r\}}\to X_j)-EI(X_r\to X_j).
$$

当前 1948-2026 主结果使用完整背景源集合

$$
\mathcal{R}_{ij}=\{r:r\ne i,\ r\ne j\},
\qquad |\mathcal{R}_{ij}|=58.
$$

平均条件边权为

$$
\bar E^{(2)}_{ij}
=
\frac{1}{58}
\sum_{r\in\mathcal{R}_{ij}}
E_{i\to j\mid r},
\qquad
\bar E^{(2)}_{ii}=0.
$$

直接边矩阵与路径矩阵为

$$
\mathbf{A}^{(2)}=\bar{\mathbf{E}}^{(2)},
\qquad
\mathbf{T}^{(2)}_L
=
\sum_{\ell=1}^{L}\left(\mathbf{A}^{(2)}\right)^\ell.
$$

1948-2026 SLP 的最新主图和结果读数见合并后的主报告。
