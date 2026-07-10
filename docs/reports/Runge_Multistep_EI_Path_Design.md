

<!-- multistep-conditioned-ei-results:start -->

### 实验结果：MLP 自迭代多步条件 EI

本次重跑使用训练好的 Runge one-step MLP，不重训模型。初始历史窗口从训练集分位数范围内做 bounded maximum-entropy intervention；之后每个 horizon 都由 MLP 闭环自迭代生成，不在中间步重新采样。对每个 horizon 先估计 pairwise EI 与联合源 EI，再用
\[
\bar E^{[h],(2)}_{ij}=\frac{1}{n-2}\sum_{r\ne i,j}\left[EI(X_{\{i,r\},t}\to \widehat X_{j,t+h})-EI(X_{r,t}\to \widehat X_{j,t+h})\right]
\]
构造 signed 条件边权；地图主指标使用正部 \(E^{[h],(2)+}_{ij}=\max(\bar E^{[h],(2)}_{ij},0)\)，并累计到 \(H=10\)。

Checkpoint 结果：未触发早停，最终绘图 horizon 为 \(H=10\)。

| H | ACE top-5 overlap | ACS top-5 overlap | AMCE top-5 overlap | all matched |
|---:|---:|---:|---:|:---:|
| 1 | 5/5 | 3/5 | 2/5 | no |
| 2 | 4/5 | 2/5 | 1/5 | no |
| 3 | 3/5 | 2/5 | 1/5 | no |
| 4 | 4/5 | 3/5 | 2/5 | no |
| 5 | 4/5 | 3/5 | 2/5 | no |
| 6 | 4/5 | 2/5 | 2/5 | no |
| 7 | 4/5 | 2/5 | 2/5 | no |
| 8 | 3/5 | 2/5 | 1/5 | no |
| 9 | 3/5 | 2/5 | 1/5 | no |
| 10 | 3/5 | 1/5 | 1/5 | no |

![Runge multistep conditioned EI map](assets/runge_multistep_conditioned_ei_map.png)

Top 节点如下，括号内为对应指标值：

| Rank | ACE | ACS | AMCE |
|---:|---:|---:|---:|
| 1 | 0 (0.2303) | 36 (0.08406) | 0 (0.2296) |
| 2 | 1 (0.1707) | 2 (0.08384) | 1 (0.17) |
| 3 | 3 (0.1605) | 41 (0.0832) | 3 (0.1605) |
| 4 | 6 (0.1318) | 17 (0.08251) | 6 (0.1315) |
| 5 | 18 (0.1175) | 28 (0.08109) | 18 (0.1173) |
| 6 | 48 (0.1154) | 53 (0.08094) | 48 (0.1156) |
| 7 | 19 (0.1119) | 54 (0.08023) | 19 (0.112) |
| 8 | 4 (0.1096) | 11 (0.07987) | 5 (0.1095) |
| 9 | 5 (0.1091) | 13 (0.07965) | 4 (0.1093) |
| 10 | 26 (0.1085) | 46 (0.07957) | 26 (0.1085) |

与当前 `Part2 earth` Runge MLP-TM-EI path-effect 排名相比：

| Metric | Spearman | Kendall | top-5 overlap | top-10 overlap |
|---|---:|---:|---:|---:|
| ACE | 0.523 | 0.360 | 3/5 | 5/10 |
| ACS | 0.353 | 0.235 | 1/5 | 5/10 |
| AMCE | 0.274 | 0.180 | 1/5 | 5/10 |

视觉上，新的三项指标仍集中在原 Runge 地图中若干高影响节点附近，但数值单位保持为直接估计得到的多步 EI 累计 bit，而不是 EI 邻接矩阵连乘后的路径权重。

<!-- multistep-conditioned-ei-results:end -->
