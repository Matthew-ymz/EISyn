# Lorenz-96 TM-EI 因果恢复实验

本实验检验 transport-map effective information（TM-EI）在连续非线性动力系统中恢复局部因果结构的能力。实验对象为 8 维 Lorenz-96 环形系统，观测变量记为 \(M_1,\ldots,M_8\)。给定当前状态 \(\mathbf{x}(t)=(x_1(t),\ldots,x_N(t))^\top\)，目标是从样本对 \(\mathbf{x}(t)\rightarrow \mathbf{x}(t+\tau)\) 或其增量中估计源变量 \(M_j(t)\) 对目标变量 \(M_i(t+\tau)\) 的 pairwise TM-EI，并将每个目标的 top-4 入边与真实动力学图比较。

## Ground Truth 动力学方程

Lorenz-96 系统的连续时间动力学为

```math
\frac{d x_i}{dt}
= \left(x_{i+1}-x_{i-2}\right)x_{i-1}-x_i+F,
\qquad i=1,\ldots,N,
```

其中 \(N=8\)，\(F=8.0\)，索引采用周期边界条件：

```math
x_{i+N}=x_i.
```

因此，目标变量 \(x_i\) 的结构方程只显式依赖四个源：自身 \(x_i\)、前一邻居 \(x_{i-1}\)、前二邻居 \(x_{i-2}\) 与后一邻居 \(x_{i+1}\)。本实验据此定义 source-to-target ground truth 邻接矩阵：

```math
\mathbf{A}_{ij}=1
\quad\Longleftrightarrow\quad
j \in \{i,\ i-1,\ i-2,\ i+1\}\pmod N.
```

该定义使邻接矩阵的行表示 target \(M_i\)，列表示 source \(M_j\)。由于每个 target 恰有 4 个真实入边，评估时对 TM-EI 矩阵逐行保留 top-4 source，与 ground truth 进行 accuracy、F1 和 ROC-AUC 比较。

## 实验设计

### 1. 直接 next-state 观测估计

基线实验直接读取无表头的 `yt.csv` 与 `yt+1.csv`，构造一步样本对

```math
\mathbf{x}(t) \rightarrow \mathbf{x}(t+1),
```

并对所有 \((M_j(t),M_i(t+1))\) 组合估计 pairwise TM-EI。该设置最接近原始观测数据，但在连续时间短步长轨迹中，\(M_i(t)\) 与 \(M_i(t+1)\) 存在强自相关，因而对角线 EI 往往显著高于邻居耦合项。这一现象反映状态持久性，而不应直接解释为最优因果恢复。

复现命令：

```bash
python -m exp.TM.lorzen_tm_ei exp/TM/lorzen/input \
  --output-dir exp/TM/lorzen/results \
  --top-k 4
```

### 2. 滞后 next-state 观测估计

为降低一步采样中的自相关主导效应，实验从相邻切片重构轨迹，并估计

```math
\mathbf{x}(t) \rightarrow \mathbf{x}(t+\tau).
```

滞后扫描显示，\(\tau=3\) 时既保留局部动力学传播，又显著削弱一步对角线优势，是当前观测数据路径下最稳定的设置。

推荐命令：

```bash
python -m exp.TM.lorzen_tm_ei exp/TM/lorzen/input \
  --output-dir exp/TM/lorzen/results_lag3 \
  --top-k 4 \
  --target-mode next \
  --estimator-mode observed \
  --lag 3
```

### 3. 增量目标与 surrogate intervention

第二类校正把目标改为状态增量：

```math
\Delta x_i(t)=x_i(t+1)-x_i(t).
```

该目标直接对应连续系统的局部速度场，因而可以弱化 \(x_i(t)\rightarrow x_i(t+1)\) 的持久性贡献。实验进一步拟合二次 transition surrogate，并在源变量均值附近的独立均匀干预盒内采样：

```math
\tilde{\mathbf{x}}\sim\mathrm{Unif}
\left(\bar{\mathbf{x}}-\frac{L}{2},\bar{\mathbf{x}}+\frac{L}{2}\right),
```

再用 surrogate 生成目标并估计 TM-EI。该设计将观测相关性与干预式输入分布分离，使 `L` 与 intervention sample count 成为可显式扫描的实验变量。

推荐命令：

```bash
python -m exp.TM.lorzen_tm_ei exp/TM/lorzen/input \
  --output-dir exp/TM/lorzen/results_tuned \
  --top-k 4 \
  --target-mode delta \
  --estimator-mode surrogate_intervention \
  --box-width 1.5 \
  --sample-count 2000 \
  --seed 17
```

### 4. Lorenz-96 MLP + TM-EI intervention

第三类实验不依赖外部给定的相邻切片，而是用 RK4 数值积分生成 Lorenz-96 轨迹，训练 `StandardScaler + MLPRegressor` 预测 \(\mathbf{x}(t+\tau)\)，再从独立均匀干预分布采样输入，经 MLP 生成目标后计算 TM-EI。该流程检验 TM-EI 在“已学习动力学模型 + 干预采样”条件下能否恢复真实结构。

复现命令：

```bash
python -m exp.TM.lorzen_tm_ei \
  --simulate-lorenz96-mlp \
  --output-dir exp/TM/lorzen/results_mlp \
  --top-k 4 \
  --lag 15 \
  --box-width 8.0 \
  --sample-count 3000 \
  --steps 8000 \
  --burn-in 1000 \
  --train-sample-count 5000 \
  --hidden-layer-sizes 128,64 \
  --max-iter 1000 \
  --seed 4
```

该路径额外输出单面板 causal graph：先按每个 target 的 row-wise top-4 选择边，再显示被选择单元格的非负 raw EI 强度，未选择单元格置零。这样避免固定绝对阈值过小导致弱但排序正确的邻居边被视觉上淹没。

## 结果解读

| 设置 | target | estimator | 关键参数 | Accuracy | F1 | AUC |
|---|---|---|---:|---:|---:|---:|
| 直接观测一步 | next | observed | lag=1 | 0.750 | 0.750 | 0.897 |
| 滞后观测 | next | observed | lag=3 | 1.000 | 1.000 | 1.000 |
| 观测增量 | delta | observed | lag=1 | 0.750 | 0.750 | 0.923 |
| surrogate 干预 | delta | surrogate_intervention | L=1.5, n=2000 | 1.000 | 1.000 | 1.000 |
| MLP 干预 | next | mlp_intervention | lag=15, L=8.0, n=3000 | 1.000 | 1.000 | 0.990 |

主要结论如下。

1. 一步 next-state 的失败模式不是 TM-EI 无法识别 Lorenz-96 结构，而是短时间步连续轨迹的自保持项过强。lag sweep 显示，lag=1 时对角线均值与非对角线均值之比约为 43.0；lag=3 时该比值下降至约 5.8，同时 accuracy、F1 与 AUC 均达到 1.000。继续增大 lag 后，局部因果传播被更长时间的非线性混合稀释，性能反而下降。

2. 增量目标更接近真实微分方程右端，但仅在观测分布上估计 delta 仍不足以完全恢复结构。引入 surrogate intervention 后，独立均匀输入打破了观测轨迹中的相关性结构，参数扫描中多个 \(L\) 与 sample count 组合达到或接近满分，说明恢复结果不是单一超参数偶然产物。

3. MLP intervention 路径在独立模拟数据上同样恢复了 top-4 ground truth 图。训练集 \(R^2=0.997\)，测试集 \(R^2=0.996\)，说明 MLP 近似了 Lorenz-96 多步 transition；在此基础上的 TM-EI top-4 图达到 accuracy=1.000、F1=1.000、AUC=0.990，支持“先学习动力学，再在干预分布上估计因果强度”的实验路线。

4. 因果图评估采用 row-wise top-4，而非固定全局阈值。原因是每个 Lorenz-96 target 的真实入度已知为 4，且不同 target 的 EI 数值尺度可能不同；逐行 top-k 更符合“每个方程恢复其结构源项”的科学问题。

### 协同项验证

为检验“邻居边较弱是因为真实机制含双线性交互，而不是这些边不真实”的解释，额外计算二源 transport-map 协同残差：

```math
\mathrm{Syn}(X_a,X_b\rightarrow Y)
= \mathrm{EI}(X_a,X_b\rightarrow Y)
- \mathrm{EI}(X_a\rightarrow Y)
- \mathrm{EI}(X_b\rightarrow Y).
```

该验证直接针对 Lorenz-96 的两个乘积子项：

```math
x_{i+1}x_{i-1},
\qquad
x_{i-2}x_{i-1},
```

以及完整邻居交互项 \((x_{i+1}-x_{i-2})x_{i-1}\)。在与 MLP intervention 图一致的 uniform-box 采样口径下，乘积子项的单源 EI 均值接近 0，但二源 joint EI 均值约为 8.74，协同残差也约为 8.74；对完整交互项，二源 joint EI 均值约为 0.35，协同残差约为 0.35。这说明乘积机制主要由二源组合解释，单个邻居变量本身只能解释很少信息。

在 observed 轨迹口径下，乘积子项仍表现出强协同：positive product 的 joint EI 均值为 4.30，协同残差为 4.09；negative product 的 joint EI 均值为 4.26，协同残差为 4.15。完整交互项的二源协同残差约为 0.31--0.34。相反，完整 RHS 中线性自项 \(-x_i\) 的 observed EI 均值约为 1.09，说明观测轨迹上自项和状态持久性会显著抬高对角线边权。

因此，当前 EI 因果图中不同边权差距大的直接机制证据是：对角线边来自线性自项与状态持久性，容易被单源 EI 捕获；邻居机制来自双线性乘积项，信息主要储存在二源协同中，拆成 pairwise 单边后每条边的 raw EI 会显著小于对角线。

## 输出文件

- `results/lorzen_tm_ei_matrix.csv`：target-by-source TM-EI 矩阵。
- `results/lorzen_tm_ei_edges.csv`：长表形式的边强度、raw MI 与 bias correction。
- `results/lorzen_tm_ei_topk_graph.csv`：逐 target top-4 阈值化后的 TM 图。
- `results/lorzen_lorenz96_groundtruth.csv`：source-to-target Lorenz-96 ground truth 图。
- `results/lorzen_tm_ei_heatmap.png`：TM-EI、预测图与 ground truth 的三面板比较图。
- `results/lorzen_tm_ei_summary.json`：指标与产物路径摘要。
- `results_lag3/`：推荐的 lag=3 观测 next-state 结果。
- `results_lag_sweep/lorzen_tm_lag_sweep.csv`：观测 next-state 的 lag 扫描表。
- `results_lag_sweep/lorzen_tm_lag_sweep.png`：lag 扫描诊断图。
- `results_tuned/`：推荐的 delta-target surrogate-intervention 结果。
- `results_sweep/lorzen_tm_parameter_sweep.csv`：`L` 与 intervention sample count 参数扫描表。
- `results_sweep/lorzen_tm_parameter_sweep.png`：参数扫描诊断图。
- `results_mlp/`：模拟 Lorenz-96 MLP-intervention 结果。
- `results_mlp/lorzen_mlp_training_history.csv`：MLP 训练损失曲线数据。
- `results_mlp/lorzen_ei_causal_graph.png`：按 row-wise top-4 显示 raw nonnegative EI 的单面板因果图。
- `results_synergy/uniform_box/lorzen_product_synergy.csv`：uniform-box 采样下乘积项与完整交互项的二源协同验证。
- `results_synergy/uniform_box/lorzen_rhs_synergy.csv`：uniform-box 采样下 RHS 自项与邻居乘积对的 EI 对照。
- `results_synergy/observed/lorzen_product_synergy.csv`：观测轨迹采样下乘积项与完整交互项的二源协同验证。
- `results_synergy/observed/lorzen_rhs_synergy.csv`：观测轨迹采样下 RHS 自项与邻居乘积对的 EI 对照。
