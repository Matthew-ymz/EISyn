# HCP Lausanne-83 真实数据 PhiEID 操作方案

本文记录当前 HCP 数据下载进度，并给出用真实 HCP 数据替换原 synthetic smoke 的具体操作方案。当前结果只覆盖 1 个 subject 的最小真实 pilot，不能代表 HCP 群体结论。

## 1. 当前下载状态

当前已经下载到最小可用真实数据集：

| Subject | Run | 文件 | 状态 |
|---|---|---|---|
| 100307 | REST1_LR | `MNINonLinear/aparc+aseg.nii.gz` | 已完成，913 KB |
| 100307 | REST1_LR | `MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz` | 已完成，948 MB |
| 100307 | REST1_LR | `MNINonLinear/Results/rfMRI_REST1_LR/Movement_RelativeRMS.txt` | 已完成，11 KB |

本地路径：

```text
data/hcp_s1200/HCP_1200/100307/MNINonLinear/aparc+aseg.nii.gz
data/hcp_s1200/HCP_1200/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz
data/hcp_s1200/HCP_1200/100307/MNINonLinear/Results/rfMRI_REST1_LR/Movement_RelativeRMS.txt
```

没有检测到仍在运行的 `aws s3 cp` 或 HCP pilot 下载进程。

## 2. 已完成的真实数据最小运行

已用真实 HCP 数据跑通 1 个 subject、1 个 run、20 个 circular-shift null：

```bash
/opt/anaconda3/envs/hcp-neuro/bin/python scripts/run_hcp_lausanne_phi_eid_pilot.py \
  --subjects 100307 \
  --runs REST1_LR \
  --null-reps 20 \
  --output-dir results/hcp_lausanne_phi_eid_pilot_real_minimal \
  --null-figure-base fig/hcp_lausanne_phi_eid_real_minimal_null_comparison \
  --decomposition-figure-base fig/hcp_lausanne_phi_eid_real_minimal_decomposition \
  --report docs/log/hcp_lausanne_phi_eid_real_minimal.md
```

ROI 提取结果：

| 项目 | 数值 |
|---|---:|
| ROI time series shape | 1200 x 83 |
| lagged sample count | 1199 |
| empty ROI | 0 |
| voxel count min / median / max | 111 / 916 / 3964 |
| movement RMS mean | 0.062254 |
| movement RMS max | 0.163717 |

输出文件：

```text
results/hcp_lausanne_phi_eid_pilot_real_minimal/summary.json
results/hcp_lausanne_phi_eid_pilot_real_minimal/roi_timeseries/sub-100307_REST1_LR_lausanne83_timeseries.npz
fig/hcp_lausanne_phi_eid_real_minimal_null_comparison.{png,svg,pdf}
fig/hcp_lausanne_phi_eid_real_minimal_decomposition.{png,svg,pdf}
docs/log/hcp_lausanne_phi_eid_real_minimal.md
```

## 3. 当前最小真实结果

Ridge 一步预测质量：

| 指标 | 数值 |
|---|---:|
| validation correlation | 0.912155 |
| Ridge RMSE | 0.589450 |
| persistence RMSE | 0.614816 |
| RMSE / persistence RMSE | 0.958743 |

这说明 Ridge transition model 略优于 persistence baseline，可以作为第一版 Gaussian PhiEID screening 的 transition model。

PhiEID 对比：

| 指标 | 数值 |
|---|---:|
| observed raw PhiEID | 13.582205 bits |
| whole EI | 63.015498 bits |
| singleton EI sum | 49.433293 bits |
| null raw PhiEID mean | 8.219855 bits |
| null raw PhiEID std | 0.372700 bits |
| null raw PhiEID min / max | 7.675046 / 9.127837 bits |
| null >= observed | 0 / 20 |
| empirical p-value | 0.047619 |

当前最小真实结果支持一个很窄的判断：在 subject `100307` 的 `REST1_LR` 上，真实 ROI 同步结构下的 PhiEID 高于独立 circular-shift null。这个结论仍是 single-subject pilot，不应写成 HCP 群体规律。

![Real minimal null comparison](../../fig/hcp_lausanne_phi_eid_real_minimal_null_comparison.png)

## 4. Phi 分解的当前读数

模块级 greedy atoms 的前几项：

| Rank | Atom | PhiEID |
|---:|---|---:|
| 1 | DMN + Vis + VAN + FPN + Lim + Sub | 2.639192 |
| 2 | DMN + Vis + FPN + Lim + Sub | 2.292675 |
| 3 | DMN + Som + Vis + VAN + FPN + Lim + Sub | 2.162849 |
| 4 | DMN + Vis + FPN + Lim | 1.848999 |
| 5 | DMN + Vis + FPN | 1.420971 |
| 6 | DMN + FPN | 1.154656 |

ROI leave-one-out burden 的 top candidates：

| Rank | ROI | Module | Burden |
|---:|---|---|---:|
| 1 | ctx-rh-parstriangularis | FPN | 0.612599 |
| 2 | ctx-rh-superiorfrontal | FPN | 0.600080 |
| 3 | ctx-rh-supramarginal | VAN | 0.592918 |
| 4 | ctx-lh-superiorparietal | VAN | 0.587995 |
| 5 | ctx-lh-superiortemporal | DMN | 0.586883 |
| 6 | ctx-rh-lateraloccipital | Vis | 0.581638 |
| 7 | ctx-rh-middletemporal | DMN | 0.562435 |
| 8 | ctx-lh-lateraloccipital | Vis | 0.546748 |
| 9 | ctx-rh-inferiorparietal | VAN | 0.545514 |
| 10 | ctx-lh-lingual | Vis | 0.530770 |

当前分解读数提示高阶 residual 主要涉及 transmodal-control 与 sensory network 的组合，尤其包含 DMN、FPN、VAN、Visual、Limbic 和 Subcortical。ROI 级结果只能解释为候选 burden，不是精确的 83D exhaustive high-order atom。

![Real minimal decomposition](../../fig/hcp_lausanne_phi_eid_real_minimal_decomposition.png)

## 5. 替换 synthetic 结果的操作方案

### Step 1: 保留 synthetic 文档作为流程验证记录

原文档 `docs/reports/HCP_Lausanne83_PhiEID_pilot_conclusion.md` 记录的是 synthetic smoke 的负结果。它不应被直接覆盖，因为它说明 null 检验和报告流程能拦住无效信号。

推荐新增或后续更新一份真实数据报告，引用以下真实数据输出：

```text
results/hcp_lausanne_phi_eid_pilot_real_minimal/summary.json
fig/hcp_lausanne_phi_eid_real_minimal_null_comparison.png
fig/hcp_lausanne_phi_eid_real_minimal_decomposition.png
docs/log/hcp_lausanne_phi_eid_real_minimal.md
```

### Step 2: 先把单 subject 跑法固定为真实 smoke

以后检查真实数据读取是否正常，用这个命令：

```bash
conda activate hcp-neuro

python scripts/run_hcp_lausanne_phi_eid_pilot.py \
  --subjects 100307 \
  --runs REST1_LR \
  --null-reps 20 \
  --output-dir results/hcp_lausanne_phi_eid_pilot_real_minimal \
  --null-figure-base fig/hcp_lausanne_phi_eid_real_minimal_null_comparison \
  --decomposition-figure-base fig/hcp_lausanne_phi_eid_real_minimal_decomposition \
  --report docs/log/hcp_lausanne_phi_eid_real_minimal.md
```

通过标准：

- ROI time series 是 `1200 x 83`。
- 没有 empty ROI。
- Ridge validation correlation 为正。
- `RMSE / persistence RMSE < 1`。
- observed PhiEID 高于大多数 null。

### Step 3: 下载并运行 5-10 个 subject

当前默认 subject list：

```text
100307,103414,105115,110411,111312,113619,115320,117122,118528,118730
```

下载并运行：

```bash
conda activate hcp-neuro

AWS_PROFILE=hcp python scripts/run_hcp_lausanne_phi_eid_pilot.py \
  --download \
  --subjects 100307,103414,105115,110411,111312,113619,115320,117122,118528,118730 \
  --runs REST1_LR \
  --null-reps 20 \
  --output-dir results/hcp_lausanne_phi_eid_pilot_real_10sub \
  --null-figure-base fig/hcp_lausanne_phi_eid_real_10sub_null_comparison \
  --decomposition-figure-base fig/hcp_lausanne_phi_eid_real_10sub_decomposition \
  --report docs/log/hcp_lausanne_phi_eid_real_10sub.md
```

如果已提前下载好数据，可以去掉 `--download`。

### Step 4: 做真实 Phi 识别和对比

主检验：

$$
\Phi^{EID}
= EI(\mathbf{x}_t; \mathbf{x}_{t+1})
- \sum_i EI(x_t^i; \mathbf{x}_{t+1}).
$$

其中 $\mathbf{x}_t$ 是 83 维 ROI 状态。真实结果需要同时报告：

- 每个 subject/run 的 observed raw PhiEID。
- 每个 subject/run 的 null distribution。
- empirical p-value。
- group-level observed minus null mean。
- Ridge validation correlation 和 persistence baseline skill。
- motion summary。

对比判断：

- 若多数 subject 的 observed PhiEID 高于 null，且 Ridge 预测优于 persistence，可以进入模块/ROI 解释。
- 若 PhiEID 高于 null 但 Ridge 弱于 persistence，优先修模型或预处理。
- 若 PhiEID 不高于 null，不解释分解结果。

### Step 5: 做 Phi 分解

模块级：

- 使用 DMN/Som/Vis/VAN/DAN/FPN/Lim/Sub 映射。
- 报告 greedy atom 的均值、出现频率和跨 subject 稳定性。
- 优先解释跨 subject 稳定出现的 atom，而不是单 subject 最大项。

ROI 级：

- 使用 leave-one-out burden。
- 报告 top ROI 的平均 burden 和 top-k 出现频率。
- 只称为 candidate burden，不称为精确高阶原子。

### Step 6: 扩展可靠性检查

完成 5-10 人后再做：

```bash
python scripts/run_hcp_lausanne_phi_eid_pilot.py \
  --download \
  --subjects 100307,103414,105115,110411,111312,113619,115320,117122,118528,118730 \
  --runs REST1_LR,REST1_RL \
  --null-reps 20 \
  --output-dir results/hcp_lausanne_phi_eid_pilot_real_10sub_rest1_lrrl \
  --null-figure-base fig/hcp_lausanne_phi_eid_real_10sub_rest1_lrrl_null_comparison \
  --decomposition-figure-base fig/hcp_lausanne_phi_eid_real_10sub_rest1_lrrl_decomposition \
  --report docs/log/hcp_lausanne_phi_eid_real_10sub_rest1_lrrl.md
```

重点看：

- `REST1_LR` 和 `REST1_RL` 的 PhiEID 是否方向一致。
- top module atoms 是否稳定。
- top ROI burden 是否集中在相似网络。
- motion 是否解释了 PhiEID。

## 6. 当前可以写进报告的一句话

当前真实最小 pilot 已经替代 synthetic smoke 跑通：subject `100307` 的 `REST1_LR` 提取出完整的 83 ROI 时间序列，Ridge transition model 略优于 persistence，观测 raw PhiEID 为 13.58 bits，高于 20 个 circular-shift null 的全部取值，经验 p-value 为 0.0476。这个结果支持继续扩展到 5-10 个 subject 做真实数据验证，但还不能作为 HCP 群体神经科学结论。

