# UniCM：历史六项干预信息 prior 的固定参数比较

本报告保留分布修正前的干预比较。当前主图已将 ΦR、ΦWMS、ΦSI、causal density 改为自然历史—真实未来估计，并加入十一模态全阶 SURD；新结果和时间成本见[自然观测指标与 SURD 比较](unicm_observational_prior_comparison.md)。下文“当前”均指这一历史六项版本。

当前移除 **MIM**：它使用与 EI 相同的最大熵干预样本，读数就是 singleton EI，不能作为独立的普通观测 MI 对照。EI-Shapley 使用全联盟归因，仍与 singleton EI 加权不同。也移除 **Conditional MI + self MI**：这是本实验将对角 self MI 与非对角 CMI 拼接的加权接口，核对原文未将该组合作为独立指标定义。[文献核对与新增指标公式](unicm_literature_prior_expansion.md)。继续排除 O-Shapley、MMI-PID synergy、target-averaged Ξ；不新增单独二阶 PEID。

当前保留新增的两项已有定义：**正向 stochastic interaction ΦSI、causal density**。各指标的原始信息量取自文献，转换为统一 ridge 惩罚仍是本实验的比较接口；不是原论文既有的冻结 Transformer 预测方法。

**统一固定原全阶 Ξ 验证最优 α=30000、γ=3、floor_fraction=0.05。** 所有 prior 都使用自己的 attribution，启用正 γ 加权，不复制 Ξ 的惩罚向量，不搜索参数。Uniform 作为单独均匀基线，α也固定30000。Frozen和既有Univariate保留参考定义。

![六项信息 prior 的固定参数比较](../../results/unicm_information_prior_comparison_literature_shared_xi_n16384/information_prior_comparison.png)

*图1。A 为同一96个起报、264个target–lead单元的平均测试nRMSE。B 正值表示比较方法优于Ξ，横线为共享4000次循环12月块bootstrap的逐项95%区间，未作多重比较校正。C 为逐lead相对同参数Uniform的点估计改善。所有信息方法均固定α=30000、γ=3。图例在C轴外右侧，没有覆盖曲线。*

## 1. 当前结果

| 方法 | 测试 nRMSE ↓ | ACC ↑ | 相对 Frozen 的 nRMSE 降幅 |
|---|---:|---:|---:|
| 全阶 Ξ-Shapley | 0.985959 | 0.390495 | 10.43% |
| ΦR pair prior | 1.086247 | 0.289206 | 1.32% |
| EI-Shapley | 0.991224 | 0.383311 | 9.95% |
| 双输出 ΦWMS | 1.086161 | 0.290287 | 1.32% |
| 正向 ΦSI pair prior | 1.087799 | 0.293977 | 1.17% |
| Causal density（outgoing） | 1.089645 | 0.364071 | 1.01% |
| Uniform ridge（α=30000） | 1.115712 | 0.346927 | -1.36% |
| Frozen（参考） | 1.100731 | 0.291394 | 0.00% |
| Univariate（既有参考） | 1.001663 | 0.291394 | 9.00% |

Ξ 的平均测试 nRMSE 点估计最低；与 EI-Shapley 的差异区间跨零，不能从点估计排序断言显著优于它。SI 与 CD 的点估计也低于 Frozen；这包含普通线性校准收益，不能全部归因于信息 prior。

以下差值为“比较方法 nRMSE − Ξ nRMSE”，正值表示Ξ更好，与图B符号相反。

| 比较方法 | 差值 | 逐项95%区间 |
|---|---:|---:|
| ΦR pair prior | +0.100288 | [+0.057149, +0.143080] |
| EI-Shapley | +0.005265 | [-0.001875, +0.013414] |
| 双输出 ΦWMS | +0.100202 | [+0.057039, +0.143206] |
| 正向 ΦSI pair prior | +0.101840 | [+0.057682, +0.145305] |
| Causal density（outgoing） | +0.103686 | [+0.049375, +0.156148] |

固定参数下，Ξ相对ΦR、ΦWMS、正向ΦSI和CD的区间排除零；相对EI的区间跨零。后者不构成等效性证明。参数由Ξ验证集选出，这些结果回答固定参数下替换prior的问题，不能称为每个方法独立充分调参后的最优性能排名。

仅替换 attribution 的 checkpoint 来源时得到以下nRMSE；观测设计输入始终使用同一个三checkpoint冻结集成，并非三次独立重训：

| Attribution 来源 | 全阶 Ξ-Shapley | ΦR pair prior | EI-Shapley | 双输出 ΦWMS | 正向 ΦSI pair prior | Causal density（outgoing） |
|---|---:|---:|---:|---:|---:|---:|
| Checkpoint 1 | 0.986282 | 1.071890 | 0.988999 | 1.070788 | 1.073915 | 1.064254 |
| Checkpoint 2 | 0.983465 | 1.094519 | 0.990684 | 1.094679 | 1.096470 | 1.092946 |
| Checkpoint 3 | 0.982381 | 1.083619 | 0.978615 | 1.083424 | 1.082912 | 1.060449 |

Ξ相对SI与CD的方向在三个checkpoint attribution中一致；相对EI在checkpoint3发生反转。因此对低幅度差异保持保守解释。

## 2. 指标接入与权重

| 指标 | 原始信息量/归因 | 目标分辨率 |
|---|---|---|
| 全阶 Ξ-Shapley | 全2048联盟总EI扣除单源EI，精确Shapley | target × lead |
| EI-Shapley | 联盟总EI，精确Shapley，不扣单源项 | target × lead |
| ΦR / ΦWMS | 指定双过去源—双未来读出，55边各半分配 | lead，广播至目标 |
| 正向 ΦSI | 分块未来给定自身过去的条件熵和，减联合条件熵；同55边半分配 | lead，广播至目标 |
| Causal density | 全系统cross-source条件TE平均，按outgoing发送源分配；不含对角self MI | lead，广播至目标 |

源块、读出缓存与代理channel一致，但原始指标需要的目标条件化和归因范围不同，不能将上述设计称为目标分辨率完全匹配的纯定义比较。尤其CD的outgoing分配和各pair prior均广播到目标；它们没有Ξ与EI的target-specific惩罚。原生定义、方向、归因与这些限制见[文献扩展报告](unicm_literature_prior_expansion.md)。

对某个target–lead单元，令源attribution为 $c_m$、源均值为 $\bar c$：

$$
\epsilon=\max(0.05\bar c,10^{-12}),\qquad
r_m=\frac{c_m+\epsilon}{\bar c+\epsilon},\qquad
p_m=\frac{r_m^{-\gamma}}{\frac1{11}\sum_{n=1}^{11}r_n^{-\gamma}}.
\tag{2.1}
$$

同一源的四个特征共用惩罚 $\alpha p_m$。α控制总体收缩，γ控制不同源惩罚差异；floor_fraction固定，不另搜索。该权重只决定拟合系数的收缩，最终预测由拟合系数生成。归一化抹去共同尺度，所以总信息量较大本身不保证预测更好。

Univariate仍只是目标自身冻结预测的截距/斜率校准，并非Unique EI加权。已移除的 MIM 在当前干预分布下就是单源 EI；该等同关系不表示 singleton EI 惩罚与全联盟 EI-Shapley 惩罚相同。互信息的数学定义仍成立，但这里不保留其作为普通观测 MI 的竞争身份。

## 3. ΦR、WMS与Uniform此前相等的诊断

历史允许γ=0时，ΦR、WMS、Uniform均选择α=1000、γ=0；全部25,344个预测逐项相同，nRMSE均为1.012753597030。式（2.1）在γ=0下给出均匀惩罚，所以相等来自关闭加权，不能解释为指标相同。该选项已从当前信息方法的选择范围排除。

当前固定γ=3后，ΦR=1.086247161、WMS=1.086161116、Uniform=1.115712360，预测均不同。ΦR的MMI双重冗余修正确实算入边权；ΦR/WMS归一化源份额相关0.998446、惩罚相关0.998865，因而效果接近。ΦR加回冗余的理论意义不等于监督预测优化保证。完整公式、历史选参表及逐项诊断保存在[五项历史报告](../log/unicm_information_prior_comparison_five_methods_historical.md)。

新增正向SI同样没有误接成WMS：3960条边满足SI=WMS+输出TC，最大误差1.11e-15 bit。相关残差可使SI增加，不能将SI当成纯协同。

## 4. 验证及主图

复用三checkpoint各16384个独立bounded-uniform完整12月历史、bound4、January initialization、sampling seed20260901的原干预预测缓存。仅重建72个小型affine读出，使用同一degree-1 Gaussian TM、covariance ridge1e-6。此前新增 SI 与 CD 的补算没有重新调用或训练 Transformer；本次移除 MIM 仅复用已有评估与 bootstrap 数组重绘。

沿用253 fit、36 validation、96 test起报以及时间空档；上游归一化拟合期1980-01—2003-12，44维下游标准化和nRMSE分母只取fit。bootstrap seed20261001、12月块、4000次共享重采样。六项信息prior与Uniform的校准容量相同；Frozen和Univariate仅为既有参考。

15项解析检查通过，新增独立复制、交叉传递、SI相关残差、CD与Gaussian Schur-complement条件MI核对。非负容差1e-8 bit，容差内归零数和显著负值数均为0；Shapley闭合最大1.78e-15 bit，半边闭合最大7.11e-15 bit。旧八项attribution与原缓存逐项相同，当前保留方法及参考的25个可对应评估数组、包括预测和bootstrap数组均逐项复现。

Earth主图i已同步为图1A的九行结果；底部高度比例由2.0增至2.4、画布高由11.6增至12.0英寸，底部i/j宽度仍为1.3:1。整体图已检查，标签/数值分离，热图与树保留原数据，j仍是原Ξ打乱对照，两个主图PNG副本逐字节相同。

![更新后的Earth主图](assets/unicm_main_with_hypergraph.png)

## 5. 复现与证据

- [指标定义、文献与候选筛选](unicm_literature_prior_expansion.md)
- [本轮实验约定](../log/unicm_literature_prior_expansion_contract.md)
- [搜索查询与失败URL记录](../log/unicm_literature_search_provenance_20261002.json)
- [当前摘要](../../results/unicm_information_prior_comparison_literature_shared_xi_n16384/summary.json)、[评估数组](../../results/unicm_information_prior_comparison_literature_shared_xi_n16384/evaluation_arrays.npz)、[attribution缓存](../../results/unicm_information_prior_comparison_literature_shared_xi_n16384/information_centralities.npz)、[复现及绘图审计](../../results/unicm_information_prior_comparison_literature_shared_xi_n16384/literature_expansion_verification.json)
- [五项历史结果与重合诊断](../log/unicm_information_prior_comparison_five_methods_historical.md)、[此前O/WMS/PED可行性记录](unicm_oinfo_wms_ped_feasibility.md)
- [实验入口](../../scripts/run_unicm_information_prior_comparison.py)和[解析测试](../../tests/test_unicm_information_prior_comparison.py)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python scripts/run_unicm_information_prior_comparison.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python scripts/compose_unicm_main_hypergraph.py
```

默认固定Ξ参数；再次运行复用 attribution 缓存，仅拟合当前六项先验。历史 MIM 的 attribution、预测及 bootstrap 数组作为内部追溯缓存保留，不进入当前方法表或图表。当前缓存中 MIM 与 EI-Shapley 减 Ξ-Shapley 的最大差为 `6.66e-16` bit；保留方法的摘要数值、预测与 bootstrap 数组未变。显式独立选参入口仍保留，仅允许正γ网格，未在本轮使用。
