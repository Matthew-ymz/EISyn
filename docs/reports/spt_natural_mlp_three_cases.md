# 三案例：共同自然数据 MLP 后的 SPT F1

2026-10-16。用户批准30分钟总上限，仅补算Kuramoto A、HO-LV A、Kuramoto B的自然MLP和SPT，不重跑其他基线，不调参或按F1选择模型。

任务状态：`complete`；固定预算起点后累计1479.0秒。

| 案例 | 旧非MLP模块F1 | 自然MLP后模块F1 | 数值重复范围 | 状态 |
|---|---:|---:|---|---|
| Kuramoto A | 1.0000 | 0.0000* | [0.0, 0.15384615384615383]（4/4可评分） | numerically_unresolved |
| HO-LV A | 1.0000 | 0.0000* | [0.0, 0.0]（2/4可评分） | numerically_unresolved |
| Kuramoto B | 1.0000 | 0.0000* | [0.0, 0.0]（4/4可评分） | numerically_unresolved |

以上新分数来自完整高档均值EI表，而非复用旧图。星号数值仍受积分档位变化和/或树稳定性限制；本轮只能判断是否重现旧图表现，不能把诊断值当作已充分收敛的真实机制恢复率。

* 表示完整均值EI表通过Syn审计并生成的诊断树，但积分精度或树稳定性尚未通过；不能作为已证实的可靠F1。显著Syn非负违规的重复不构树、不补分，重复范围仅包含可评分的重复；均值EI仍使用全部四份完整表，不按审计或F1选择重复。旧1.0来自字典模型，不是本次MLP结果。

## 固定比较口径

- 使用三个案例各自原WMS/SHAP自然轨迹池。每例按同一固定链索引48/8/8划分训练/验证/测试，24576/4096/4096相邻状态对，无跨集合共享状态。Kuramoto B仍映射原缓存键C。
- Kuramoto A复用已批准试算的冻结MLP；其余采用同样两层64单元tanh和AdamW单一设置，验证集仅选轮次。Kuramoto用sin/cos输入和圆周损失；LV用训练统计量标准化的实数坐标和MSE。这些状态域适配分别声明。
- 自然训练后冻结MLP与训练残差条件密度；Kuramoto为独立恒定von Mises，LV为独立恒定Gaussian条件TM，无噪声下限。后续干预均为原生坐标独立Uniform(-pi,pi)^9，target固定完整未来九坐标，时距0.2，信息单位nats。
- 固定N128/P64及N512/P128两档，每档四scramble；同一联合下查询全部511非空源子集，审计全部9330二分，再执行标准逐节点精确SPT至单节点。原生Syn容差0.01 nats，容差内负值数量显式记录；显著负值失败，无静默裁剪。
- 评分沿用旧图的耦合模块参考，忽略primitive模块内部任意二叉细化，并保留严格clade F1。分数不等于全部物理超边恢复率。每例仅一个自然数据集，四scramble仅为数值重复。
- 0.01 nats的EI SEM/两档变化、ESS≥10、最大权重<.25、后验剩余步<1e-3及四次评分树一致共同决定数值标签。未过检查可展示完整合法均值树的诊断F1，但不能据此宣布可靠恢复。

## 各例审计

### Kuramoto A

状态：`numerically_unresolved`。
训练轮次148，选中轮次123，模型摘要`7fefa8a3116f957e4d715ed63175a9272b281c3d4f15b151b418421d7b257ae9`。自然测试RMSE=0.033590（本例原生坐标单位）。
模块F1=0.0000；严格clade F1=0.0000；根切分=[[1, 2, 6, 7, 8], [3, 4, 5, 9]]；评分树一致0/4。
未通过检查：`['maximum_grid_change>=.01_nats', 'scored_module_tree_varies_across_scrambles']`。
- N128/P64：均值树`exploratory_only`；最大EI SEM=0.0178946 nats；min ESS=42.4；max weight=0.04236。
  该档模块F1=0.0；raw Syn最小估计值=-0.0035327674335459136 nats；阈值=-0.01 nats；容差内负值数=6022；显著违规数=0。
  全源raw Ξ估计=-0.002762648 nats；容差内负估计仅作数值零处理，不是负Syn的证据。
- N512/P128：均值树`exploratory_only`；最大EI SEM=0.00541461 nats；min ESS=86.02；max weight=0.02175。
  该档模块F1=0.0；raw Syn最小估计值=-0.0025468183852996873 nats；阈值=-0.01 nats；容差内负值数=5745；显著违规数=0。
  全源raw Ξ估计=-0.0019218999 nats；容差内负估计仅作数值零处理，不是负Syn的证据。

### HO-LV A

状态：`numerically_unresolved`。
训练轮次67，选中轮次42，模型摘要`75ef68b56bbd241c161248d165eafc0ea3443d515bb5f915324eec16439c344f`。自然测试RMSE=0.032267（本例原生坐标单位）。
模块F1=0.0000；严格clade F1=0.0000；根切分=[[1, 4, 6], [2, 3, 5, 7, 8, 9]]；评分树一致0/4。
未通过检查：`['one_or_more_numerical_repeat_Syn_audits_failed', 'maximum_grid_change>=.01_nats', 'scored_module_tree_varies_across_scrambles']`。
- N128/P64：均值树`exploratory_only`；最大EI SEM=0.0200201 nats；min ESS=41.74；max weight=0.05361。
  该档模块F1=0.0；raw Syn最小估计值=-0.006493639878678792 nats；阈值=-0.01 nats；容差内负值数=2050；显著违规数=0。
  全源raw Ξ估计=0.0082958564 nats；容差内负估计仅作数值零处理，不是负Syn的证据。
- N512/P128：均值树`exploratory_only`；最大EI SEM=0.00736586 nats；min ESS=65.53；max weight=0.03495。
  该档模块F1=0.0；raw Syn最小估计值=-0.003233190735434377 nats；阈值=-0.01 nats；容差内负值数=2016；显著违规数=0。
  全源raw Ξ估计=0.0038655813 nats；容差内负估计仅作数值零处理，不是负Syn的证据。

### Kuramoto B

状态：`numerically_unresolved`。
训练轮次116，选中轮次91，模型摘要`c75a5cef4c4de1a0a55ba0bb08bf47f1cd1d59fb5c924db3b66130ccc43012fc`。自然测试RMSE=0.032902（本例原生坐标单位）。
模块F1=0.0000；严格clade F1=0.0000；根切分=[[1, 3, 6, 7], [2, 4, 5, 8, 9]]；评分树一致0/4。
未通过检查：`['maximum_grid_change>=.01_nats', 'scored_module_tree_varies_across_scrambles']`。
- N128/P64：均值树`exploratory_only`；最大EI SEM=0.018036 nats；min ESS=42.39；max weight=0.04238。
  该档模块F1=0.0；raw Syn最小估计值=-0.004773590472305855 nats；阈值=-0.01 nats；容差内负值数=6159；显著违规数=0。
  全源raw Ξ估计=-0.0040597014 nats；容差内负估计仅作数值零处理，不是负Syn的证据。
- N512/P128：均值树`exploratory_only`；最大EI SEM=0.00473394 nats；min ESS=86；max weight=0.02176。
  该档模块F1=0.0；raw Syn最小估计值=-0.0012452393179902543 nats；阈值=-0.01 nats；容差内负值数=4964；显著违规数=0。
  全源raw Ξ估计=-0.00032749917 nats；容差内负估计仅作数值零处理，不是负Syn的证据。

## 解释边界与复现

自然数据上的低预测误差不能保证均匀干预域可信。此前Kuramoto A试算已显示明显干预外推误差，因此即使新F1高，也只说明该冻结MLP诱导联合的模块恢复，不能自动证明学到了真实机制。若F1下降，不能直接断言是SPT算法失效；模型表达、自然数据覆盖与积分近似均可能参与，本轮未做消融。

本次重新通过Zotero核对父项P6UJCVG8题名 *Emergent hierarchical organization of causal interactions in complex systems*，列举唯一主文DXGC7JEA（19/19页）及SI MWIWKSVG（28/28页），读取Methods(10)–(12)、SI S1.2、S5.1、S7。附件无明确稿件日期/版本，元数据日期不构成新稿证据。一般条件TM和模型介导信息积分是明确的估计器适配，区别于SI S1.2仿射/Wishart路径；本例完整未来坐标亦不同于SI S7增量target。

`scripts/run_spt_natural_mlp_three_cases.py`；内部预算、协议、共用自然划分、模型、全EI及审计缓存位于`results/spt_natural_mlp_three_cases/`，全部模型摘要隔离，未复用旧字典模型EI。父进程按持久化绝对截止终止自己的工作进程组，重启不重置预算。

执行修复记录：依据实测吞吐将案例积分改为同一进程内三案例并行，科学设置和绝对截止均未改变。首次并行出现SciPy Sobol方向表初始化线程冲突，已停止该进程并隔离其新高档结果；仅保留冲突前经摘要核验的正确积分前缀。随后串行化采样器构造、保留积分并行，串并行采样逐点一致测试通过。两次修复日志、被排除缓存与原始正确前缀摘要均保留；报告采用的是修复后结果。

[原MLP试算](spt_natural_mlp_pilot.md) · [旧字典模型比较](spt_matched_three_case_comparison.md)。

最终核验：25项软件测试通过；14份正确缓存前缀逐值摘要一致；三个自然池与原网络匹配；科学核心文件未改变，模型摘要与EI缓存一致；短期监督已结束。
