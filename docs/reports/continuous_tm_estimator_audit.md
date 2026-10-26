# 连续 EI：稿件与实际估计器核对

核对日期：2026-10-26。范围为 Overleaf 附录 S1.2、正文图 1c 的生成与计算路径，以及保留在仓库中的特征升维接口。本次只修改方法描述，未改变估计器、训练模型或重新生成科学实验结果。

## 结论

原稿 S12–S15 描述的是“非对称特征升维后分别作 affine/Gaussian 信息估计”。该写法确实存在结构性问题，但当前图 1c 对应的计算脚本并未调用这组特征升维。实际使用的是原始坐标上的三次多项式三角 TM。不能由原稿的缺陷直接推断图 1c 使用了同一个错误估计器。

三项密度分别拟合所带来的估计误差仍需与总体非负性定理区分。本次文稿修订不等于完成了该数值估计器的普遍非负性证明。

## 原稿中具体的不一致

原稿 S12 为

\[
\boldsymbol\phi_s(x)=(x,x^2,x^3)^{\mathsf T},\qquad
\boldsymbol\phi_j(x_1,x_2)=(x_1,x_2,x_1x_2,x_1^2,x_2^2)^{\mathsf T}.
\]

S15 将三种特征下各自的 Gaussian/affine 近似 MI 相减。联合源的近似特征空间没有单源空间中的三次项；“保留原坐标”不能使高斯近似继承精确 MI 对可逆特征表示的不变性。相同样本、相同估计器名称和维度修正也不能消除这种不兼容。

具体反例：令独立标准高斯变量满足 \(Y=X_1^3+\varepsilon\)。真实 PEID Syn 为零。上述特征的总体 Gaussian 近似给出

\[
\widehat I_j=\tfrac12\log(16/7),\quad
\widehat I_1=\tfrac12\log16,\quad
\widehat I_2=0,
\quad\widehat{\mathrm{Syn}}=-\tfrac12\log7=-0.972955\ldots\ \mathrm{nats}.
\]

这里 \(\operatorname{Var}(Y)=16\)，联合特征只能通过 \(X_1\) 解释 9 的方差，残差为 7；单源特征包含 \(X_1^3\)，残差为 1。该差异在样本量趋于无穷时仍存在。此诊断声明非负容差为 \(10^{-10}\) nats，显著违规数为 1，判定失败，未作任何 Syn 裁剪。

## 实际调用链

1. [图 1 组图脚本](../../scripts/compose_part1_figure1.py:252)读取 `sine_beta_intervention_sample_robustness.json` 的 `updated_full_result`，要求每条件 5,120 个干预样本。
2. [计算脚本](../../scripts/run_sine_beta_intervention_sample_robustness.py:234)把原始 `x`、`y` 及 MLP 的 `z` 输出传入 `summarize_two_source_synergy_transport_map`。
3. [两源汇总函数](../../yrd/__init__.py:205)以 1、1、2 列原始源坐标分别进行三项查询，没有调用 `lift_transport_source_features`。
4. `yrd/__init__.py` 末尾的[兼容导入](../../yrd/__init__.py:6153)覆盖了文件前部的旧 affine 实现。运行时实际指向 [exp/TM/transport_map_density.py](../../exp/TM/transport_map_density.py:311)，默认 `degree=3`、`joint_order="target_first"`。
5. 该实现用三次多项式条件均值、Gaussian 条件残差构造三角密度。三个块密度分别拟合；没有减去原稿 S14 的 Wishart log-det 偏差项，返回的 `bias_correction` 为零。
6. MI 接口返回 bits。图 1 准备阶段乘以 \(\ln2\) 后展示 nats。两源汇总函数先对各项 EI 作 `max(0, EI)`，随后保留 Syn 差值的符号；没有对 Syn 作 `max(0, Syn)`。

运行时输入记录为 `(12,1)`、`(12,1)`、`(12,2)`，确认该汇总路径未把源扩展为 3、3、5 个特征。这里只核对了调用路径，没有训练模型。历史提交 `4c6aa767` 中的汇总函数也采用原始坐标。

在线源码快照中的图 1 PNG 与本地 `paper_assets/figure1_peid_spt_main.png` SHA-256 一致：`f6648947a39195d6838d1e3c968e2215b1493db32a9eea74f347b8acc93d2cde`。缓存未保存完整的估计器代码哈希，不能把当前运行时核对描述为历史执行环境的逐项复现。

## 哪些代码仍保留该特征构造

- [lift_transport_source_features](../../yrd/__init__.py:189)的 3/5 维映射与原稿 S12 一致，但图 1c 的汇总函数没有使用它。其他工具或 notebook 对它仍有调用，本次未改动这些路径。
- [正弦频率探索脚本](../../scripts/run_sine_frequency_mlp_peid.py:116)确实分别升维单源与联合源；联合源还增加所选谐波的正弦、余弦特征，并使用 affine log-det 与偏差修正。这不是图 1c 的当前计算路径，其近似一致性需要单独验证，不能套用本次对原始坐标 TM 的描述。
- UniCM 的[联盟 EI 表](../../scripts/analyze_unicm_11mode_shapley.py:181)和交通 SPT 的[共同 affine 通道](../../scripts/analyze_nyc_taxi_mgstn_spt.py:302)属于另一类高维读出，不应由 S12 的五维特征式统一概括。本次没有完成全部高维估计假设的复核。

## Overleaf 已修改的位置

| 位置 | 原描述的问题 | 本次处理 |
|---|---|---|
| S1.2，共同设置段 | 把相同样本与估计家族写成估计一致性的保证 | 区分共同干预定义与有限密度近似的数值性质 |
| S11 前后的 affine 段 | 把低维实验概括为 affine/Gaussian | 保留 affine 密度式，明确其适用分支及高维协议的位置 |
| S12 | 旧的 3/5 维源特征映射不对应图 1c | 改为原始坐标上的三次多项式条件均值三角 TM，说明 target-first、ridge 和残差尺度 |
| 原 S13–S14 | 把 Wishart 偏差修正写为实际多项式 TM 的步骤 | S13 改为实际 log-density ratio；现有单项 EI 的非负投影仅作行内说明 |
| 原 S15 及后文 | 以旧特征式定义差值，并声称同样设置足以保证一致 | 现 S14 改为原始源集合的三项 EI 差值，保留符号，简述独立密度拟合的近似误差 |

按用户追加要求，将单项 EI 处理式改为行内公式，不再单独编号；S1.2 现有四个编号公式 S11–S14，S1.3 首式顺延为 S15。新 S12 标签为 `eq:polynomial-transport-map`，三项差值保留标签 `eq:transport-map-two-source-synergy`；原标签的活动引用已核对。旧文完整保留为 LaTeX 注释，新文标红。

同时删除稿件中的 bits/nats 换算说明和 84 个估计的审计数值，将该段的 `readout` 改为具体的方法表述。单位、容差和缓存核对结果保留在本报告中，供实现一致性检查使用。

图 1c 现有 84 个 seed–parameter Syn 记录（21 个参数值 × 4 个种子）的展示审计使用零 nats 容差，最小值为 0.3745771944 nats，负值数为 0，数值零区间内的值数为 0。这里审计的是保留的 Syn 记录，不能从中恢复单项 EI 投影前的原始估计及投影次数。

## 稿件来源与核验范围

本轮重新通过 Zotero 核对父项 `R68U4QXB` 的题名 *Revealing hierarchies of synergistic causal interactions in complex systems*，重新取得正文 `GRX5IJJQ`（19/19 页）与附录 `UTEBTIWV`（34/34 页）全文，读取正文连续 EI 与方法定位、附录 S1.2／旧 S11–S15。附件仍未给出明确的稿件版本或日期，元数据日期不用于判定版本先后。

已直接重读在线 `supplementary.tex` 的 S12–S15 和共同设置段；其修订前内容与上述旧式一致。图文件与缓存对照使用 2026-10-26 18:53 导出的在线源码快照。这里只作估计器描述一致性核对，未声称完成全文方法有效性审计。

最终在线附录编译：Errors 0，Warnings 1；唯一编译警告为已有的 `Command \showhyphens has changed.`，未出现未定义引用或文献警告。已视觉核对附录第 4 页：单项 EI 式为行内公式，差值式编号为 S14，下一节首式为 S15；新增内容标红，下一节恢复黑色，未见公式溢出。Current file Review 仍显示 “No comments or suggestions”。[核验截图](../../output/continuous_tm_description/overleaf_s1_2.jpg)。
