本文档用于检查 `Part1.md` 文件里的所有实验。

检查以下几点确保实验的公平性：

1. 同一 panel 内，横轴参数在零点时，从生成数据到计算流程都和其他参数点一致。
2. 同一 panel 内，MLP 训练、WMS、SURD 以及其他直接读取观测数据的方法必须使用同一套数据口径。默认优先使用同一条自然轨迹及其对齐的 source-target 样本，不得为某个方法单独更换状态分布。MLP+SHAP 和 MLP+PEID 必须复用同一个 fitted MLP。
3. PEID 在 MLP 训练完成后，可以独立均匀采样 intervention states，再将这些样本输入 fitted MLP 计算 PEID。这属于 PEID 方法内部的干预读出流程，不要求 intervention states 与 WMS、SURD 等观测方法使用的自然轨迹样本相同，也不应把 `readout_state_digest == peid_readout_state_digest` 作为普遍公平性条件。
4. 计算信息指标时，同一 panel 内要么都使用离散分箱方式估计，要么都使用 TM 方法估计。如果使用 TM，在估计概率密度和互信息时，映射函数的多项式阶数必须一致。

### 公平性审计

公平性审计必须按以下口径执行。旧版 `fairness_audit.passed` 若仍要求 PEID intervention states 与观测方法共享 readout states，则不能作为本口径下的通过证据，需要先更新审计条件再重新检查。

1. **零点流程一致**：同一 panel 和 seed 下，各横轴参数点使用参数匹配的 train/readout 输入状态池，即输入状态 digest 在参数扫描中保持一致；参数只进入真实映射和由此产生的目标。零点与正参数点调用相同的训练、SHAP、WMS、SURD 和 PEID 流程，图中零点直接报告估计 residual，且 `raw_*` 与展示值相同。
2. **观测数据与模型一致**：对每个参数和 seed，MLP 训练、WMS、SURD 及其他观测方法使用同一套自然轨迹数据口径和相同的 source-target 对齐方式。默认使用完全相同的自然轨迹样本；若预先注册 train/readout split，则所有观测方法必须遵守同一 split 规则，且不得为不同方法改变数据分布。MLP+SHAP 与 MLP+PEID 复用同一个 fitted MLP；`shap_mlp_model_digest == peid_mlp_model_digest`。
3. **PEID 干预采样独立**：MLP 拟合完成后，MLP+PEID 可以从注册干预域独立均匀采样 intervention states，并在该 fitted MLP 上读出 PEID。审计应检查干预域、采样方式、样本量和随机种子是否记录完整，但不要求 PEID intervention-state digest 与 WMS/SURD 的自然轨迹 digest 相同。Oracle PEID 若用于机制参照，应与 MLP+PEID 使用相同的干预协议。
4. **信息估计器一致**：同一 panel 的 WMS、SURD、MLP+PEID 与 Oracle PEID 使用相同类别的信息估计器；使用 TM 时统一多项式阶数，使用分箱时统一分箱规则。SHAP interaction、PCMCI 和 Neural Granger 等原生读数不是互信息量，不受此条约束，但必须明确标注其单位和语义。
5. **持续检查**：自动审计应检查零点流程、观测数据口径、source-target 对齐、共享 fitted MLP、PEID 干预协议、估计器一致性和零点原始值。不得仅因 PEID 使用独立均匀 intervention states 而将公平性判为失败。


统一协议如下：

- 同一 panel 中，MLP、WMS、SURD 和其他观测方法默认使用同一条自然轨迹及相同的 source-target 对齐样本。自然轨迹是观测方法公平比较的首选数据口径。
- 如果某个 panel 因实验设计必须使用 broad one-step samples，而不是自然轨迹，则 MLP、WMS、SURD 和其他观测方法必须共同使用这套 broad one-step 数据口径，并在报告中明确说明原因。不得让 MLP 使用 broad states、WMS/SURD 使用自然轨迹，或反向混用。
- MLP+PEID 的数据流程分为两个阶段：先使用与 WMS/SURD 同口径的数据训练 MLP；再从注册干预域独立均匀采样 intervention states，将其输入同一个 fitted MLP 计算 PEID。第二阶段样本属于 PEID 方法内部过程，不参与观测数据一致性比较。
- 在同一系统、参数和 seed 下，SHAP 与 MLP+PEID 使用同一个 fitted MLP；JSON 中记录 MLP digest，用来审计二者是否确实共享模型。JSON 还应分别记录观测数据 digest 和 PEID intervention-state digest，但两者无需相等。
- Oracle PEID 仅作为事后机制一致性诊断，并与 MLP+PEID 复用相同的干预域、采样 states 和目标噪声；Oracle PEID 不参与 MLP 训练数据与观测方法数据是否一致的判断。
- Standard Map、Wilson-Cowan refractory、Kuramoto、Ikeda y_tau 和 Nicholson-Bailey 的正式信息量数值使用三阶 transport map。Coupled Hénon 的 WMS、SURD、MLP+PEID 与 Oracle PEID 统一使用每变量 `6` 个等宽 bins；SHAP interaction 仍读取连续 MLP 响应。估计器一致性不意味着样本状态必须相同：PEID 仍可使用独立均匀 intervention states。
- 图中 MLP+PEID 直接报告配置 estimator 返回的 Syn，不再做 `max(0, Syn)` 截断。若估计值为小负数，则保留在图和 JSON 中，解释为有限样本、密度模型或 surrogate 误差诊断，而不是手动投影到非负轴。
- panel a 使用 `symlog` 纵轴，并在图内明确标注；这是为了同时保留 Standard Map 上 SURD 的极端退化估计和约 `0.03-0.18` bits 的 PEID 趋势，不改变任何原始数值。
- 对由扫描参数显式关闭的结构交互，主图仍显示同一套生成数据和同一 fitted MLP 经配置 estimator 得到的零点 residual；`raw_*` 字段保留为同值审计列。若 MLP residual 明显大于同 estimator 的 Oracle 零点 residual，则说明 surrogate 在 broad readout 上仍有形状误差，而不是说明真实机制存在协同。
