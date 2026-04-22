# 空气质量多城市多步长高阶因果图搜索设计

## 1. 目标

本设计面向一轮新的空气质量实验搜索，目标是在 `上海 / 南京 / 杭州 / 北京` 与 `1h / 3h / 6h / 12h / 24h` 这些候选设置中，找出既能保持 `O3` 多步预测精度，又能产生更清晰、可解释、且高阶协同更强的因果图配置。

这轮搜索的主目标是：

- 在测试集预测质量过关的前提下，最大化跨站 `O3 + PM2.5 -> O3` 的 `Syn`
- 同时压低 `Syn` 的负边比例与负值质量
- 额外补充 `PM2.5 -> O3` 的 pairwise 因果分析，作为次级解释证据

这里不把 “`Syn / pairwise` 越接近 `1` 越好” 作为目标。正确口径是：

- `Syn` 的绝对值越大越好
- `Syn / pairwise` 只作为辅助诊断，用来排除 “`Syn` 虽然和 pairwise 同量级，但整体非常小” 的假阳性配置

## 2. 范围

### 2.1 需要覆盖的实验轴

- 城市：
  - `shanghai`
  - `nanjing`
  - `hangzhou`
  - `beijing`
- 预测步长：
  - `1h`
  - `3h`
  - `6h`
  - `12h`
  - `24h`
- 输入变量：
  - 固定为 `10` 个变量
- 预测输出：
  - 固定联合预测所有站点的 `O3` 与 `PM2.5`

### 2.2 本轮不做的事

- 不把搜索主循环继续塞进 `exp/yrd_shanghai_tm_graph.ipynb`
- 不优先扩展更多城市或更多目标变量组合
- 不把 `PM2.5` 作为主排序目标
- 不把 `TM` 的上海旧 profile 直接复用为默认全局 profile

其中 `shanghai` 在本轮中的定位是：

- 把已有 `1h` 结果作为 anchor baseline
- 重点补试 `3h / 6h / 12h / 24h`
- 同时保留与其他城市统一比较的能力

## 3. 现有代码基线

仓库已有三块可以直接复用的底层能力：

- `yrd/data.py`
  - 已支持按 `city_en` 选站点
  - `build_windowed_samples` 已支持多 `horizons`
- `yrd/train.py`
  - 已支持联合训练多输出模型
  - `train_joint_model_with_history` 可直接作为统一训练入口
- `yrd/coupling.py`
  - 已有 `NIS` 与 `TM` 的 EI / synergy 汇总逻辑

当前真正写死的是 `yrd/shanghai_notebook.py` 及 `exp/yrd_shanghai_tm_graph.ipynb`：

- 逻辑偏向 “上海 + one-step + O3 图”
- 缓存和图产物命名也偏向单 notebook 工作流

因此，本设计不重写模型本体，而是新增一个面向 `城市 x horizon` 搜索的实验驱动层。

## 4. 总体策略

推荐采用 `NIS 粗筛 + TM 精排` 的两阶段策略。

### 4.1 Stage A: 粗筛

对所有 `城市 x horizon` 组合执行：

1. 训练或复用该组合的联合预测模型
2. 记录测试集 `O3 / PM2.5` 预测指标
3. 用较便宜的 `NIS` 计算：
   - `O3 + PM2.5 -> O3` 的 synergy 指标
   - `PM2.5 -> O3` 的 pairwise 指标
   - 负边比例与负值量级统计
4. 生成 coarse leaderboard

这一阶段的作用是快速排除：

- 预测精度明显不够的组合
- `Syn` 极小或负边过多的组合

### 4.2 Stage B: 精排

只对 Stage A 的 shortlist 组合执行：

1. 固定较好的模型权重
2. 重新估计该配置自己的 `TM` 干预采样盒宽 `L_v`
3. 用多 seed、多 `M` 的 `TM` 采样估计因果边
4. 汇总 seed 聚合后的稳定性与强度指标
5. 生成正式图和最终 leaderboard

这一阶段的作用是确认：

- 高 `Syn` 是否在 `TM` 下仍成立
- 负边是否足够少
- 最终图结构是否清晰、可解释

## 5. 新增驱动层设计

### 5.1 新增模块

建议新增一个通用实验驱动模块，例如：

- `yrd/air_search.py`

它负责：

- 统一构造 `dataset / city / horizon / target` 配置
- 驱动训练、评估、NIS、TM、缓存写入
- 输出标准化结果表

### 5.2 新增 CLI

建议新增：

- `scripts/run_air_search.py`

建议支持参数：

- `--cities nanjing,hangzhou,beijing`
- `--horizons 1,3,6,12,24`
- `--stage coarse|refine|report`
- `--force-retrain`
- `--force-recompute-coupling`
- `--top-k`
- `--tm-sample-counts`
- `--tm-seeds`
- `--tm-gammas`

### 5.3 notebook 角色调整

notebook 不再承担主搜索循环，而只用于：

- 读取最终 shortlist 的 cache
- 出正式图
- 为 `docs/研究框架.md` 提供可引用图件与结论文字

## 6. `TM` 盒宽 `L_v` 的重新选定

这一轮 `TM` 不能继续直接套用上海旧的 `train-q99` profile。应改成：

- 对每个 shortlist 配置单独重算 `L_v`
- 采样中心仍采用该配置 `train input mean`
- `L_v` 必须至少覆盖 train 支持域
- 允许比覆盖 train 支持域的最小盒宽略大一点

### 6.1 基础统计

对每个变量 `v` 计算：

- `train_min_v`
- `train_max_v`
- `center_v`
- `L_cover_v = max(center_v - train_min_v, train_max_v - center_v)`

### 6.2 最终盒宽

定义：

- `L_v = gamma * L_cover_v`

其中：

- `gamma = 1.00` 表示刚好覆盖 train 支持域
- `gamma > 1.00` 表示在此基础上适度外扩

推荐在 refine 阶段只扫一个小集合：

- `gamma in {1.00, 1.10, 1.20}`

排序逻辑不是单纯挑最大的 `gamma`，而是：

- 优先选择能覆盖 train 支持域的最小 `gamma`
- 如果该 `gamma` 下边符号仍然不稳，再升到更大的 `gamma`

### 6.3 非负变量裁剪

继续沿用非负变量的 `0` 下界裁剪。默认包含：

- `O3`
- `PM2.5`
- `t2m`
- `d2m`
- `sp`
- `tp`
- `blh`
- `msdwswrf`

### 6.4 必须持久化的 `L_v` 诊断

对每个 refine run，至少保存：

- `train_min/max/center`
- `L_cover_v`
- `gamma`
- `L_v`
- lower bound
- synthetic sample clipping ratio
- synthetic sample 超出 train 支持域的比例统计

## 7. 排名与筛选逻辑

最终不采用黑箱总分，而采用顺序筛选。

### 7.1 第一层：预测门槛

必须满足：

- `O3` 测试集显著优于 `PersistenceBaseline`

主要记录：

- `RMSE`
- `corr`

`PM2.5` 只要求不要明显崩掉，不作为主排序目标。

### 7.2 第二层：主目标

主排序指标是跨站 `O3 + PM2.5 -> O3` 的 `Syn`，并且：

- `Syn` 绝对值越大越好

建议同时记录：

- non-self `Syn mean`
- non-self positive mean
- non-self positive mass
- non-self median

### 7.3 第三层：稳定性约束

稳定性约束包括：

- `negative ratio`
- `negative mass ratio`
- seed 聚合后的 mixed-sign edge count

也就是：

- 先追求 `Syn` 大
- 再在高 `Syn` 组合中优先保留负值更少的配置

### 7.4 第四层：辅助量级诊断

`Syn / pairwise` 只作为辅助诊断，不作为主目标。

它的作用是排除：

- `Syn` 绝对值非常小，只是因为 pairwise 也很小，所以看起来“同量级”

因此，这一层只用于标注：

- `Syn` 是否明显小于主导 pairwise EI

### 7.5 第五层：次级解释证据

最后再看：

- `PM2.5 -> O3` 的 pairwise EI 图是否也足够清晰、有解释力

它可以用于：

- tie-breaker
- 文档叙述补充

但不压过 `Syn` 主目标。

## 8. 结果产物

### 8.1 cache 布局

建议使用分层路径：

- `exp/cache/yrd_coupling/air_search/<city>/<horizon>h/<run_tag>/`

每个 run 至少保存：

- checkpoint
- forecast metrics
- `NIS` summary
- `TM` summary
- `L_v` diagnostics
- edge tables
- graph assets

### 8.2 日志布局

延续 `continuous-tuning-agent` 的日志规范，统一写入：

- `docs/log/air_tuning/run_history.jsonl`
- `docs/log/air_tuning/leaderboard.md`
- `docs/log/air_tuning/notes.md`
- `docs/log/air_tuning/next_steps.md`
- `docs/log/air_tuning/tuning_report.md`

### 8.3 正式图

最终 shortlist 至少导出三类图：

- `O3 -> O3` pairwise graph
- `PM2.5 -> O3` pairwise graph
- `O3 + PM2.5 -> O3` synergy graph

按仓库规范，正式引用资产优先：

- `png`
- `pdf`

## 9. 测试与验证

### 9.1 单元测试

至少补三类测试：

- `L_v` 生成器测试
  - 覆盖 train 支持域
  - 正确处理非负裁剪
- `PM2.5 -> O3` pairwise summary 测试
  - 确认新分析链路与现有 `O3 -> O3` / synergy 链路不混淆
- batch driver 配置测试
  - 确认城市、dataset、horizon 路由正确

### 9.2 smoke 验证

至少跑一个非上海城市和一个多步长配置，验证：

- 训练
- 缓存
- `NIS`
- 日志写入

都能走通。

## 10. 风险

主要风险有：

- `TM` 采样成本高，尤其是多 seed、多 `M` 时
- 长 horizon 预测更难，可能导致 coarse 阶段大量组合直接淘汰
- 不同城市站点数不同，会影响边数规模和图解释方式
- `Syn` 放大后若负边依然很多，说明 estimator 稳定性仍是瓶颈

## 11. 执行顺序

推荐执行顺序：

1. 新增 `air_search` 驱动层与 CLI
2. 实现城市 / dataset / horizon 路由
3. 实现 coarse 阶段的训练与 `NIS` 汇总
4. 实现新的 `L_v` 估计器与持久化诊断
5. 实现 refine 阶段的 `TM` 批量评估
6. 实现 `PM2.5 -> O3` pairwise 图与结果汇总
7. 补测试
8. 跑 smoke
9. 启动正式搜索
