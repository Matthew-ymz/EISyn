# 2m 气温 Runge 多 Horizon 实验说明

## 实验定位

本实验把 NCEP/NCAR Reanalysis 1 的 monthly near-surface 2m air temperature 接入 Runge-style 分量、MLP-TM-EI 和 PEID 流程。它用于扫描月尺度遥相关传播时间，不是原 Runge SLP 日资料到周尺度实验的直接复现。

默认数据为 `data/ncep_reanalysis_runge_validation/air.2m.mon.mean.nc`，变量为 `air`，单位为 K。主扫描的预测间隔为 `horizon=1..12` 个月。

## 处理流程

1. 对 monthly 2m 气温逐月去季节化，并按逐月标准差标准化。
2. 对每个格点沿时间轴去线性趋势。
3. 使用纬度面积权重、PCA 和 Varimax 得到 60 个 monthly component scores。
4. 用最近 4 个月 component states 预测未来第 `horizon` 个月 component state。
5. 每个 horizon 单独训练/缓存 MLP transition model，并计算 pairwise MLP-TM-EI、path-effect gateway/mediator 和二阶 PEID hypergraph。
6. 汇总 1 到 12 个月的预测性能、top gateway、top mediator、top PEID hyperedge 和 horizon profile 图。

## 推荐命令

快速前处理 smoke：

```bash
python scripts/run_runge_t2m_multihorizon.py \
  --input-netcdf data/ncep_reanalysis_runge_validation/air.2m.mon.mean.nc \
  --output-dir /tmp/eisyn_t2m_smoke_pre \
  --n-components 8 \
  --max-lag 2 \
  --pc-alpha 0.05 \
  --causal-backend regression \
  --horizons 1-2 \
  --skip-downstream
```

轻量下游 smoke：

```bash
python scripts/run_runge_t2m_multihorizon.py \
  --input-netcdf data/ncep_reanalysis_runge_validation/air.2m.mon.mean.nc \
  --output-dir /tmp/eisyn_t2m_smoke_downstream \
  --n-components 4 \
  --max-lag 2 \
  --pc-alpha 0.05 \
  --causal-backend regression \
  --horizons 1-2 \
  --lag 2 \
  --hidden-dim 8 \
  --epochs 1 \
  --intervention-samples 64 \
  --ensemble-ridge-alphas '' \
  --linear-blend-grid-steps 0 \
  --order-max 2 \
  --candidate-top-sources 3 \
  --candidate-target-topk 3 \
  --null-reps 1
```

主实验：

```bash
python scripts/run_runge_t2m_multihorizon.py \
  --input-netcdf data/ncep_reanalysis_runge_validation/air.2m.mon.mean.nc \
  --output-dir . \
  --n-components 60 \
  --max-lag 4 \
  --horizons 1-12 \
  --lag 4 \
  --hidden-dim 128 \
  --num-layers 1 \
  --dropout 0.5 \
  --epochs 120 \
  --intervention-samples 4096 \
  --ensemble-ridge-alphas 10,100,1000,3000 \
  --linear-blend-grid-steps 101 \
  --order-max 2 \
  --null-reps 20
```

长运行状态写入 `docs/log/live_status.md`，各 horizon 的 stdout/stderr 写入 `docs/log/logs/runge_t2m_hXX_pairwise.log` 和 `docs/log/logs/runge_t2m_hXX_peid.log`。

## 输出结构

- `results/runge_t2m_monthly/component_monthly_scores.csv`：monthly Varimax component scores。
- `results/runge_t2m_monthly/component_maps.npz`：component maps、解释方差和经纬度。
- `results/runge_t2m_monthly/horizon_01/` 到 `horizon_12/`：每个预测间隔的 pairwise 和 PEID 结果。
- `results/runge_t2m_monthly/horizon_summary.csv`：跨 horizon 汇总表。
- `results/runge_t2m_monthly/multihorizon_report.md`：自动生成的中文结果摘要。
- `fig/runge_t2m_monthly/horizon_profile.png`：预测误差、相关性、mean EI、top ACE/AMCE/PEID 随 horizon 的变化。

## 解释边界

- `horizon` 表示从最近输入月之后向未来第几个月预测，例如 `horizon=6` 是用最近 4 个月预测 6 个月后的 component state。
- 月尺度数据只能回答月到季节尺度传播问题，不能替代日资料上的周尺度传播分析。
- EI 和 PEID 是训练后预测映射在 bounded maximum-entropy intervention 下的信息读数，不等同于观测条件相关或真实物理干预强度。
