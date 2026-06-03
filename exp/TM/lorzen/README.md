# Lorzen TM-EI Reproduction

This folder contains the Lorzen transition data from the provided archive and the transport-map EI reproduction artifacts.

## Run

From the repository root:

```bash
python -m exp.TM.lorzen_tm_ei exp/TM/lorzen/input --output-dir exp/TM/lorzen/results --top-k 4
```

The script reads headerless `yt.csv` and `yt+1.csv`, estimates pairwise source-to-target EI with the repository transport-map backend, and compares the top-4 incoming sources per target against the Lorenz-96 groundtruth graph.

## Why the Direct Next-State EI Has a Strong Diagonal

The direct observed-data run uses `M_i(t+1)` as the target. In a continuous-time, one-step Lorenz trajectory, `M_i(t)` and `M_i(t+1)` are strongly autocorrelated, so pairwise EI assigns very large scores to the diagonal. This is a state-persistence effect, not necessarily a better causal-graph score.

For graph recovery, use the increment target:

```text
delta_i(t) = M_i(t+1) - M_i(t)
```

Then estimate EI under independent uniform interventions from a fitted quadratic transition surrogate. This makes `L` (`--box-width`) and the intervention `--sample-count` explicit.

Recommended tuned command:

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

## Current Results

- direct observed `next` target: accuracy 0.750, F1 0.750, AUC 0.897
- observed `delta` target: accuracy 0.750, F1 0.750, AUC 0.923
- tuned surrogate intervention: accuracy 1.000, F1 1.000, AUC 1.000
- raw samples: 1000
- tuned intervention samples: 2000
- variables: 8
- graph threshold: top-4 sources per target
- tuned `L` / `box_width`: 1.5

## Outputs

- `results/lorzen_tm_ei_matrix.csv`: target-by-source TM-EI matrix.
- `results/lorzen_tm_ei_edges.csv`: long-form edge table with raw MI and bias correction.
- `results/lorzen_tm_ei_topk_graph.csv`: thresholded TM graph.
- `results/lorzen_lorenz96_groundtruth.csv`: source-to-target Lorenz-96 reference graph.
- `results/lorzen_tm_ei_heatmap.png`: directly viewable comparison figure.
- `results/lorzen_tm_ei_summary.json`: manifest with metrics and artifact paths.
- `results_tuned/`: recommended delta-target surrogate-intervention result.
- `results_sweep/lorzen_tm_parameter_sweep.csv`: parameter scan over `L` and intervention sample count.
- `results_sweep/lorzen_tm_parameter_sweep.png`: parameter-scan diagnostic figure.
