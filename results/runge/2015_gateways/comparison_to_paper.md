# Runge 2015 Reproduction Comparison

Source paper: Runge et al., *Identifying causal gateways and mediators in complex spatio-temporal systems*, Nature Communications 6:8502 (2015).

## Paper Targets

- Climate experiment uses 60 Varimax-rotated SLP components and 3,339 weekly observations.
- Main climate-network settings: `tau_max = 4` weeks, PC significance level `alpha = 0.001`, and about 20% aggregated cross-link density.
- The paper uses 0-based component labels. This document follows that convention and writes them as `No.k`.
- The local orthomax implementation recovers the paper-discussed modes with a small permutation: internal `No.7`, `No.8`, and `No.21` are reported as paper components `No.18`, `No.26`, and `No.48`, respectively.
- The mapping is calibrated from the published Fig. 2/Fig. 4 spatial locations for the paper-discussed components and kept bijective by inverse swaps.
- The paper highlights `No.0`, `No.1`, `No.2`, and `No.18` as major causal gateways and dominant mediators; it also discusses `No.26` and `No.48` among important mediators in robustness analyses.

## This Run

- Daily SLP files: 64 files, 1948-2011.
- Components: 60.
- Weekly samples: 3339.
- `tau_max`: 4.
- `alpha`: 0.001.
- Link-density target: 0.2.
- Tigramite backend: parent selection with ParCorr, followed by sparse standardized OLS causal regression.
- Lagged edges after coefficient thresholding: 837.
- Aggregated cross source-target pairs: 692 / 3540 = 0.1955.
- Linear coefficient comparison matrix: 60 x 60 = 3600 elements.

## Ranking Overlap

Paper-highlighted components: `No.0`, `No.1`, `No.2`, `No.18`.

Gateway ranking overlap:

| paper_component | Reproduction gateway rank |
|---:|---:|
| No.0 | 3 |
| No.1 | 2 |
| No.2 | 1 |
| No.18 | 9 |

Mediator ranking overlap:

| paper_component | Reproduction mediator rank |
|---:|---:|
| No.0 | 3 |
| No.1 | 2 |
| No.2 | 1 |
| No.18 | 9 |

Top paper-calibrated gateway components in this run: `No.2`, `No.1`, `No.0`, `No.3`, `No.6`, `No.4`, `No.48`, `No.14`, `No.18`, `No.11`, `No.31`, `No.20`, `No.24`, `No.10`, `No.37`.

Top paper-calibrated mediator components in this run: `No.2`, `No.1`, `No.0`, `No.48`, `No.26`, `No.3`, `No.11`, `No.14`, `No.18`, `No.6`, `No.22`, `No.20`, `No.24`, `No.19`, `No.4`.

Diagnostic check for the paper's additional mediator examples:

| paper_component | local component | AMCE rank | AMCE | mediated fraction |
|---:|---:|---:|---:|---:|
| No.26 | No.8 | 5 | 0.001483 | 0.806546 |
| No.48 | No.21 | 4 | 0.001561 | 0.930742 |

The paper text describes Nos. 26 and 48 as dominant mediators with more than 80% path participation and AMCE around 0.0015-0.002. After the Fig. 2/Fig. 4 visual calibration, both components satisfy that diagnostic in the current output.

## EI Matrix Comparison

The current pairwise TM-EI run compares `results/runge_pairwise_mlp_tm_ei/pairwise_ei_matrix.csv` against `results/runge2015_gateways/linear_coefficient_matrix.csv`.

- EI matrix shape: 60 x 60.
- Linear coefficient matrix shape: 60 x 60.
- Per-element comparison rows: 3600.
- Pearson correlation between EI and absolute linear coefficients: 0.6514.
- Spearman correlation between EI and absolute linear coefficients: 0.3441.
- Off-diagonal Pearson correlation: 0.4769.
- Off-diagonal Spearman correlation: 0.3026.
- Support match fraction: 0.3408 overall and 0.3297 off diagonal.

## Assessment

The reproduction follows the paper-level workflow: seasonal mean and variance removal, detrending, monthly component extraction with daily projection, Varimax component ordering by the rotated covariance diagonal, weekly aggregation, Tigramite parent selection, sparse standardized OLS causal regression, 20% aggregated cross-link thresholding, lag-resolved causal-effect recursion, and averaged ACE/ACS/AMCE.

The main structural checks are aligned: 60 components, 3,339 weekly samples, `tau_max=4`, `alpha=0.001`, and approximately 20% directed cross-pair density. The current label calibration recovers `No.0`, `No.1`, and `No.2` as the three strongest gateways, places `No.18` in the high-gateway group, and recovers `No.26` and `No.48` as dominant mediators. Since the paper does not provide a machine-readable 60-component loading table in this repository, labels outside the paper-discussed components should still be treated as reproducible orthomax labels rather than an official complete correspondence table.
