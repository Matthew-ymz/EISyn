# Runge PEID causal hypergraph summary

Higher-order PEID extension of the pairwise MLP-TM-EI gateway readout.
Delta_K is computed by Mobius inversion on the joint EI lattice.
This run is restricted to order 1 and order 2; no order-3 hyperedges are estimated.

## Run

- Components: 60
- Lagged samples: 3335
- Intervention samples: 4096
- Order max: 2
- Null reps: 20 (block size 26)
- MLP cache reused: True
- Self-check max |closed - recursive|: 8.674e-19 (5 samples)

## Hyperedge counts

- Order 1: 3600
- Order 2: 1630
- Order 3: 0

## Top hyper-gateways

| paper_component | hyper_ace_order1 | hyper_ace_order2 | hyper_ace_total |
| --- | ---: | ---: | ---: |
| No.0 | 0.012109 | 0.000567541 | 0.0126766 |
| No.3 | 0.012483 | 0 | 0.012483 |
| No.24 | 0.010523 | 0.00040476 | 0.0109277 |
| No.15 | 0.00993564 | 0.000455824 | 0.0103915 |
| No.4 | 0.0102933 | 0 | 0.0102933 |
| No.12 | 0.00953403 | 0 | 0.00953403 |
| No.18 | 0.00860969 | 0.000630473 | 0.00924016 |
| No.13 | 0.0085282 | 0.000590106 | 0.0091183 |
| No.9 | 0.00889977 | 0 | 0.00889977 |
| No.7 | 0.00817527 | 0.000512945 | 0.00868821 |
| No.1 | 0.00782346 | 0.000366095 | 0.00818955 |
| No.16 | 0.00758058 | 0.000349514 | 0.0079301 |
| No.11 | 0.00790131 | 0 | 0.00790131 |
| No.5 | 0.00751637 | 0.000307075 | 0.00782345 |
| No.29 | 0.00724169 | 0.000440729 | 0.00768242 |

## Top hyper-mediators

| paper_component | path_amce | synergy_order2 | hyper_amce_total |
| --- | ---: | ---: | ---: |
| No.18 | 6.17258e-06 | 0.000630473 | 0.000636646 |
| No.13 | 1.01959e-05 | 0.000590106 | 0.000600302 |
| No.0 | 2.02675e-06 | 0.000567541 | 0.000569567 |
| No.7 | 1.09789e-05 | 0.000512945 | 0.000523924 |
| No.6 | 2.98659e-06 | 0.000515243 | 0.000518229 |
| No.14 | 4.765e-06 | 0.000458691 | 0.000463456 |
| No.15 | 1.64406e-06 | 0.000455824 | 0.000457468 |
| No.29 | 7.35036e-06 | 0.000440729 | 0.000448079 |
| No.32 | 2.54655e-06 | 0.000430695 | 0.000433241 |
| No.24 | 3.83988e-06 | 0.00040476 | 0.0004086 |
| No.1 | 3.8033e-06 | 0.000366095 | 0.000369898 |
| No.16 | 2.90208e-06 | 0.000349514 | 0.000352417 |
| No.5 | 2.49576e-06 | 0.000307075 | 0.000309571 |
| No.35 | 2.73868e-06 | 0.000281405 | 0.000284144 |
| No.43 | 5.99149e-06 | 0 | 5.99149e-06 |

## Ranking comparison vs pairwise baseline

- Gateway Spearman: 0.7663
- Gateway Kendall: 0.5797
- Mediator Spearman: 0.9530
- Mediator Kendall: 0.8915
