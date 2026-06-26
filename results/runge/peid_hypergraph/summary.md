# Runge PEID causal hypergraph summary

Higher-order PEID extension of the pairwise MLP-TM-EI gateway readout.
Delta_K is computed by Mobius inversion on the joint EI lattice.
This run is restricted to order 1 and order 2; no order-3 hyperedges are estimated.

## Run

- Components: 60
- Lagged samples: 3333
- Intervention samples: 4096
- Order max: 2
- Null reps: 20 (block size 26)
- MLP cache reused: False
- Self-check max |closed - recursive|: 3.469e-18 (5 samples)

## Hyperedge counts

- Order 1: 3600
- Order 2: 1625
- Order 3: 0

## Top hyper-gateways

| paper_component | hyper_ace_order1 | hyper_ace_order2 | hyper_ace_total |
| --- | ---: | ---: | ---: |
| No.4 | 0.0182878 | 0.000778721 | 0.0190666 |
| No.2 | 0.016445 | 0.000978251 | 0.0174233 |
| No.1 | 0.0164678 | 0.000827547 | 0.0172953 |
| No.6 | 0.0166141 | 0.000551241 | 0.0171654 |
| No.3 | 0.0153497 | 0.000749674 | 0.0160994 |
| No.0 | 0.0145012 | 0.000766275 | 0.0152675 |
| No.18 | 0.0135958 | 0.000538571 | 0.0141344 |
| No.10 | 0.0141243 | 0 | 0.0141243 |
| No.5 | 0.0132552 | 0 | 0.0132552 |
| No.11 | 0.0122786 | 0.000585993 | 0.0128646 |
| No.22 | 0.0119034 | 0.000658271 | 0.0125617 |
| No.15 | 0.0118545 | 0.000551169 | 0.0124057 |
| No.13 | 0.011538 | 0 | 0.011538 |
| No.57 | 0.0103213 | 0.00078012 | 0.0111014 |
| No.9 | 0.0105399 | 0.000502934 | 0.0110428 |

## Top hyper-mediators

| paper_component | path_amce | synergy_order2 | hyper_amce_total |
| --- | ---: | ---: | ---: |
| No.2 | 1.791e-05 | 0.000978251 | 0.000996161 |
| No.1 | 7.03851e-06 | 0.000827547 | 0.000834586 |
| No.4 | 6.93475e-06 | 0.000778721 | 0.000785656 |
| No.57 | 3.27958e-06 | 0.00078012 | 0.0007834 |
| No.0 | 1.01606e-05 | 0.000766275 | 0.000776436 |
| No.3 | 6.8769e-06 | 0.000749674 | 0.000756551 |
| No.22 | 9.13716e-06 | 0.000658271 | 0.000667408 |
| No.11 | 1.1036e-05 | 0.000585993 | 0.000597029 |
| No.41 | 2.541e-06 | 0.000581027 | 0.000583568 |
| No.6 | 5.10016e-06 | 0.000551241 | 0.000556341 |
| No.15 | 2.5246e-06 | 0.000551169 | 0.000553694 |
| No.18 | 3.02676e-06 | 0.000538571 | 0.000541598 |
| No.9 | 1.00153e-05 | 0.000502934 | 0.000512949 |
| No.48 | 8.21485e-06 | 0.000445219 | 0.000453434 |
| No.37 | 6.57468e-06 | 0 | 6.57468e-06 |

## Ranking comparison vs pairwise baseline

- Gateway Spearman: 0.8678
- Gateway Kendall: 0.6780
- Mediator Spearman: 0.9044
- Mediator Kendall: 0.8531
