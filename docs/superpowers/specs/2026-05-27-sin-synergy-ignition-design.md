# Sin Synergy Ignition Design

## Goal

Extend `exp/network_revival/notebook_three_node_synergy.ipynb` with a three-node continuous ODE inspired by the network-revival microscopic-intervention workflow and the known nonlinear `sin(Q2 Q3)` synergy structure.

## Scope

The notebook will add a focused section after the existing state-space EI sensitivity workflow. It will define a three-node ODE where nodes 0 and 1 are source variables and node 2 is the target:

```text
dx0/dt = -decay * x0
dx1/dt = -decay * x1
dx2/dt = -target_decay * x2 + alpha * sin(gain * x0 * x1) + (1 - alpha) * source_weight * x0
```

This keeps the target close to the user's discrete structure while making it a continuous dynamical system. The pair term is strongest when both source nodes are held high during ignition, while node 0 alone retains a weaker single-source path for comparison.

## Outputs

The notebook will evaluate node 0 ignition, node 1 ignition, equal-split pair ignition, and a no-ignition baseline over a grid of total ignition costs. It will save a reusable CSV under `results/network_revival_three_node_synergy/`, plus PNG/PDF figures with legends outside the axes.

## Validation

A regression test will assert that the notebook contains the sin ODE ignition workflow, the expected cache name, and outside-axes legend placement. The notebook will be executed after editing to ensure the section runs and writes artifacts.
