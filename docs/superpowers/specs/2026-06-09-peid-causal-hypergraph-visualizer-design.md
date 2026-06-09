# PEID Causal Hypergraph Visualizer Design

## Goal

Build an interactive browser app for constructing small dynamical systems and visualizing the PEID causal hypergraph computed from those systems.

The app is a teaching and method-exploration tool. Users should be able to change the dynamical system itself, then let the app compute pairwise effective information, joint effective information, synergy, and causal hyperedges. Users should not manually edit EI, synergy, or graph edges.

The initial motivating example is the report in `docs/reports/granger_peid_mlp_comparison.md`: a common driver `w -> x,y` and a nonlinear synergistic mechanism `{x,y} -> z`.

## Core Interpretation

The app follows the PEID paper's EI causal hypergraph interpretation:

- A directed pairwise edge `i -> j` represents `EI(X_i,t -> X_j,t+1)`.
- A directed synergistic hyperedge `{i1, ..., ir} -> j` represents positive synergistic effective information from a source set to a target.
- For order 2, synergy is `EI({sources}->target) - sum_i EI(source_i->target)`.
- For order 3 and above, the app computes exact signed Mobius interaction. It shows positive values as high-order hyperedges and keeps negative values only in diagnostics.

Display filtering changes only what is drawn. It does not change computed EI, synergy, or diagnostic tables.

## Architecture

Use a React + D3 browser interface with a Python FastAPI computation service.

In plain terms:

- The browser interface lets users build systems, start computation, inspect tables, and interact with the graph.
- The Python service performs the EI and PEID calculations.
- The graph drawing code consumes computed results and provides layout, filtering, selection, and export.

This separation keeps the PEID algorithm in Python, close to the repository's existing scripts and notebooks, while using the browser for a richer hypergraph interaction.

Suggested first-version placement:

- `apps/peid_hypergraph/backend/` for Python computation and service code,
- `apps/peid_hypergraph/frontend/` for the browser app,
- `tests/` for Python PEID computation tests,
- frontend tests colocated with the browser app if the selected build tool creates that convention.

If the repository already has a web-app convention by implementation time, follow that convention instead.

## Dynamical System Builder

The first version supports two state families.

### Boolean Exact Mode

Users can configure:

- number of nodes,
- update rule for each target node,
- source nodes used by each rule,
- optional bit-flip noise,
- maximum source order for hyperedge search.

Candidate Boolean update rules are fixed:

- `copy(a)`
- `not(a)`
- `and(a,b)`
- `or(a,b)`
- `xor(a,b)`
- `majority(a,b,c)`
- `mux(selector,a,b)`

The app does not allow arbitrary user code as update rules. Fixed rule choices make enumeration reliable and avoid unsafe or invalid formulas.

### Continuous Mode

Users can configure:

- number of nodes,
- update rule form for each target node,
- source nodes used by each rule,
- coefficients such as memory weight, source weight, coupling strength, and noise,
- intervention sample count,
- random seed,
- maximum source order for hyperedge search.

Candidate continuous update rules are fixed:

- `weighted_copy`
- `linear_sum`
- `product(a,b)`
- `sin_product(a,b)`
- `tanh_sum`
- `threshold`
- `self_memory + selected_function`

The first continuous example should include the report's sine system:

```text
w[t+1] = a_w * w[t] + noise
x[t+1] = a_x * x[t] + b_xw * w[t] + noise
y[t+1] = a_y * y[t] + b_yw * w[t] + noise
z[t+1] = a_z * z[t] + alpha * sin(x[t] * y[t]) + noise
```

The user should be able to adjust `alpha`, the common-driver strength, noise, and sample count.

## Computation

### Boolean Computation

For Boolean systems, the service enumerates the full state space and transition matrix. It then computes:

- pairwise EI for each source-target pair,
- joint EI for source sets up to the selected maximum source order,
- order-2 PEID synergy,
- order-3-and-above signed Mobius interaction when enabled.

This mode should be deterministic for a fixed system and noise setting.

### Continuous Computation

For continuous systems, the service samples independent maximum-entropy interventions over each source variable range and evaluates the configured transition function. It estimates EI values with the repository's transport-map mutual information machinery where practical.

The service should report estimator metadata so users know when values are sampled estimates rather than exact enumerations:

- estimator name,
- sample count,
- seed,
- intervention range,
- any clipping or bias correction used,
- source order.

### High-Order Rules

Default `max_source_order` is 2. Users can enable 3.

For order 3 and above:

- compute signed Mobius interaction,
- show positive values in the main hypergraph,
- hide zero and negative values from the main hypergraph,
- include all signed values in a diagnostics table,
- label the diagnostics as signed interaction rather than ordinary nonnegative synergy.

## Data Returned To The Browser

The first version should expose these computation routes:

- `GET /api/examples`: list built-in examples and their editable defaults.
- `POST /api/compute/boolean`: compute exact PEID for a Boolean system definition.
- `POST /api/compute/continuous`: compute sampled PEID for a continuous system definition.

All compute endpoints should return one common result shape:

```json
{
  "nodes": [{"id": "x", "label": "x"}],
  "pairwise_edges": [
    {"source": "w", "target": "x", "ei": 0.6188}
  ],
  "hyperedges": [
    {
      "sources": ["x", "y"],
      "target": "z",
      "source_order": 2,
      "joint_ei": 0.9957,
      "single_ei_sum": 0.2376,
      "synergy": 0.7581,
      "display_value": 0.7581
    }
  ],
  "diagnostics": {
    "signed_interactions": [],
    "estimator": "exact_or_transport_map",
    "source_intervention": "maximum_entropy_independent"
  }
}
```

The browser keeps display settings separately from computed results.

## Browser Experience

The first screen is the usable app, not a landing page.

Layout:

- left panel: dynamical system builder and parameter controls,
- center: causal hypergraph visualization,
- right panel: selected edge or hyperedge explanation and tables.

Graph conventions:

- ordinary directed arrows show pairwise EI,
- hyperedges use a visible connector point from multiple sources to a target,
- line width reflects displayed edge strength,
- color separates ordinary edges from hyperedges,
- clicking a graph object selects it and updates the explanation panel.

The selected hyperedge panel should show:

- source set and target,
- joint EI,
- singleton EI values,
- synergy or signed interaction value,
- estimator metadata,
- why it is or is not drawn in the main graph.

## Dense Graph Display Control

The app must include controls for avoiding overdraw in large systems.

Supported display controls:

- top percentage of pairwise edges,
- top percentage of hyperedges,
- top K pairwise edges,
- top K hyperedges,
- value threshold,
- optional z-score or p-value threshold when diagnostics are available.

These controls affect only the graph and visible/pinned table rows. Full computed tables remain available for inspection and export.

Export choices:

- visible graph image,
- full computed JSON,
- full computed CSV tables,
- visible edge subset CSV.

## Error Handling

The app should reject invalid system definitions before computation:

- no update rule for a target,
- rule arity mismatch,
- duplicate or missing node names,
- source order larger than node count,
- continuous sample count too small,
- continuous parameter ranges that produce nonfinite outputs.

When computation fails, the browser should show a concise message and keep the last successful result visible.

## Testing And Verification

Core tests:

- Boolean XOR identifies `{x0,x1} -> target` as synergistic while singleton EI is zero or near zero.
- Boolean copy identifies a pairwise edge and no false synergistic hyperedge.
- Boolean AND produces a mix of singleton EI and positive order-2 synergy.
- Continuous sine default reproduces the qualitative structure from `granger_peid_mlp_comparison.md`: `w -> x`, `w -> y`, and `{x,y} -> z`.
- Display percentage filtering changes visible edges without changing the full result payload.
- Negative order-3 signed interactions are absent from the main graph but present in diagnostics.

Visual verification:

- run the app locally,
- open it in the browser,
- verify the graph is nonblank,
- verify dense graph controls hide and reveal edges,
- verify clicking pairwise edges and hyperedges updates the explanation panel,
- verify exported visible graph and full data exports work.

## Out Of Scope For First Version

- arbitrary free-form formula input,
- full arbitrary Python code execution from the browser,
- large production-scale graphs,
- publication-quality graph layout optimization,
- automatic inference from uploaded observational data,
- causal effect estimation in the SCM or do-calculus sense.
