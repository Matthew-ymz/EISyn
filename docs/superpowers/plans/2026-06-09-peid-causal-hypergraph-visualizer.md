# PEID Causal Hypergraph Visualizer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first usable PEID causal hypergraph app where users configure small dynamical systems and the app computes EI, synergy, and visible causal hyperedges.

**Architecture:** Add an isolated `apps/peid_hypergraph/` app with a Python computation module and service layer plus a small React/D3 browser interface. Keep PEID math in Python and keep graph density controls in the browser so display filtering never changes computed results.

**Tech Stack:** Python dataclasses/NumPy/FastAPI/pytest; React, TypeScript, Vite, D3; local browser verification.

---

## Chunk 1: Python PEID Computation

### Task 1: Boolean Exact Computation

**Files:**
- Create: `apps/peid_hypergraph/backend/peid_hypergraph/__init__.py`
- Create: `apps/peid_hypergraph/backend/peid_hypergraph/models.py`
- Create: `apps/peid_hypergraph/backend/peid_hypergraph/boolean.py`
- Create: `apps/peid_hypergraph/backend/peid_hypergraph/compute.py`
- Create: `tests/test_peid_hypergraph_compute.py`

- [ ] **Step 1: Write failing XOR/copy/AND tests**

Add tests that call a public `compute_boolean_peid(payload)` function.

Required behaviors:
- XOR target has positive `{x0,x1}->x2` synergy while singleton `x0->x2` and `x1->x2` EI are near zero.
- Copy target has a strong pairwise edge and no false hyperedge.
- AND target has positive singleton EI and positive order-2 synergy.

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_peid_hypergraph_compute.py -q`

Expected: FAIL because `peid_hypergraph` modules do not exist.

- [ ] **Step 3: Implement minimal Boolean PEID**

Implement:
- fixed Boolean rule evaluation,
- full state enumeration,
- deterministic transition matrix with optional bit-flip noise,
- entropy and mutual information in bits,
- pairwise EI,
- joint EI,
- order-2 synergy,
- order-3 signed Mobius interaction when requested.

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m pytest tests/test_peid_hypergraph_compute.py -q`

Expected: PASS.

### Task 2: Continuous Sine Computation

**Files:**
- Modify: `apps/peid_hypergraph/backend/peid_hypergraph/models.py`
- Create: `apps/peid_hypergraph/backend/peid_hypergraph/continuous.py`
- Modify: `apps/peid_hypergraph/backend/peid_hypergraph/compute.py`
- Modify: `tests/test_peid_hypergraph_compute.py`

- [ ] **Step 1: Write failing continuous sine tests**

Add tests that call `compute_continuous_peid(payload)`.

Required behavior:
- default sine system returns nodes `w,x,y,z`,
- pairwise EI includes strong `w->x` and `w->y`,
- hyperedges include positive `{x,y}->z`,
- display filtering helper can hide edges without mutating full results.

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_peid_hypergraph_compute.py -q`

Expected: FAIL because continuous compute is missing.

- [ ] **Step 3: Implement minimal continuous compute**

Implement:
- fixed continuous rule library,
- built-in sine example,
- independent intervention sampling,
- output generation,
- transport-map MI calls for EI,
- generic display filtering helper.

Use small test sample sizes so tests stay fast.

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m pytest tests/test_peid_hypergraph_compute.py -q`

Expected: PASS.

## Chunk 2: Computation Service

### Task 3: FastAPI Wrapper

**Files:**
- Create: `apps/peid_hypergraph/backend/peid_hypergraph/api.py`
- Create: `apps/peid_hypergraph/backend/requirements.txt`
- Modify: `tests/test_peid_hypergraph_compute.py`

- [ ] **Step 1: Write failing API tests**

Add tests using FastAPI's `TestClient`:
- `GET /api/examples` returns Boolean and continuous examples.
- `POST /api/compute/boolean` returns pairwise edges and hyperedges.
- `POST /api/compute/continuous` returns metadata with estimator name.

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_peid_hypergraph_compute.py -q`

Expected: FAIL because API wrapper is missing.

- [ ] **Step 3: Implement API wrapper**

Expose:
- `GET /api/examples`
- `POST /api/compute/boolean`
- `POST /api/compute/continuous`

- [ ] **Step 4: Run tests and verify they pass**

Run: `python -m pytest tests/test_peid_hypergraph_compute.py -q`

Expected: PASS.

## Chunk 3: Browser Interface

### Task 4: React/D3 App

**Files:**
- Create: `apps/peid_hypergraph/frontend/package.json`
- Create: `apps/peid_hypergraph/frontend/index.html`
- Create: `apps/peid_hypergraph/frontend/src/main.tsx`
- Create: `apps/peid_hypergraph/frontend/src/App.tsx`
- Create: `apps/peid_hypergraph/frontend/src/styles.css`
- Create: `apps/peid_hypergraph/frontend/src/graph.ts`
- Create: `apps/peid_hypergraph/frontend/src/filtering.ts`
- Create: `apps/peid_hypergraph/frontend/src/filtering.test.ts`

- [ ] **Step 1: Write failing frontend filtering tests**

Add tests for top-percentage and top-K filtering that preserve full result arrays.

- [ ] **Step 2: Run tests and verify they fail**

Run: `cd apps/peid_hypergraph/frontend && npm test -- --run`

Expected: FAIL because frontend files do not exist or filtering is missing.

- [ ] **Step 3: Implement minimal browser app**

Implement:
- left-side node count and rule controls,
- Boolean/continuous mode switch,
- compute button,
- sample fallback data when service is unavailable,
- center D3 graph with ordinary edges and hyperedge connector points,
- graph display controls for top percent and top K,
- right-side selected edge explanation.

- [ ] **Step 4: Run frontend tests and build**

Run:
- `cd apps/peid_hypergraph/frontend && npm test -- --run`
- `cd apps/peid_hypergraph/frontend && npm run build`

Expected: PASS.

## Chunk 4: Local Integration And Verification

### Task 5: End-To-End Smoke

**Files:**
- Create: `apps/peid_hypergraph/README.md`
- Verify: `apps/peid_hypergraph/backend/peid_hypergraph/*`
- Verify: `apps/peid_hypergraph/frontend/src/*`

- [ ] **Step 1: Add README with run commands**

Document:
- Python test command,
- backend command,
- frontend install/test/build/dev commands,
- plain-language explanation of interface and display filtering.

- [ ] **Step 2: Run backend tests**

Run: `python -m pytest tests/test_peid_hypergraph_compute.py -q`

Expected: PASS.

- [ ] **Step 3: Run frontend tests and build**

Run:
- `cd apps/peid_hypergraph/frontend && npm test -- --run`
- `cd apps/peid_hypergraph/frontend && npm run build`

Expected: PASS.

- [ ] **Step 4: Start local app**

Run backend and frontend dev servers.

- [ ] **Step 5: Browser verification**

Open the frontend in the in-app browser and verify:
- page renders,
- compute button produces a graph,
- density controls hide and reveal edges,
- clicking an edge or hyperedge updates the right panel,
- no obvious text overlap.

