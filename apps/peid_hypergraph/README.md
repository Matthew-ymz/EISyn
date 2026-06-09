# PEID Causal Hypergraph App

This app lets users configure small dynamical systems and then computes the PEID causal hypergraph from the configured mechanism.

## Run Backend

From the repository root:

```bash
PYTHONPATH=apps/peid_hypergraph/backend:. uvicorn peid_hypergraph.api:app --reload --host 127.0.0.1 --port 8000
```

The backend exposes:

- `GET /api/examples`
- `POST /api/compute/boolean`
- `POST /api/compute/continuous`

## Run Frontend

From `apps/peid_hypergraph/frontend`:

```bash
npm install --cache .npm-cache
npm test -- --run
npm run build
npm run dev
```

The frontend development server proxies `/api` calls to `http://127.0.0.1:8000`.

## Tests

From the repository root:

```bash
python -m pytest tests/test_peid_hypergraph_compute.py -q
```

From `apps/peid_hypergraph/frontend`:

```bash
npm test -- --run
npm run build
```

## Interface

The left panel controls the dynamical system. In Boolean mode, users can change the node count and choose a fixed target function such as `xor`, `and`, or `majority`. In continuous mode, users can tune the sine common-driver example from the PEID report.

The center panel draws the computed causal graph. Ordinary arrows represent pairwise EI. Connector-point arrows represent hyperedges.

The right panel shows the selected edge or hyperedge decomposition. Display controls hide or reveal graph elements, but they do not change the full computed result.

