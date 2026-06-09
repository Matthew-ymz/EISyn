from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .compute import compute_boolean_peid, compute_continuous_peid


BOOLEAN_XOR_EXAMPLE = {
    "id": "boolean_xor",
    "label": "Boolean XOR synergy",
    "state_family": "boolean",
    "payload": {
        "variables": ["x0", "x1", "x2"],
        "update_rules": {
            "x0": {"type": "copy", "inputs": ["x0"]},
            "x1": {"type": "copy", "inputs": ["x1"]},
            "x2": {"type": "xor", "inputs": ["x0", "x1"]},
        },
        "noise": 0.0,
        "max_source_order": 2,
    },
}

CONTINUOUS_SINE_EXAMPLE = {
    "id": "continuous_sine_common_driver",
    "label": "Continuous sine common-driver synergy",
    "state_family": "continuous",
    "payload": {
        "example": "sine_common_driver",
        "alpha": 1.0,
        "beta": 0.75,
        "noise_std": 0.03,
        "intervention_samples": 900,
        "seed": 11,
        "max_source_order": 2,
    },
}


def _compute_or_400(fn, payload: dict[str, object]) -> dict[str, object]:
    try:
        return fn(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def create_app() -> FastAPI:
    app = FastAPI(title="PEID causal hypergraph visualizer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/examples")
    def examples() -> dict[str, object]:
        return {"examples": [BOOLEAN_XOR_EXAMPLE, CONTINUOUS_SINE_EXAMPLE]}

    @app.post("/api/compute/boolean")
    def compute_boolean(payload: dict[str, object]) -> dict[str, object]:
        return _compute_or_400(compute_boolean_peid, payload)

    @app.post("/api/compute/continuous")
    def compute_continuous(payload: dict[str, object]) -> dict[str, object]:
        return _compute_or_400(compute_continuous_peid, payload)

    return app


app = create_app()
