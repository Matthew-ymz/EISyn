from __future__ import annotations

import json
from pathlib import Path


def test_granger_peid_mlp_notebook_has_smoke_entrypoint() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "granger_peid_mlp_comparison.ipynb"

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") in {"code", "markdown"}
    )

    assert "run_comparison_grid" in source
    assert "GRANGER_PEID_NOTEBOOK_SMOKE" in source
    assert "granger_vs_peid_summary.png" in source
    assert "representative_causal_graphs.png" in source
    assert "experiment_report_panels.png" in source
    assert "granger_peid_mlp_comparison.md" in source
