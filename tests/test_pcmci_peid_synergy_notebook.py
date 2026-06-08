from __future__ import annotations

import json
from pathlib import Path


def test_pcmci_peid_synergy_notebook_has_required_experiment_story() -> None:
    path = Path(__file__).resolve().parents[1] / "exp" / "pcmci_peid_synergy_comparison.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") in {"code", "markdown"}
    )

    assert "PCMCI 与 MLP+PEID" in source
    assert "tanh(1.6x_ty_t)" in source
    assert "PCMCI-ParCorr" in source
    assert "PCMCI-CMIknn" in source
    assert "Oracle + PEID" in source
    assert "run_benchmark" in source
    assert "PCMCI_PEID_FULL" in source
    assert "typed_graph_comparison.png" in source
    assert "peid_decomposition.png" in source
    assert "recovery_metrics.png" in source
    assert "robustness_curves.png" in source
    assert "Partial Effective Information Decomposition for Synergistic Causality" in source
    assert "MYATYWAJ" in source
    assert "不能自动解决隐藏混杂" in source
