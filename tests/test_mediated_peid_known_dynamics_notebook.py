from __future__ import annotations

import json
from pathlib import Path


def test_mediated_peid_known_dynamics_notebook_has_required_sections() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "mediated_peid_known_dynamics.ipynb"

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") in {"code", "markdown"}
    )

    assert "x_{t+1} = a_x x_t + eta_x" in source
    assert "m_{t+1} = a_m m_t + b_x tanh(x_t) + eta_m" in source
    assert "z_{t+1} = a_z z_t + b_m tanh(m_t) + d_x tanh(x_t) + eta_z" in source
    assert "Partial Effective Information Decomposition for Synergistic Causality" in source
    assert "Zotero item key `MYATYWAJ`" in source
    assert "fig/mediated_peid_known_dynamics/mediated_peid_known_dynamics.png" in source
    assert "bbox_to_anchor=(1.02, 0.5)" in source
    assert "MEDIATED_PEID_NUMERIC_CHECKS" in source
    assert "mediator_blocked" in source
    assert "distractor_ei" in source
