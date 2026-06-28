from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
FROZEN_RESULTS = ROOT / "results" / "iid_fig6_phi_eid_comparison" / "whole_system_phi_eid_phase_comparison.npz"
CANONICAL_FIGURE = ROOT / "fig" / "iid_fig6_phi_eid_comparison" / "whole_system_phi_eid_phase_comparison.png"
OUTPUT_DIR = ROOT / "_reproduced"
OUTPUT_FIGURE = OUTPUT_DIR / "whole_system_phi_eid_phase_comparison.png"
OUTPUT_DOC = OUTPUT_DIR / "iid_fig6_phi_eid_comparison.md"
EXPECTED_FIGURE_SHA256 = "5a1a8a77450041eda88fe29f9c595d489703b6dd6524ec007e828ff2f59c0c57"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_plotting_module():
    script_path = ROOT / "scripts" / "reproduce_iid_fig6_phi_eid_comparison.py"
    spec = importlib.util.spec_from_file_location("iid_fig6_plotter", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load plotting script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plotter = load_plotting_module()
    payload = dict(np.load(FROZEN_RESULTS, allow_pickle=True))

    plotter.plot_comparison(payload, OUTPUT_FIGURE)
    plotter.write_doc(payload, OUTPUT_DOC, OUTPUT_FIGURE)

    output_sha = sha256(OUTPUT_FIGURE)
    canonical_sha = sha256(CANONICAL_FIGURE)
    print(f"reproduced_png_sha256={output_sha}")
    print(f"canonical_png_sha256={canonical_sha}")
    print(f"frozen_results_sha256={sha256(FROZEN_RESULTS)}")

    if output_sha != EXPECTED_FIGURE_SHA256:
        raise SystemExit(
            "Reproduced PNG does not match the expected SHA-256. "
            "Check Python, numpy, matplotlib, and font environment."
        )
    if canonical_sha != EXPECTED_FIGURE_SHA256:
        raise SystemExit("Bundled canonical PNG checksum mismatch.")

    print(f"OK: exact PNG reproduced at {OUTPUT_FIGURE}")
    print(f"Doc regenerated at {OUTPUT_DOC}")


if __name__ == "__main__":
    main()
