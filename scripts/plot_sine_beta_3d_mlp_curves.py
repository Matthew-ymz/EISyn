#!/usr/bin/env python3
"""Plot the simple-coefficient beta sweep using the cached 3D MLP readouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_granger_peid_mlp import DEFAULT_FIGURE_DIR, DEFAULT_RESULT_DIR
from scripts.run_hidden_w_sine_beta_mlp_peid import (
    plot_combined_with_hidden_w_mlp_readouts,
)


DEFAULT_FULL_RESULT = DEFAULT_RESULT_DIR / "sine_beta_simple_coefficients_full_state.json"
DEFAULT_3D_RESULT = DEFAULT_RESULT_DIR / "sine_beta_simple_coefficients_hidden_w.json"
DEFAULT_LIANG_RESULT = DEFAULT_RESULT_DIR / "sine_beta_simple_coefficients_liang.json"
DEFAULT_STEM = "sine_beta_simple_coefficients_3d_mlp"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-result", type=Path, default=DEFAULT_FULL_RESULT)
    parser.add_argument("--three-d-result", type=Path, default=DEFAULT_3D_RESULT)
    parser.add_argument("--liang-result", type=Path, default=DEFAULT_LIANG_RESULT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    args = parser.parse_args()

    full_result = _read_json(args.full_result)
    three_d_result = _read_json(args.three_d_result)
    liang_result = _read_json(args.liang_result)

    config = dict(three_d_result.get("config", {}))
    observed = list(config.get("observed_variables", []))
    hidden = list(config.get("hidden_variables", []))
    if observed != ["x", "y", "z"] or hidden != ["w"]:
        raise ValueError(
            "The cached MLP result is not the expected 3D condition: "
            f"observed={observed}, hidden={hidden}"
        )

    output = plot_combined_with_hidden_w_mlp_readouts(
        three_d_result,
        full_result,
        args.figure_dir,
        liang_result=liang_result,
        stem=args.stem,
    )
    if output is None:
        raise RuntimeError("The cached sweep did not contain plottable summary rows.")
    print(output)


if __name__ == "__main__":
    main()
