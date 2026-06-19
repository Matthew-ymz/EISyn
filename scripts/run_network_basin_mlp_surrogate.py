"""Run the Neural ER MLP-surrogate basin-Syn validation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp.network_revival.network_basin_pair_ignition import (  # noqa: E402
    MLPBasinSurrogateConfig,
    plot_mlp_basin_surrogate_results,
    run_mlp_basin_surrogate_experiment,
)


def main() -> None:
    config = MLPBasinSurrogateConfig()
    results = run_mlp_basin_surrogate_experiment(config, force=False)
    paths = plot_mlp_basin_surrogate_results(results, config)
    print(
        json.dumps(
            {
                "summary": results["summary"],
                "figures": {key: {name: str(path) for name, path in item.items()} for key, item in paths.items()},
            },
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
