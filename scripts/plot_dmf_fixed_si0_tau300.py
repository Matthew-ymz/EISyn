from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_dmf_fixed_uniform_multihorizon import (
    DECOMPOSITION_METRICS,
    KURAMOTO_CACHE,
    METRICS,
    build_summary,
    load_kuramoto_summary,
    plot_confirmatory_results,
    plot_determinism_degeneracy,
    plot_shape_alignment,
)


DEFAULT_CHUNKS = ROOT / "results" / "dmf_fixed_si0_tau300" / "confirmatory_chunks"
DEFAULT_OUTPUT = ROOT / "results" / "dmf_fixed_si0_tau300" / "aggregate_9point.npz"
G_INDICES = (0, 4, 6, 7, 8, 9, 12, 16, 20)
SEEDS = tuple(range(8))


def aggregate(chunks_dir: Path) -> dict[str, np.ndarray]:
    position = {index: pos for pos, index in enumerate(G_INDICES)}
    shape = (len(SEEDS), len(G_INDICES))
    combined = {name: np.full(shape, np.nan, dtype=float) for name in METRICS + DECOMPOSITION_METRICS}
    combined["G"] = np.full(len(G_INDICES), np.nan, dtype=float)
    combined["clip_fraction"] = np.full(shape, np.nan, dtype=float)

    for path in sorted(chunks_dir.glob("direct_se_tau300_g*_seed*.npz")):
        with np.load(path) as archive:
            modes = [str(item) for item in archive["modes"]]
            seeds = np.asarray(archive["seeds"], dtype=int)
            selected = np.asarray(archive["selected_g_indices"], dtype=int)
            if modes != ["direct"] or seeds.size != 1 or str(np.asarray(archive["source_state"]).item()) != "se":
                raise ValueError(f"Unexpected control protocol in {path.name}.")
            seed = int(seeds[0])
            if seed not in SEEDS:
                continue
            seed_position = SEEDS.index(seed)
            for local_position, g_index in enumerate(selected):
                if int(g_index) not in position:
                    continue
                global_position = position[int(g_index)]
                if np.isfinite(combined["whole_ei"][seed_position, global_position]):
                    raise ValueError(f"Duplicate seed/G record in {path.name}.")
                for name in METRICS + DECOMPOSITION_METRICS:
                    combined[name][seed_position, global_position] = float(archive[name][0, 0, local_position])
                combined["clip_fraction"][seed_position, global_position] = float(
                    archive["clip_fraction"][0, 0, local_position]
                )
                combined["G"][global_position] = float(archive["G"][local_position])

    if np.any(~np.isfinite(combined["whole_ei"])):
        raise ValueError("Missing one or more fixed-sI0 control conditions.")
    combined["seeds"] = np.asarray(SEEDS, dtype=int)
    combined["selected_g_indices"] = np.asarray(G_INDICES, dtype=int)
    combined["horizon"] = np.asarray(300, dtype=int)
    combined["sample_count"] = np.asarray(512, dtype=int)
    combined["source_count"] = np.asarray(83, dtype=int)
    combined["target_state"] = np.asarray("se_si")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate fixed-sI0 DMF tau=300 results and plot the final comparison.")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = aggregate(args.chunks_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **combined)
    summary = build_summary(combined)
    summary["source_background"] = "sI fixed at zero"
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_confirmatory_results(combined, ROOT / "fig" / "dmf_fixed_si0_tau300_main")
    plot_determinism_degeneracy(combined, ROOT / "fig" / "dmf_fixed_si0_tau300_determinism_degeneracy")
    plot_shape_alignment(
        combined, load_kuramoto_summary(KURAMOTO_CACHE), ROOT / "fig" / "dmf_kuramoto_ei_alignment_fixed_si0_tau300",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
