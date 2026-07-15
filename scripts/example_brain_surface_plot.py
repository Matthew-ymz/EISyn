#!/usr/bin/env python3
"""Minimal example for scripts.brain_surface_plot (requires nilearn)."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.brain_surface_plot import load_fsaverage_meshes, plot_brain_map_four_views, save_brain_map


def smooth_demo_values(coordinates: np.ndarray) -> np.ndarray:
    """Synthetic spatial field; replace with real vertex-wise metric values."""

    xyz = (coordinates - coordinates.mean(axis=0)) / coordinates.std(axis=0)
    return 30.0 + 1.2 * xyz[:, 1] + 0.8 * np.sin(1.7 * xyz[:, 2]) - 0.4 * xyz[:, 0]


def main() -> None:
    left_mesh, right_mesh = load_fsaverage_meshes("fsaverage5")
    figure, _ = plot_brain_map_four_views(
        left_mesh,
        right_mesh,
        smooth_demo_values(left_mesh.coordinates),
        smooth_demo_values(right_mesh.coordinates),
        cmap="viridis",
        vmin=27.0,
        vmax=34.0,
        colorbar_label="Metric (a.u.)",
    )
    outputs = save_brain_map(figure, ROOT / "fig" / "brain_surface_four_views_demo")
    plt.close(figure)
    print("Saved:", *(str(path) for path in outputs), sep="\n")


if __name__ == "__main__":
    main()
