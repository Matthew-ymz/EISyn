from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.brain_surface_plot import SurfaceMesh, parcel_values_to_vertices, plot_brain_map_four_views
from scripts.plot_dmf_roi_yeo7_summary import map_cross_roi_to_desikan_vertices


def tetrahedron(offset: float) -> SurfaceMesh:
    coordinates = np.array([
        [offset - 1.0, -1.0, -1.0],
        [offset + 1.0, -1.0, -1.0],
        [offset, 1.0, -1.0],
        [offset, 0.0, 1.0],
    ])
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return SurfaceMesh(coordinates, faces, np.array([-1.0, 0.0, 0.5, 1.0]))


def test_parcel_values_are_mapped_by_explicit_labels() -> None:
    actual = parcel_values_to_vertices([0, 10, 10, 42, -1], {10: 2.5, 42: 7.0})
    np.testing.assert_allclose(actual[1:4], [2.5, 2.5, 7.0])
    assert np.isnan(actual[[0, 4]]).all()


def test_four_view_figure_has_shared_normalization() -> None:
    left = tetrahedron(-1.5)
    right = tetrahedron(1.5)
    figure, norm = plot_brain_map_four_views(
        left, right, [0.0, 1.0, 2.0, np.nan], [2.0, 3.0, 4.0, np.nan],
        colorbar_label="Score",
    )
    assert (norm.vmin, norm.vmax) == (0.0, 4.0)
    assert len(figure.axes) == 5
    plt.close(figure)


def test_vertex_count_mismatch_is_rejected() -> None:
    mesh = tetrahedron(0.0)
    with pytest.raises(ValueError, match="left_values"):
        plot_brain_map_four_views(mesh, mesh, [1.0], np.ones(4))


def test_cross_roi_values_map_by_desikan_name() -> None:
    region_labels = np.array([f"ctx-lh-roi_{index}" for index in range(34)], dtype=object)
    cross_roi = np.arange(34, dtype=float)
    atlas_names = {0: "unknown", **{index + 1: f"roi_{index}" for index in range(34)}}
    vertex_labels = np.array([0, 1, 2, 34])
    actual = map_cross_roi_to_desikan_vertices(
        region_labels, cross_roi, vertex_labels, atlas_names, hemisphere="lh"
    )
    assert np.isnan(actual[0])
    np.testing.assert_allclose(actual[1:], [0.0, 1.0, 33.0])
