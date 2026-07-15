#!/usr/bin/env python3
"""Publication-ready four-view cortical surface maps.

The plotting core depends only on NumPy and Matplotlib.  Meshes may come from
FreeSurfer, GIFTI, nilearn, neuromaps, or any other source after conversion to
``SurfaceMesh``.  Coordinates are assumed to follow the FreeSurfer convention:
x left-to-right, y posterior-to-anterior, and z inferior-to-superior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


@dataclass(frozen=True)
class SurfaceMesh:
    """Triangular cortical mesh and optional sulcal-depth background."""

    coordinates: np.ndarray
    faces: np.ndarray
    sulc: np.ndarray | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=float)
        faces = np.asarray(self.faces, dtype=int)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (n_vertices, 3).")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (n_faces, 3).")
        if faces.size and (faces.min() < 0 or faces.max() >= len(coordinates)):
            raise ValueError("faces contain an out-of-range vertex index.")
        if self.sulc is not None and np.asarray(self.sulc).shape != (len(coordinates),):
            raise ValueError("sulc must contain one value per vertex.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "faces", faces)
        if self.sulc is not None:
            object.__setattr__(self, "sulc", np.asarray(self.sulc, dtype=float))


def parcel_values_to_vertices(
    parcel_labels: Sequence[int],
    parcel_values: Mapping[int, float] | Sequence[float],
    *,
    label_order: Sequence[int] | None = None,
    background_labels: Sequence[int] = (-1, 0),
) -> np.ndarray:
    """Expand parcel-level values to vertices without assuming contiguous labels.

    A mapping is safest.  For an array, ``label_order[i]`` identifies the parcel
    receiving ``parcel_values[i]``.  Background and unknown labels remain NaN.
    """

    labels = np.asarray(parcel_labels, dtype=int)
    if labels.ndim != 1:
        raise ValueError("parcel_labels must be one-dimensional.")
    if isinstance(parcel_values, Mapping):
        lookup = {int(label): float(value) for label, value in parcel_values.items()}
    else:
        values = np.asarray(parcel_values, dtype=float)
        if values.ndim != 1:
            raise ValueError("parcel_values must be one-dimensional.")
        if label_order is None:
            raise ValueError("label_order is required when parcel_values is an array.")
        if len(label_order) != len(values):
            raise ValueError("label_order and parcel_values must have equal length.")
        lookup = {int(label): float(value) for label, value in zip(label_order, values)}

    output = np.full(labels.shape, np.nan, dtype=float)
    background = set(int(label) for label in background_labels)
    for label in np.unique(labels):
        if int(label) not in background and int(label) in lookup:
            output[labels == label] = lookup[int(label)]
    return output


def _robust_unit(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    low, high = np.percentile(finite, (2.0, 98.0))
    if high <= low:
        return np.full_like(values, 0.5, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _face_mean(vertex_values: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sampled = np.asarray(vertex_values, dtype=float)[faces]
    # Requiring all three vertices prevents metric colors bleeding across the
    # atlas/background boundary (especially the medial wall).
    valid = np.all(np.isfinite(sampled), axis=1)
    count = np.sum(np.isfinite(sampled), axis=1)
    total = np.nansum(sampled, axis=1)
    means = np.divide(total, count, out=np.full(len(faces), np.nan), where=count > 0)
    return means, valid


def _face_colors(
    mesh: SurfaceMesh,
    vertex_values: np.ndarray,
    *,
    cmap: Colormap,
    norm: Normalize,
    background_color: str,
) -> np.ndarray:
    values, valid = _face_mean(vertex_values, mesh.faces)
    colors = np.tile(mpl.colors.to_rgba(background_color), (len(mesh.faces), 1))
    colors[valid] = cmap(norm(values[valid]))

    triangles = mesh.coordinates[mesh.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    light = np.array([-0.35, -0.25, 0.90])
    light /= np.linalg.norm(light)
    geometry = 0.78 + 0.22 * np.abs(normals @ light)

    if mesh.sulc is not None:
        sulc, _ = _face_mean(mesh.sulc, mesh.faces)
        sulc_unit = np.nan_to_num(_robust_unit(sulc), nan=0.5)
        sulc_shade = 0.84 + 0.20 * sulc_unit
    else:
        sulc_shade = 1.0
    colors[:, :3] = np.clip(colors[:, :3] * (geometry * sulc_shade)[:, None], 0.0, 1.0)
    return colors


def _draw_surface(
    axis,
    mesh: SurfaceMesh,
    values: np.ndarray,
    *,
    cmap: Colormap,
    norm: Normalize,
    elev: float,
    azim: float,
    background_color: str,
) -> None:
    triangles = mesh.coordinates[mesh.faces]
    collection = Poly3DCollection(
        triangles,
        facecolors=_face_colors(mesh, values, cmap=cmap, norm=norm, background_color=background_color),
        edgecolors="none",
        linewidths=0.0,
        antialiased=False,
    )
    axis.add_collection3d(collection)
    mins = mesh.coordinates.min(axis=0)
    maxs = mesh.coordinates.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.51 * float(np.max(maxs - mins))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=elev, azim=azim)
    axis.set_proj_type("ortho")
    axis.set_axis_off()
    axis.set_facecolor("none")


def draw_brain_map_four_views(
    axes: Sequence,
    colorbar_axis,
    left_mesh: SurfaceMesh,
    right_mesh: SurfaceMesh,
    left_values: Sequence[float],
    right_values: Sequence[float],
    *,
    cmap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    symmetric: bool = False,
    background_color: str = "#D7D7D7",
    view_labels: bool = False,
    colorbar_label_size: float = 8.0,
) -> Normalize:
    """Draw four views into existing 3D axes and a horizontal colorbar axis."""

    if len(axes) != 4:
        raise ValueError("axes must contain four 3D axes in lateral/medial order.")
    left_values = np.asarray(left_values, dtype=float)
    right_values = np.asarray(right_values, dtype=float)
    if left_values.shape != (len(left_mesh.coordinates),):
        raise ValueError("left_values must contain one value per left-mesh vertex.")
    if right_values.shape != (len(right_mesh.coordinates),):
        raise ValueError("right_values must contain one value per right-mesh vertex.")
    finite = np.concatenate((left_values[np.isfinite(left_values)], right_values[np.isfinite(right_values)]))
    if finite.size == 0:
        raise ValueError("At least one finite cortical value is required.")
    requested_vmin, requested_vmax = vmin, vmax
    if symmetric:
        limit = max(abs(float(finite.min())), abs(float(finite.max())))
        if limit == 0.0:
            limit = 1.0
        vmin, vmax = -limit, limit
    else:
        vmin = float(finite.min()) if vmin is None else float(vmin)
        vmax = float(finite.max()) if vmax is None else float(vmax)
        if vmax == vmin and requested_vmin is None and requested_vmax is None:
            padding = 0.5 if vmin == 0.0 else 0.01 * abs(vmin)
            vmin, vmax = vmin - padding, vmax + padding
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("vmin and vmax must be finite with vmax > vmin.")

    color_map = mpl.colormaps.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    specs = (
        (left_mesh, left_values, 0.0, 180.0, "LH lateral"),
        (right_mesh, right_values, 0.0, 0.0, "RH lateral"),
        (left_mesh, left_values, 0.0, 0.0, "LH medial"),
        (right_mesh, right_values, 0.0, 180.0, "RH medial"),
    )
    for axis, (mesh, values, elev, azim, label) in zip(axes, specs):
        _draw_surface(
            axis, mesh, values, cmap=color_map, norm=norm, elev=elev, azim=azim,
            background_color=background_color,
        )
        if view_labels:
            axis.text2D(0.04, 0.92, label, transform=axis.transAxes, fontsize=8, color="0.25")

    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=color_map)
    colorbar = axes[0].get_figure().colorbar(
        scalar_mappable, cax=colorbar_axis, orientation="horizontal"
    )
    colorbar.outline.set_linewidth(0.7)
    colorbar.ax.tick_params(labelsize=colorbar_label_size, width=0.7, length=3)
    if colorbar_label:
        colorbar.set_label(colorbar_label, fontsize=colorbar_label_size, labelpad=3)
    return norm


def plot_brain_map_four_views(
    left_mesh: SurfaceMesh,
    right_mesh: SurfaceMesh,
    left_values: Sequence[float],
    right_values: Sequence[float],
    *,
    cmap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    symmetric: bool = False,
    background_color: str = "#D7D7D7",
    view_labels: bool = False,
    figsize: tuple[float, float] = (7.2, 6.0),
) -> tuple[Figure, Normalize]:
    """Create a standalone four-view surface figure with one shared color scale."""

    figure = plt.figure(figsize=figsize, facecolor="white", constrained_layout=False)
    grid = figure.add_gridspec(
        3, 2, height_ratios=(1.0, 1.0, 0.11),
        left=0.015, right=0.985, bottom=0.08, top=0.99, wspace=-0.12, hspace=-0.16,
    )
    axes = [
        figure.add_subplot(grid[row, column], projection="3d")
        for row in range(2)
        for column in range(2)
    ]
    colorbar_axis = figure.add_subplot(grid[2, :])
    colorbar_axis.set_position((0.17, 0.065, 0.66, 0.035))
    norm = draw_brain_map_four_views(
        axes,
        colorbar_axis,
        left_mesh,
        right_mesh,
        left_values,
        right_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=colorbar_label,
        symmetric=symmetric,
        background_color=background_color,
        view_labels=view_labels,
    )
    return figure, norm


def save_brain_map(figure: Figure, output_base: str | Path, *, dpi: int = 450) -> tuple[Path, ...]:
    """Save PNG, PDF, and SVG versions using a suffix-free output path."""

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output_base.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg"))
    figure.savefig(outputs[0], dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    figure.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    return outputs


def load_fsaverage_meshes(mesh: str = "fsaverage5") -> tuple[SurfaceMesh, SurfaceMesh]:
    """Fetch/load a nilearn fsaverage mesh (optional dependency)."""

    try:
        from nilearn import datasets, surface
    except ImportError as error:
        raise ImportError(
            "load_fsaverage_meshes requires nilearn. Install it with `pip install nilearn`. "
            "The rest of this module does not require nilearn."
        ) from error
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    left_coordinates, left_faces = surface.load_surf_mesh(fsaverage.infl_left)
    right_coordinates, right_faces = surface.load_surf_mesh(fsaverage.infl_right)
    left_sulc = surface.load_surf_data(fsaverage.sulc_left)
    right_sulc = surface.load_surf_data(fsaverage.sulc_right)
    return (
        SurfaceMesh(left_coordinates, left_faces, left_sulc),
        SurfaceMesh(right_coordinates, right_faces, right_sulc),
    )


__all__ = [
    "SurfaceMesh",
    "draw_brain_map_four_views",
    "load_fsaverage_meshes",
    "parcel_values_to_vertices",
    "plot_brain_map_four_views",
    "save_brain_map",
]
