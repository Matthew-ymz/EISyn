from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from sklearn.decomposition import PCA


DEFAULT_DATA_DIR = Path("data/ncep_reanalysis_slp")
RESULT_SUBDIR = Path("results/runge2015_gateways")
FIG_SUBDIR = Path("fig/runge2015_gateways")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


@dataclass(frozen=True)
class RungeConfig:
    mode: Literal["smoke", "full"] = "full"
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = Path(".")
    start_year: int = 1948
    end_year: int = 2012
    n_components: int = 60
    max_lag: int = 4
    pc_alpha: float = 0.001
    seed: int = 42
    causal_backend: Literal["tigramite", "regression"] = "tigramite"


@dataclass(frozen=True)
class LaggedEdge:
    source: int
    target: int
    lag: int
    coefficient: float
    p_value: float


@dataclass(frozen=True)
class SemEffects:
    direct_effects: pd.DataFrame
    total_effects: pd.DataFrame
    path_effects: pd.DataFrame
    gateway_scores: pd.DataFrame
    mediator_scores: pd.DataFrame


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    root_candidate = _repo_root() / candidate
    return root_candidate if root_candidate.exists() else candidate.resolve()


def varimax(loadings: np.ndarray, *, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Rotate columns of a loading matrix using orthogonal Varimax."""

    phi = np.asarray(loadings, dtype=float)
    if phi.ndim != 2:
        raise ValueError("loadings must be a two-dimensional array.")
    n_rows, n_cols = phi.shape
    if n_cols == 0:
        raise ValueError("loadings must contain at least one component.")

    rotation = np.eye(n_cols)
    previous = 0.0
    for _ in range(max_iter):
        transformed = phi @ rotation
        u, singular_values, vh = np.linalg.svd(
            phi.T
            @ (
                transformed**3
                - (gamma / n_rows) * transformed @ np.diag(np.sum(transformed**2, axis=0))
            ),
            full_matrices=False,
        )
        rotation = u @ vh
        total = float(np.sum(singular_values))
        if previous and total < previous * (1.0 + tol):
            break
        previous = total
    return phi @ rotation, rotation


def weekly_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    n_weeks = len(frame) // 7
    if n_weeks == 0:
        return frame.iloc[0:0].copy()
    trimmed = frame.iloc[: n_weeks * 7]
    values = trimmed.to_numpy(dtype=float).reshape(n_weeks, 7, frame.shape[1]).mean(axis=1)
    index = trimmed.index[::7][:n_weeks]
    weekly = pd.DataFrame(values, index=index, columns=frame.columns)
    weekly.index.name = "time"
    return weekly


def load_daily_slp(data_dir: str | Path, start_year: int, end_year: int) -> xr.DataArray:
    resolved = _resolve_path(data_dir)
    daily_dir = resolved / "daily"
    paths = [daily_dir / f"slp.{year}.nc" for year in range(int(start_year), int(end_year) + 1)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Missing NCEP daily SLP file(s): {preview}")

    arrays: list[xr.DataArray] = []
    for path in paths:
        with xr.open_dataset(path) as ds:
            if "slp" not in ds:
                raise ValueError(f"{path} does not contain variable 'slp'.")
            arrays.append(ds["slp"].load())
    slp = xr.concat(arrays, dim="time").sortby("time")
    return slp.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))


def daily_slp_paths(data_dir: str | Path, start_year: int, end_year: int) -> list[Path]:
    resolved = _resolve_path(data_dir)
    daily_dir = resolved / "daily"
    return [daily_dir / f"slp.{year}.nc" for year in range(int(start_year), int(end_year) + 1)]


def standardize_daily_anomalies(slp: xr.DataArray) -> xr.DataArray:
    day = slp["time"].dt.dayofyear
    counts = slp.groupby(day).count("time")
    if int(counts.max()) <= 1:
        anomaly = slp - slp.mean("time")
        scale = slp.std("time").where(lambda value: value > 0.0, 1.0)
        return (anomaly / scale).fillna(0.0)
    climatology = slp.groupby(day).mean("time")
    anomaly = slp.groupby(day) - climatology
    scale = anomaly.groupby(day).std("time")
    scale = scale.where(np.isfinite(scale) & (scale > 0.0), 1.0)
    standardized = anomaly.groupby(day) / scale
    return standardized.fillna(0.0)


def latitude_area_weights(latitudes: xr.DataArray | np.ndarray) -> np.ndarray:
    lat = np.asarray(latitudes, dtype=float)
    weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, None))
    weights[~np.isfinite(weights)] = 0.0
    return weights


def fit_varimax_components(
    standardized_slp: xr.DataArray,
    *,
    n_components: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    values = np.asarray(standardized_slp.values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("SLP array must have shape [time, lat, lon].")
    n_time, n_lat, n_lon = values.shape
    if n_components < 1 or n_components > min(n_time, n_lat * n_lon):
        raise ValueError("n_components must be between 1 and min(time, grid_size).")

    weights = latitude_area_weights(standardized_slp["lat"].values)
    weighted = values * weights[None, :, None]
    matrix = weighted.reshape(n_time, n_lat * n_lon)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=int(n_components), svd_solver="randomized", random_state=int(seed))
    pca_scores = pca.fit_transform(matrix)
    loadings = pca.components_.T * np.sqrt(np.maximum(pca.explained_variance_, 0.0))[None, :]
    rotated_loadings, rotation = varimax(loadings)
    rotated_scores = pca_scores @ rotation
    rotated_scores = (rotated_scores - rotated_scores.mean(axis=0, keepdims=True)) / np.maximum(
        rotated_scores.std(axis=0, ddof=1, keepdims=True), 1.0e-12
    )

    columns = [f"component_{index + 1:02d}" for index in range(int(n_components))]
    dates = pd.to_datetime(standardized_slp["time"].values)
    scores = pd.DataFrame(rotated_scores, index=dates, columns=columns)
    maps = rotated_loadings.reshape(n_lat, n_lon, int(n_components))
    return scores, maps, np.asarray(pca.explained_variance_ratio_, dtype=float)


def discover_causal_edges(
    weekly_scores: pd.DataFrame,
    *,
    max_lag: int,
    pc_alpha: float,
    backend: Literal["tigramite", "regression"],
) -> list[LaggedEdge]:
    if backend == "tigramite":
        return _discover_causal_edges_tigramite(weekly_scores, max_lag=max_lag, pc_alpha=pc_alpha)
    if backend == "regression":
        return _discover_causal_edges_regression(weekly_scores, max_lag=max_lag, pc_alpha=pc_alpha)
    raise ValueError(f"Unsupported causal backend: {backend}")


def ensure_causal_backend_available(backend: Literal["tigramite", "regression"]) -> None:
    if backend == "regression":
        return
    if backend != "tigramite":
        raise ValueError(f"Unsupported causal backend: {backend}")
    if importlib.util.find_spec("tigramite") is None:
        raise RuntimeError(
            "tigramite is required for the paper-aligned causal reconstruction. "
            "Install it with `pip install tigramite` or run smoke tests with "
            "`--causal-backend regression`."
        )


def _discover_causal_edges_tigramite(weekly_scores: pd.DataFrame, *, max_lag: int, pc_alpha: float) -> list[LaggedEdge]:
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise RuntimeError(
            "tigramite is required for the paper-aligned causal reconstruction. "
            "Install it with `pip install tigramite` or run smoke tests with "
            "`--causal-backend regression`."
        ) from exc

    data = pp.DataFrame(weekly_scores.to_numpy(dtype=float), var_names=list(weekly_scores.columns))
    pcmci = PCMCI(dataframe=data, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    result = pcmci.run_pcmci(tau_max=int(max_lag), pc_alpha=float(pc_alpha))
    p_matrix = np.asarray(result["p_matrix"], dtype=float)
    val_matrix = np.asarray(result["val_matrix"], dtype=float)
    n_components = weekly_scores.shape[1]
    edges: list[LaggedEdge] = []
    for source in range(n_components):
        for target in range(n_components):
            for lag in range(1, int(max_lag) + 1):
                p_value = float(p_matrix[source, target, lag])
                if np.isfinite(p_value) and p_value <= pc_alpha:
                    edges.append(
                        LaggedEdge(
                            source=source,
                            target=target,
                            lag=lag,
                            coefficient=float(val_matrix[source, target, lag]),
                            p_value=p_value,
                        )
                    )
    return edges


def _discover_causal_edges_regression(weekly_scores: pd.DataFrame, *, max_lag: int, pc_alpha: float) -> list[LaggedEdge]:
    values = weekly_scores.to_numpy(dtype=float)
    n_time, n_components = values.shape
    edges: list[LaggedEdge] = []
    candidates: list[LaggedEdge] = []
    for source in range(n_components):
        for target in range(n_components):
            for lag in range(1, int(max_lag) + 1):
                if n_time <= lag + 2:
                    continue
                x = values[:-lag, source]
                y = values[lag:, target]
                if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
                    continue
                fit = stats.linregress(x, y)
                edge = LaggedEdge(
                    source=source,
                    target=target,
                    lag=lag,
                    coefficient=float(fit.slope),
                    p_value=float(fit.pvalue) if np.isfinite(fit.pvalue) else 1.0,
                )
                candidates.append(edge)
                if edge.p_value <= pc_alpha:
                    edges.append(edge)
    if edges or not candidates:
        return edges
    candidates.sort(key=lambda edge: (edge.p_value, -abs(edge.coefficient)))
    return candidates[: min(3, len(candidates))]


def compute_sem_effects(edges: Sequence[LaggedEdge], *, n_components: int, max_lag: int) -> SemEffects:
    direct = np.zeros((int(n_components), int(n_components)), dtype=float)
    direct_rows: list[dict[str, float | int]] = []
    for edge in edges:
        direct[edge.source, edge.target] += float(edge.coefficient)
        direct_rows.append(
            {
                "source": int(edge.source),
                "target": int(edge.target),
                "lag": int(edge.lag),
                "coefficient": float(edge.coefficient),
                "p_value": float(edge.p_value),
            }
        )

    total = np.zeros_like(direct)
    power = direct.copy()
    for _ in range(max(1, int(n_components))):
        total += power
        power = power @ direct

    total_rows = [
        {"source": source, "target": target, "total_effect": float(total[source, target])}
        for source in range(int(n_components))
        for target in range(int(n_components))
        if source != target and abs(float(total[source, target])) > 1.0e-12
    ]
    path_rows: list[dict[str, float | int]] = []
    for source in range(int(n_components)):
        for mediator in range(int(n_components)):
            if source == mediator or abs(direct[source, mediator]) <= 1.0e-12:
                continue
            for target in range(int(n_components)):
                if target in (source, mediator):
                    continue
                mediated = float(direct[source, mediator] * total[mediator, target])
                if abs(mediated) > 1.0e-12:
                    path_rows.append(
                        {
                            "source": source,
                            "mediator": mediator,
                            "target": target,
                            "amce": mediated,
                            "abs_amce": abs(mediated),
                        }
                    )

    gateway_rows = []
    mediator_rows = []
    mediated_total = float(sum(float(row["abs_amce"]) for row in path_rows))
    for component in range(int(n_components)):
        outgoing = float(np.sum(np.abs(total[component, :])))
        incoming = float(np.sum(np.abs(total[:, component])))
        gateway_rows.append(
            {
                "component": component,
                "ace": outgoing,
                "acs": incoming,
                "incoming_total_effect": incoming,
                "direct_out_strength": float(np.sum(np.abs(direct[component, :]))),
                "direct_in_strength": float(np.sum(np.abs(direct[:, component]))),
            }
        )
        mediated = sum(float(row["abs_amce"]) for row in path_rows if int(row["mediator"]) == component)
        mediator_rows.append(
            {
                "component": component,
                "amce": mediated,
                "mediated_fraction": float(mediated / mediated_total) if mediated_total > 0.0 else 0.0,
            }
        )

    return SemEffects(
        direct_effects=pd.DataFrame(direct_rows),
        total_effects=pd.DataFrame(total_rows),
        path_effects=pd.DataFrame(path_rows),
        gateway_scores=pd.DataFrame(gateway_rows).sort_values("ace", ascending=False).reset_index(drop=True),
        mediator_scores=pd.DataFrame(mediator_rows).sort_values("amce", ascending=False).reset_index(drop=True),
    )


def save_ranking_figure(frame: pd.DataFrame, output_path: str | Path, *, title: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        plot_frame = pd.DataFrame({"component": [], "score": []})
    else:
        score_columns = [column for column in ("ace", "acs", "amce") if column in frame.columns]
        if not score_columns:
            raise ValueError("ranking frame must contain one of ace, acs, or amce.")
        primary = score_columns[0]
        plot_frame = frame.sort_values(primary, ascending=False).head(15).copy()

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    x = np.arange(len(plot_frame))
    if len(plot_frame) == 0:
        ax.text(0.5, 0.5, "No significant links", ha="center", va="center", transform=ax.transAxes)
    elif "ace" in plot_frame.columns and "acs" in plot_frame.columns:
        ax.bar(x - 0.18, plot_frame["ace"], width=0.36, label="ACE", color="#4c78a8")
        ax.bar(x + 0.18, plot_frame["acs"], width=0.36, label="ACS", color="#f58518")
    else:
        score_column = "amce" if "amce" in plot_frame.columns else "ace"
        ax.bar(x, plot_frame[score_column], width=0.64, label=score_column.upper(), color="#54a24b")
    labels = [f"C{int(component) + 1}" for component in plot_frame.get("component", [])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("effect score")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_component_map_figure(component_maps: np.ndarray, output_path: str | Path, *, n_show: int = 6) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    n_components = component_maps.shape[-1]
    count = min(int(n_show), n_components)
    fig, axes = plt.subplots(2, int(np.ceil(count / 2)), figsize=(8.0, 4.2), constrained_layout=True)
    flat_axes = np.ravel(axes)
    vlim = float(np.nanpercentile(np.abs(component_maps[..., :count]), 98))
    vlim = max(vlim, 1.0e-9)
    for index, ax in enumerate(flat_axes):
        if index >= count:
            ax.axis("off")
            continue
        image = ax.imshow(component_maps[..., index], cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_title(f"C{index + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=list(flat_axes[:count]), location="right", shrink=0.72, label="rotated loading")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def save_causal_network_figure(edges: Sequence[LaggedEdge], output_path: str | Path, *, n_components: int) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    ax.set_aspect("equal")
    ax.axis("off")

    angles = np.linspace(0.0, 2.0 * np.pi, int(n_components), endpoint=False)
    positions = {
        index: np.array([np.cos(angle), np.sin(angle)], dtype=float)
        for index, angle in enumerate(angles)
    }
    strengths = [abs(float(edge.coefficient)) for edge in edges]
    max_strength = max(strengths) if strengths else 1.0
    sorted_edges = sorted(edges, key=lambda edge: abs(float(edge.coefficient)), reverse=True)
    for edge in sorted_edges[: min(80, len(sorted_edges))]:
        start = positions[int(edge.source)]
        end = positions[int(edge.target)]
        if int(edge.source) == int(edge.target):
            loop_center = start * 1.05
            circle = plt.Circle(loop_center, 0.12, fill=False, color="#6f6f6f", lw=0.8 + 2.0 * abs(edge.coefficient) / max_strength, alpha=0.65)
            ax.add_patch(circle)
            continue
        delta = end - start
        start2 = start + 0.11 * delta
        end2 = end - 0.11 * delta
        ax.annotate(
            "",
            xy=end2,
            xytext=start2,
            arrowprops={
                "arrowstyle": "->",
                "color": "#4f6f8f",
                "lw": 0.6 + 2.4 * abs(edge.coefficient) / max_strength,
                "alpha": 0.38 + 0.55 * abs(edge.coefficient) / max_strength,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    for index, position in positions.items():
        ax.scatter([position[0]], [position[1]], s=280, color="#f2f5f8", edgecolor="#36454f", zorder=3)
        ax.text(position[0], position[1], f"C{index + 1}", ha="center", va="center", fontsize=8, zorder=4)
    ax.set_title("Lagged causal network")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def dependency_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "xarray", "scipy", "scikit-learn", "tigramite"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, n: int = 10) -> str:
    subset = frame.loc[:, list(columns)].head(n).copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in subset.to_dict("records"):
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary_report(
    path: str | Path,
    *,
    manifest: dict[str, object],
    gateway_scores: pd.DataFrame,
    mediator_scores: pd.DataFrame,
) -> Path:
    output = Path(path)
    lines = [
        "# Runge 2015 causal gateways and mediators reproduction",
        "",
        "This report reproduces the core workflow from `Identifying causal gateways and mediators in complex spatio-temporal systems` on the local NCEP/NCAR sea-level-pressure data.",
        "",
        "## Method",
        "",
        "- Daily SLP fields are restricted to the configured year range, transformed to standardized daily anomalies, and latitude-area weighted.",
        "- The weighted anomaly matrix is reduced to Varimax-rotated PCA components.",
        "- Component scores are aggregated to weekly resolution.",
        "- Lagged causal links are reconstructed with the configured backend; the full reproduction uses Tigramite PCMCI with ParCorr.",
        "- Causal gateways are ranked by outgoing average causal effect (ACE), susceptibility by incoming total effect (ACS), and mediators by absolute mediated causal effect (AMCE).",
        "",
        "## Run",
        "",
        f"- Years: {manifest['config']['start_year']}-{manifest['config']['end_year']}",
        f"- Components: {manifest['config']['n_components']}",
        f"- Weekly lag maximum: {manifest['config']['max_lag']}",
        f"- Backend: {manifest['config']['causal_backend']}",
        f"- Daily samples: {manifest['n_daily_samples']}",
        f"- Weekly samples: {manifest['n_weekly_samples']}",
        f"- Causal links: {manifest['n_edges']}",
        "",
        "## Top causal gateways",
        "",
        _markdown_table(gateway_scores, ["component", "ace", "acs", "direct_out_strength", "direct_in_strength"]),
        "",
        "## Top causal mediators",
        "",
        _markdown_table(mediator_scores, ["component", "amce", "mediated_fraction"]),
        "",
        "## Artifacts",
        "",
        "- `causal_edges.csv`: lagged directed links.",
        "- `gateway_scores.csv`: component ACE/ACS rankings.",
        "- `mediator_scores.csv`: component AMCE rankings.",
        "- `mediated_path_effects.csv`: source-mediator-target path effects.",
        "- `component_weekly_scores.csv`: weekly rotated component scores.",
        "- `fig/runge2015_gateways/*.png`: component maps, network, gateway ranking, and mediator ranking.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def save_outputs(
    config: RungeConfig,
    *,
    daily_scores: pd.DataFrame,
    weekly_scores: pd.DataFrame,
    component_maps: np.ndarray,
    explained_variance_ratio: np.ndarray,
    edges: Sequence[LaggedEdge],
    effects: SemEffects,
) -> dict[str, object]:
    result_dir = config.output_dir / RESULT_SUBDIR
    fig_dir = config.output_dir / FIG_SUBDIR
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    daily_scores.to_csv(result_dir / "component_daily_scores.csv", index_label="time")
    weekly_scores.to_csv(result_dir / "component_weekly_scores.csv", index_label="time")
    np.savez_compressed(
        result_dir / "component_maps.npz",
        component_maps=component_maps,
        explained_variance_ratio=explained_variance_ratio,
    )

    edge_rows = [asdict(edge) for edge in edges]
    edge_columns = ["source", "target", "lag", "coefficient", "p_value"]
    with (result_dir / "causal_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=edge_columns)
        writer.writeheader()
        writer.writerows(edge_rows)

    effects.direct_effects.to_csv(result_dir / "direct_effects.csv", index=False)
    effects.total_effects.to_csv(result_dir / "total_effects.csv", index=False)
    effects.path_effects.to_csv(result_dir / "mediated_path_effects.csv", index=False)
    effects.gateway_scores.to_csv(result_dir / "gateway_scores.csv", index=False)
    effects.mediator_scores.to_csv(result_dir / "mediator_scores.csv", index=False)

    save_component_map_figure(component_maps, fig_dir / "component_maps.png")
    save_causal_network_figure(edges, fig_dir / "causal_network.png", n_components=config.n_components)
    save_ranking_figure(effects.gateway_scores, fig_dir / "gateway_ranking.png", title="Causal gateway ranking")
    save_ranking_figure(effects.mediator_scores, fig_dir / "mediator_ranking.png", title="Causal mediator ranking")

    manifest: dict[str, object] = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "data_files": [str(path) for path in daily_slp_paths(config.data_dir, config.start_year, config.end_year)],
        "dependency_versions": dependency_versions(),
        "n_daily_samples": int(len(daily_scores)),
        "n_weekly_samples": int(len(weekly_scores)),
        "n_edges": int(len(edges)),
        "top_gateways": effects.gateway_scores.head(10).to_dict("records"),
        "top_mediators": effects.mediator_scores.head(10).to_dict("records"),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary_report(
        result_dir / "summary.md",
        manifest=manifest,
        gateway_scores=effects.gateway_scores,
        mediator_scores=effects.mediator_scores,
    )
    return manifest


def run_pipeline(config: RungeConfig) -> dict[str, object]:
    ensure_causal_backend_available(config.causal_backend)
    slp = load_daily_slp(config.data_dir, config.start_year, config.end_year)
    standardized = standardize_daily_anomalies(slp)
    daily_scores, component_maps, explained = fit_varimax_components(
        standardized,
        n_components=config.n_components,
        seed=config.seed,
    )
    weekly_scores = weekly_aggregate(daily_scores)
    edges = discover_causal_edges(
        weekly_scores,
        max_lag=config.max_lag,
        pc_alpha=config.pc_alpha,
        backend=config.causal_backend,
    )
    effects = compute_sem_effects(edges, n_components=config.n_components, max_lag=config.max_lag)
    return save_outputs(
        config,
        daily_scores=daily_scores,
        weekly_scores=weekly_scores,
        component_maps=component_maps,
        explained_variance_ratio=explained,
        edges=edges,
        effects=effects,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--start-year", type=int, default=1948)
    parser.add_argument("--end-year", type=int, default=2012)
    parser.add_argument("--n-components", type=int, default=60)
    parser.add_argument("--max-lag", type=int, default=RungeConfig.max_lag)
    parser.add_argument("--pc-alpha", type=float, default=RungeConfig.pc_alpha)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--causal-backend", choices=["tigramite", "regression"], default="tigramite")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = RungeConfig(
        mode=args.mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        n_components=args.n_components,
        max_lag=args.max_lag,
        pc_alpha=args.pc_alpha,
        seed=args.seed,
        causal_backend=args.causal_backend,
    )
    try:
        manifest = run_pipeline(config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"n_edges": manifest["n_edges"], "result_dir": str(config.output_dir / RESULT_SUBDIR)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
