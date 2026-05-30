from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import pickle
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover - optional dependency at import time
    loadmat = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
DEFAULT_LAUSANNE_ATLAS_PATH = (
    REPO_DIR / "data" / "Lausanne2008-33.zip"
)
DEFAULT_HCP83_CONNECTIVITY_STEM = REPO_DIR / "data" / "external" / "hcp_lausanne83_connectivity"
DEFAULT_RESULT_DIR = SCRIPT_DIR / "result"
DEFAULT_LAUSANNE_CONNECTIVITY_SCALE = 0.2
PHIID_SOURCE_URL = "https://github.com/Imperial-MIND-lab/integrated-info-decomp"


@dataclass(frozen=True)
class DMFParameters:
    """Local DMF parameters for the empirical-SC whole-brain model.

    Time constants and integration steps are expressed in seconds.
    """

    w_e: float = 1.0
    w_i: float = 0.7
    i0: float = 0.382
    w_plus: float = 1.4
    j_nmda: float = 0.15

    gain_e: float = 310.0
    threshold_e: float = 0.403
    shape_e: float = 0.16

    gain_i: float = 615.0
    threshold_i: float = 0.288
    shape_i: float = 0.087

    tau_e: float = 0.100
    tau_i: float = 0.010
    gamma_e: float = 0.641
    sigma: float = 0.01

    dt: float = 1.0e-4
    t_total: float = 1.5
    burn_in: float = 0.3

    init_se: float = 0.001
    init_si: float = 0.001


@dataclass(frozen=True)
class FICParameters:
    """Settings for per-region feedback inhibitory control calibration."""

    target_rate_hz: float = 3.0
    tolerance_hz: float = 0.05
    max_iterations: int = 12
    learning_rate: float = 0.025
    initial_j: float = 1.0
    min_j: float = 0.1
    max_j: float = 10.0
    calibration_sigma: float = 0.0
    warm_start_state: bool = True


@dataclass(frozen=True)
class StabilizationParameters:
    """Heuristics for detecting when the firing-rate trace has stabilized."""

    window: float = 0.05
    tolerance_hz: float = 0.05
    confirm_windows: int = 3


@dataclass(frozen=True)
class BalloonWindkesselParameters:
    """Hemodynamic parameters for a Friston-style Balloon-Windkessel transform."""

    tau_s: float = 1.54
    tau_f: float = 2.46
    tau_0: float = 0.98
    alpha: float = 0.32
    e0: float = 0.34
    v0: float = 0.02
    k1: float = 7.0 * 0.34
    k2: float = 2.0
    k3: float = 2.0 * 0.34 - 0.2
    neural_gain: float = 0.01


@dataclass(frozen=True)
class JFICSchedule:
    """Optional externally supplied J_FIC values."""

    values: np.ndarray
    g_values: np.ndarray | None = None


def is_pickle_connectivity_path(path: str | Path) -> bool:
    """Return whether the connectivity path points to a pickle atlas file."""

    name = str(path).lower()
    return name.endswith(".pkl") or name.endswith(".pkl.gz")


def is_lausanne_archive_path(path: str | Path) -> bool:
    """Return whether the path points to a Lausanne atlas archive."""

    name = str(path).lower()
    return is_pickle_connectivity_path(path) or name.endswith(".zip")


def load_pickle_module():
    """Load a pickle implementation that can read protocol-5 files on Python 3.7."""

    vendor_dir = SCRIPT_DIR / "_vendor" / "pickle5"
    if vendor_dir.exists():
        vendor_dir_str = str(vendor_dir)
        if vendor_dir_str not in sys.path:
            sys.path.insert(0, vendor_dir_str)

    try:
        import pickle5 as pickle_module  # type: ignore[import-not-found]

        return pickle_module
    except ImportError:
        return pickle


def load_reference_row_labels(path: str | Path = SCRIPT_DIR / "sn.csv") -> list[str]:
    """Load region labels from the first column of a reference CSV matrix."""

    path = resolve_input_path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if not rows:
        raise ValueError(f"Reference label CSV contains no rows: {path}")
    return [row[0].strip() for row in rows]


def _first_zip_member(zip_file: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in zip_file.namelist() if name.endswith(suffix)]
    if not matches:
        raise ValueError(f"Zip archive contains no `{suffix}` member.")
    return matches[0]


def load_lausanne_region_labels(path: str | Path) -> list[str]:
    """Load Lausanne parcel labels from the atlas export metadata."""

    path = resolve_input_path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            label_member = _first_zip_member(archive, "/export/amplitude/amplitude.csv")
            text = archive.read(label_member).decode("utf-8")
            rows = list(csv.reader(io.StringIO(text)))
    else:
        export_path = path.parent / "export" / "amplitude" / "amplitude.csv"
        if not export_path.exists():
            raise FileNotFoundError(
                f"Could not find Lausanne region labels next to {path}: {export_path}"
            )
        with export_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))

    data_rows = [
        [cell.strip() for cell in row]
        for row in rows
        if row and row[0].strip() and not row[0].strip().startswith("#")
    ]
    if len(data_rows) < 2:
        raise ValueError(f"Lausanne label export contains no parcel rows: {path}")

    labels = [row[0] for row in data_rows[1:]]
    if not labels:
        raise ValueError(f"Lausanne label export contains no labels: {path}")
    return labels


def save_labeled_connectivity_csv(
    path: str | Path,
    matrix: np.ndarray,
    *,
    row_labels: Iterable[str],
) -> Path:
    """Save one connectivity matrix with a leading label column."""

    output_path = resolve_output_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(matrix, dtype=float)
    labels = list(row_labels)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Connectivity matrix must be square, got shape {matrix.shape}.")
    if len(labels) != matrix.shape[0]:
        raise ValueError(
            f"Expected {matrix.shape[0]} row labels, got {len(labels)}."
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{value:.16g}" for value in row]])

    return output_path


def transfer_function(current: np.ndarray, gain: float, threshold: float, shape: float) -> np.ndarray:
    """Smooth current-to-rate transform used by the DMF model."""

    y = gain * (current - threshold)
    denominator = 1.0 - np.exp(-shape * y)

    rate = np.empty_like(y)
    near_singularity = np.abs(denominator) < 1.0e-12
    rate[~near_singularity] = y[~near_singularity] / denominator[~near_singularity]
    rate[near_singularity] = 1.0 / shape
    return rate


def _safe_logdet(matrix: np.ndarray, *, atol: float = 1.0e-10) -> float:
    sym = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(sym)
    return float(np.log(np.clip(eigenvalues, atol, None)).sum())


def gaussian_mutual_information(
    covariance: np.ndarray,
    *,
    sources: Sequence[int],
    targets: Sequence[int],
    log_base: float = np.e,
    atol: float = 1.0e-10,
) -> float:
    """Gaussian mutual information from a covariance matrix."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square.")

    source = list(sources)
    target = list(targets)
    if not source or not target:
        return 0.0

    joint = source + target
    value = 0.5 * (
        _safe_logdet(covariance[np.ix_(source, source)], atol=atol)
        + _safe_logdet(covariance[np.ix_(target, target)], atol=atol)
        - _safe_logdet(covariance[np.ix_(joint, joint)], atol=atol)
    ) / np.log(log_base)
    return max(0.0, float(value))


def compute_pairwise_phi_metrics(
    rates: np.ndarray,
    *,
    atol: float = 1.0e-10,
) -> dict[str, float | int]:
    """Compute pairwise Phi^WMS and Phi^R Gaussian proxy metrics."""

    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 2 or rates.shape[0] < 3 or rates.shape[1] < 2:
        raise ValueError(
            "rates must have shape (time, regions) with at least 3 time points and 2 regions."
        )

    phi_wms_values: list[float] = []
    phi_r_values: list[float] = []
    for left in range(rates.shape[1] - 1):
        for right in range(left + 1, rates.shape[1]):
            lagged = np.column_stack(
                [
                    rates[:-1, left],
                    rates[:-1, right],
                    rates[1:, left],
                    rates[1:, right],
                ]
            )
            covariance = np.cov(lagged, rowvar=False, bias=False)
            covariance = 0.5 * (covariance + covariance.T)

            tdmi = gaussian_mutual_information(
                covariance,
                sources=[0, 1],
                targets=[2, 3],
                atol=atol,
            )
            self_left = gaussian_mutual_information(
                covariance,
                sources=[0],
                targets=[2],
                atol=atol,
            )
            self_right = gaussian_mutual_information(
                covariance,
                sources=[1],
                targets=[3],
                atol=atol,
            )
            phi_wms = tdmi - self_left - self_right
            double_redundancy = min(
                gaussian_mutual_information(
                    covariance,
                    sources=[source],
                    targets=[target],
                    atol=atol,
                )
                for source in (0, 1)
                for target in (2, 3)
            )
            phi_wms_values.append(float(phi_wms))
            phi_r_values.append(float(phi_wms + double_redundancy))

    return {
        "pair_count": len(phi_wms_values),
        "phi_wms_mean": float(np.mean(phi_wms_values)),
        "phi_r_mean": float(np.mean(phi_r_values)),
        "phi_wms_std": float(np.std(phi_wms_values)),
        "phi_r_std": float(np.std(phi_r_values)),
    }


def resolve_input_path(path: str | Path) -> Path:
    """Resolve an input path from cwd first, then from this script's directory."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()

    script_relative = SCRIPT_DIR / candidate
    if script_relative.exists():
        return script_relative.resolve()
    return script_relative


def resolve_output_path(path: str | Path) -> Path:
    """Resolve an output path relative to this script when not absolute."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (SCRIPT_DIR / candidate).resolve()


def load_numeric_array(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a numeric array from .npy, .npz, .csv, or .txt."""

    path = resolve_input_path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        matrix = np.load(path)
        metadata: dict[str, np.ndarray] = {}
    elif suffix == ".npz":
        archive = np.load(path)
        preferred_keys = ("connectivity", "j_fic", "values")
        key = next((candidate for candidate in preferred_keys if candidate in archive), archive.files[0])
        matrix = archive[key]
        metadata = {name: archive[name] for name in archive.files if name != key}
    elif suffix == ".csv":
        matrix, metadata = load_csv_array(path)
    else:
        matrix = np.loadtxt(path, delimiter=None)
        metadata = {}

    return np.asarray(matrix, dtype=float), metadata


def load_csv_array(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a CSV array, optionally discarding label/header columns.

    Supported layouts include:
    - pure numeric CSV matrices
    - numeric matrices with a leading label column
    - numeric matrices with one header row and one leading label column
    - comment-prefixed metadata rows starting with `#`
    """

    path = resolve_input_path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(handle)
            if any(cell.strip() for cell in row)
        ]

    rows = [row for row in raw_rows if row and not row[0].startswith("#")]
    if not rows:
        raise ValueError(f"CSV file contains no data rows: {path}")

    candidates: list[tuple[list[list[str]], dict[str, np.ndarray]]] = [
        (rows, {}),
    ]

    if rows and len(rows[0]) > 1:
        candidates.append(
            (
                [row[1:] for row in rows],
                {"row_labels": np.asarray([row[0] for row in rows], dtype=object)},
            )
        )

    if len(rows) > 1:
        candidates.append(
            (
                rows[1:],
                {"column_labels": np.asarray(rows[0], dtype=object)},
            )
        )

    if len(rows) > 1 and len(rows[0]) > 1:
        candidates.append(
            (
                [row[1:] for row in rows[1:]],
                {
                    "row_labels": np.asarray([row[0] for row in rows[1:]], dtype=object),
                    "column_labels": np.asarray(rows[0][1:], dtype=object),
                },
            )
        )

    last_error: ValueError | None = None
    for candidate_rows, metadata in candidates:
        try:
            matrix = np.asarray(
                [[float(cell) for cell in row] for row in candidate_rows],
                dtype=float,
            )
        except ValueError as exc:
            last_error = exc
            continue

        if matrix.ndim == 2:
            return matrix, metadata

    raise ValueError(
        f"Could not parse numeric CSV data from {path}."
    ) from last_error


def load_lausanne_atlas_entries(path: str | Path) -> list[dict[str, np.ndarray]]:
    """Load the protocol-5 Lausanne atlas pickle that stores count/mean/std matrices."""

    path = resolve_input_path(path)
    pickle_module = load_pickle_module()

    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                pickle_member = _first_zip_member(archive, ".pkl")
                raw = pickle_module.loads(archive.read(pickle_member))
        else:
            open_fn = gzip.open if str(path).lower().endswith(".pkl.gz") else open
            with open_fn(path, "rb") as handle:
                raw = pickle_module.load(handle)
    except ValueError as exc:
        raise ValueError(
            f"Could not load pickle atlas from {path}. "
            "If this is a protocol-5 pickle on Python 3.7, install `pickle5` into "
            "`dynamic_brain/_vendor/pickle5`."
        ) from exc

    entries = list(np.asarray(raw, dtype=object).ravel())
    if not entries:
        raise ValueError(f"Pickle atlas contains no entries: {path}")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "count" not in entry:
            raise ValueError(
                f"Atlas entry {index} must be a dict containing `count`, got {type(entry)}."
            )
    return entries


def prepare_lausanne_count_connectivity(
    count_matrix: np.ndarray,
    *,
    region_labels: Iterable[str] | None = None,
    connectivity_scale: float = DEFAULT_LAUSANNE_CONNECTIVITY_SCALE,
    expected_regions: int | None = None,
    max_regions: int | None = None,
    return_labels: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Remove the Lausanne Unknown parcel, max-normalize, and validate the count matrix."""

    raw = np.asarray(count_matrix, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"Lausanne count matrix must be square, got shape {raw.shape}.")
    if raw.shape[0] < 2:
        raise ValueError(f"Lausanne count matrix is too small to crop: {raw.shape}.")

    if region_labels is None:
        kept_labels = [f"region_{index:03d}" for index in range(raw.shape[0] - 1)]
        cropped = raw[:-1, :-1]
    else:
        labels = [str(label).strip() for label in region_labels]
        if len(labels) != raw.shape[0]:
            raise ValueError(
                f"Expected {raw.shape[0]} Lausanne labels, got {len(labels)}."
            )
        keep_mask = np.asarray(
            [label.casefold() != "unknown" for label in labels],
            dtype=bool,
        )
        removed = int((~keep_mask).sum())
        if removed != 1:
            raise ValueError(
                f"Expected exactly one Lausanne `Unknown` label, found {removed}."
            )
        cropped = raw[np.ix_(keep_mask, keep_mask)]
        kept_labels = [label for label, keep in zip(labels, keep_mask) if keep]

    scale = float(np.nanmax(cropped))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Lausanne count matrix has invalid normalization scale: {scale}.")

    if not np.isfinite(connectivity_scale) or connectivity_scale <= 0.0:
        raise ValueError(
            f"connectivity_scale must be positive and finite, got {connectivity_scale}."
        )

    normalized = (cropped / scale) * float(connectivity_scale)
    connectivity = validate_connectivity_matrix(
        normalized,
        expected_regions=expected_regions,
        max_regions=max_regions,
    )
    if return_labels:
        return connectivity, kept_labels
    return connectivity


def load_lausanne_count_connectivity_matrix(
    path: str | Path,
    *,
    atlas_count_index: int = 0,
    connectivity_scale: float = DEFAULT_LAUSANNE_CONNECTIVITY_SCALE,
    expected_regions: int | None = None,
    max_regions: int | None = None,
) -> np.ndarray:
    """Load one normalized structural-connectivity matrix from the Lausanne atlas pickle."""

    entries = load_lausanne_atlas_entries(path)
    labels = load_lausanne_region_labels(path) if Path(path).suffix.lower() == ".zip" else None
    if atlas_count_index < 0 or atlas_count_index >= len(entries):
        raise ValueError(
            f"atlas_count_index must be in [0, {len(entries) - 1}], got {atlas_count_index}."
        )
    return prepare_lausanne_count_connectivity(
        entries[atlas_count_index]["count"],
        region_labels=labels,
        connectivity_scale=connectivity_scale,
        expected_regions=expected_regions,
        max_regions=max_regions,
    )


def _load_hcp_sc_scales(path: str | Path) -> list[np.ndarray]:
    """Load HCP structural connectivity scales from the public MATLAB archive."""

    if loadmat is None:
        raise ImportError(
            "Loading HCP `.mat` connectivity requires scipy. "
            "Install scipy or pass a pre-exported .npy/.npz/.csv matrix instead."
        )

    raw = loadmat(resolve_input_path(path), squeeze_me=False, struct_as_record=True)
    if "connMatrices" not in raw:
        raise ValueError(
            "MAT connectivity file must contain `connMatrices` in the public HCP format."
        )

    container = raw["connMatrices"]
    if container.size == 0:
        raise ValueError("HCP MAT file contains an empty `connMatrices` structure.")

    sc_cells = container[0, 0]["SC"]
    scales: list[np.ndarray] = []
    for scale_index in range(sc_cells.shape[0]):
        scale_data = np.asarray(sc_cells[scale_index, 0], dtype=float)
        if scale_data.ndim not in (2, 3):
            raise ValueError(
                "Each HCP SC entry must be a 2D or 3D square array, "
                f"got shape {scale_data.shape} at scale index {scale_index}."
            )
        if scale_data.shape[0] != scale_data.shape[1]:
            raise ValueError(
                f"HCP SC entry at scale index {scale_index} must be square, got {scale_data.shape}."
            )
        scales.append(scale_data)

    if not scales:
        raise ValueError("No structural connectivity scales were found in the HCP MAT file.")
    return scales


def load_hcp_connectivity_matrix(
    path: str | Path,
    *,
    max_regions: int | None = 100,
    scale_index: int | None = None,
    subject_index: int | None = None,
) -> np.ndarray:
    """Load one empirical SC matrix from the public HCP connectome archive."""

    scales = _load_hcp_sc_scales(path)
    region_counts = [int(scale.shape[0]) for scale in scales]

    if scale_index is None:
        eligible = [
            index
            for index, region_count in enumerate(region_counts)
            if max_regions is None or region_count <= max_regions
        ]
        if not eligible:
            raise ValueError(
                "No HCP structural connectivity scale satisfies the requested size bound. "
                f"Available region counts: {region_counts}; max_regions={max_regions}."
            )
        scale_index = max(eligible, key=lambda index: region_counts[index])
    elif scale_index < 0 or scale_index >= len(scales):
        raise ValueError(
            f"scale_index must be in [0, {len(scales) - 1}], got {scale_index}."
        )

    selected = np.asarray(scales[scale_index], dtype=float)
    n_regions = int(selected.shape[0])
    if max_regions is not None and n_regions > max_regions:
        raise ValueError(
            f"HCP scale {scale_index} has {n_regions} regions, which exceeds max_regions={max_regions}."
        )

    if selected.ndim == 3:
        n_subjects = int(selected.shape[2])
        if subject_index is None:
            matrix = np.mean(selected, axis=2)
        else:
            if subject_index < 0 or subject_index >= n_subjects:
                raise ValueError(
                    f"subject_index must be in [0, {n_subjects - 1}], got {subject_index}."
                )
            matrix = selected[:, :, subject_index]
    else:
        if subject_index is not None:
            raise ValueError(
                "subject_index was provided, but the selected HCP connectivity scale is not multi-subject."
            )
        matrix = selected

    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    return matrix


def validate_connectivity_matrix(
    matrix: np.ndarray,
    *,
    expected_regions: int | None = None,
    max_regions: int | None = None,
) -> np.ndarray:
    """Validate an empirical structural connectivity matrix."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Connectivity matrix must be square, got shape {matrix.shape}.")
    if expected_regions is not None and matrix.shape[0] != expected_regions:
        raise ValueError(
            f"Empirical SC must be {expected_regions}x{expected_regions}, got {matrix.shape}."
        )
    if max_regions is not None and matrix.shape[0] > max_regions:
        raise ValueError(
            f"Empirical SC must have at most {max_regions} regions, got {matrix.shape[0]}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Connectivity matrix contains non-finite values.")
    if np.any(matrix < 0.0):
        raise ValueError("Connectivity matrix must be non-negative.")

    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)

    if np.isclose(matrix.sum(), 0.0):
        raise ValueError("Connectivity matrix has zero total strength.")
    return matrix


def load_connectivity_matrix(
    path: str | Path,
    *,
    expected_regions: int | None = None,
    max_regions: int | None = None,
    hcp_scale_index: int | None = None,
    hcp_subject_index: int | None = None,
    atlas_count_index: int = 0,
    connectivity_scale: float = DEFAULT_LAUSANNE_CONNECTIVITY_SCALE,
) -> np.ndarray:
    """Load and validate an empirical structural connectivity matrix."""

    path = resolve_input_path(path)
    if path.suffix.lower() == ".mat":
        matrix = load_hcp_connectivity_matrix(
            path,
            max_regions=max_regions,
            scale_index=hcp_scale_index,
            subject_index=hcp_subject_index,
        )
    elif is_lausanne_archive_path(path):
        matrix = load_lausanne_count_connectivity_matrix(
            path,
            atlas_count_index=atlas_count_index,
            connectivity_scale=connectivity_scale,
            expected_regions=expected_regions,
            max_regions=max_regions,
        )
    else:
        matrix, _ = load_numeric_array(path)
        matrix = validate_connectivity_matrix(
            matrix,
            expected_regions=expected_regions,
            max_regions=max_regions,
        )
    return matrix


def _resolve_default_hcp83_path(stem: Path = DEFAULT_HCP83_CONNECTIVITY_STEM) -> Path:
    candidates = [
        stem.with_suffix(suffix)
        for suffix in (".npy", ".npz", ".csv", ".txt", ".mat")
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing HCP Lausanne-83 structural connectivity matrix. Expected one of "
        + ", ".join(str(candidate) for candidate in candidates)
        + ". Do not use local approximation files such as sc90, DTI_fiber_consensus_HCP.csv, "
        "or Lausanne2008-33 count matrices for the paper PhiR reproduction."
    )


def _load_mat_hcp83_matrix(path: Path) -> np.ndarray:
    if loadmat is None:
        raise ImportError(
            "Loading a MATLAB HCP Lausanne-83 matrix requires scipy. "
            "Install scipy or pass .npy/.npz/.csv instead."
        )
    raw = loadmat(path)
    preferred = ("connectivity", "C", "SC", "sc", "hcp_lausanne83_connectivity")
    for key in preferred:
        value = raw.get(key)
        if isinstance(value, np.ndarray) and value.shape == (83, 83):
            return np.asarray(value, dtype=float)
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.shape == (83, 83):
            return np.asarray(value, dtype=float)
    raise ValueError(f"MAT file does not contain an 83x83 HCP Lausanne-83 matrix: {path}")


def load_paper_hcp83_connectivity(path: str | Path | None = None) -> np.ndarray:
    """Load the exact HCP Lausanne-83 matrix required for the paper PhiR path."""

    resolved = _resolve_default_hcp83_path() if path is None else resolve_input_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Missing HCP Lausanne-83 structural connectivity matrix: {resolved}. "
            "Place the paper-preprocessed 83x83 matrix at "
            f"{DEFAULT_HCP83_CONNECTIVITY_STEM}.[npy|npz|csv|txt|mat]. "
            "Do not fall back to sc90, DTI_fiber_consensus_HCP.csv, or Lausanne2008-33 count matrices."
        )

    if resolved.suffix.lower() == ".mat":
        matrix = _load_mat_hcp83_matrix(resolved)
    else:
        matrix, _ = load_numeric_array(resolved)
    matrix = validate_connectivity_matrix(matrix, expected_regions=83, max_regions=83)
    return 0.5 * (matrix + matrix.T)


def _entropy_gaussian_from_cov(covariance: np.ndarray, *, atol: float = 1.0e-10) -> float:
    covariance = np.asarray(covariance, dtype=float)
    dim = covariance.shape[0]
    logdet = _safe_logdet(covariance, atol=atol)
    return 0.5 * (dim * (1.0 + np.log(2.0 * np.pi)) + logdet)


def _official_phiid_mmi_fallback(
    src: np.ndarray,
    trg: np.ndarray,
    *,
    tau: int,
    atol: float = 1.0e-10,
) -> dict[str, np.ndarray]:
    """Small Gaussian-MMI PhiID-compatible fallback matching the official atom keys."""

    src_past, src_future = src[:-tau], src[tau:]
    trg_past, trg_future = trg[:-tau], trg[tau:]
    four = np.column_stack([src_past, trg_past, src_future, trg_future])
    std = four.std(axis=0, ddof=1)
    if np.any(std <= atol) or not np.isfinite(std).all():
        raise ValueError("PhiID input contains a near-constant source or target series.")
    four = four / std
    covariance = np.cov(four, rowvar=False, bias=False)

    def mi(left: Sequence[int], right: Sequence[int]) -> float:
        left = list(left)
        right = list(right)
        joint = left + right
        value = (
            _entropy_gaussian_from_cov(covariance[np.ix_(left, left)], atol=atol)
            + _entropy_gaussian_from_cov(covariance[np.ix_(right, right)], atol=atol)
            - _entropy_gaussian_from_cov(covariance[np.ix_(joint, joint)], atol=atol)
        )
        return max(0.0, float(value))

    i_xta = mi([0], [2])
    i_xtb = mi([0], [3])
    i_yta = mi([1], [2])
    i_ytb = mi([1], [3])
    i_xytab = mi([0, 1], [2, 3])
    phi_wms = i_xytab - i_xta - i_ytb
    rtr = min(i_xta, i_xtb, i_yta, i_ytb)

    atoms = {key: np.asarray([0.0], dtype=float) for key in (
        "rtr", "rtx", "rty", "rts",
        "xtr", "xtx", "xty", "xts",
        "ytr", "ytx", "yty", "yts",
        "str", "stx", "sty", "sts",
    )}
    atoms["rtr"] = np.asarray([rtr], dtype=float)
    atoms["_phi_wms"] = np.asarray([phi_wms], dtype=float)
    return atoms


def calc_official_phiid_atoms(
    src: np.ndarray,
    trg: np.ndarray,
    *,
    tau: int = 1,
    redundancy: str = "MMI",
) -> tuple[dict[str, np.ndarray], str]:
    """Call official phyid when installed, otherwise use the vendored-compatible MMI fallback."""

    if redundancy != "MMI":
        raise ValueError("The paper PhiR reproduction path uses Gaussian MMI PhiID only.")
    src = np.asarray(src, dtype=float)
    trg = np.asarray(trg, dtype=float)
    if src.ndim != 1 or trg.ndim != 1 or src.shape != trg.shape:
        raise ValueError("PhiID source and target must be same-length 1D arrays.")
    if tau <= 0 or src.size <= tau + 2:
        raise ValueError("PhiID source and target are too short for the requested lag.")

    try:
        from phyid.calculate import calc_PhiID  # type: ignore[import-not-found]

        atoms = calc_PhiID(src, trg, tau=tau, kind="gaussian", redundancy=redundancy)
        source = "phyid"
    except ImportError:
        atoms = _official_phiid_mmi_fallback(src, trg, tau=tau)
        source = "internal_gaussian_mmi_compatible_with_phyid"
    return {key: np.asarray(value, dtype=float) for key, value in atoms.items()}, source


def transform_rates_to_bold(
    rates_hz: np.ndarray,
    *,
    dt: float,
    hemodynamic_parameters: BalloonWindkesselParameters = BalloonWindkesselParameters(),
) -> np.ndarray:
    """Transform regional firing rates into BOLD-like signals with a Balloon model."""

    rates = np.asarray(rates_hz, dtype=float)
    if rates.ndim != 2 or rates.shape[0] < 3:
        raise ValueError("rates_hz must have shape (time, regions) with at least 3 samples.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")

    hp = hemodynamic_parameters
    centered = rates - rates.mean(axis=0, keepdims=True)
    neural = hp.neural_gain * centered
    n_steps, n_regions = neural.shape
    signal = np.zeros(n_regions, dtype=float)
    flow = np.ones(n_regions, dtype=float)
    volume = np.ones(n_regions, dtype=float)
    deoxy = np.ones(n_regions, dtype=float)
    bold = np.empty((n_steps, n_regions), dtype=float)

    for step in range(n_steps):
        signal += dt * (neural[step] - signal / hp.tau_s - (flow - 1.0) / hp.tau_f)
        flow = np.clip(flow + dt * signal, 1.0e-6, None)
        extraction = (1.0 - np.power(1.0 - hp.e0, 1.0 / flow)) / hp.e0
        volume += dt * (flow - np.power(volume, 1.0 / hp.alpha)) / hp.tau_0
        volume = np.clip(volume, 1.0e-6, None)
        deoxy += dt * (flow * extraction - deoxy * np.power(volume, 1.0 / hp.alpha - 1.0)) / hp.tau_0
        deoxy = np.clip(deoxy, 1.0e-6, None)
        bold[step] = hp.v0 * (
            hp.k1 * (1.0 - deoxy)
            + hp.k2 * (1.0 - deoxy / volume)
            + hp.k3 * (1.0 - volume)
        )

    return bold


def compute_paper_phi_r_metrics(
    bold_timeseries: np.ndarray,
    *,
    tau: int = 1,
    redundancy: str = "MMI",
    max_pairs: int | None = None,
) -> dict[str, np.ndarray | float | int | str | list[str]]:
    """Compute paper-style pairwise Gaussian-MMI PhiR on BOLD-like time series."""

    bold = np.asarray(bold_timeseries, dtype=float)
    if bold.ndim != 2 or bold.shape[0] <= tau + 2 or bold.shape[1] < 2:
        raise ValueError("bold_timeseries must have shape (time, regions) with enough lagged samples.")

    phi_r_values: list[float] = []
    phi_wms_values: list[float] = []
    rtr_values: list[float] = []
    pair_indices: list[tuple[int, int]] = []
    phiid_source = ""

    pair_counter = 0
    for left in range(bold.shape[1] - 1):
        for right in range(left + 1, bold.shape[1]):
            if max_pairs is not None and pair_counter >= max_pairs:
                break
            src = bold[:, left]
            trg = bold[:, right]
            try:
                atoms, phiid_source = calc_official_phiid_atoms(src, trg, tau=tau, redundancy=redundancy)
            except ValueError:
                continue
            rtr = float(np.ravel(atoms["rtr"])[0])
            if "_phi_wms" in atoms:
                phi_wms = float(np.ravel(atoms["_phi_wms"])[0])
            else:
                covariance = np.cov(
                    np.column_stack([src[:-tau], trg[:-tau], src[tau:], trg[tau:]]),
                    rowvar=False,
                    bias=False,
                )
                phi_wms = (
                    gaussian_mutual_information(covariance, sources=[0, 1], targets=[2, 3])
                    - gaussian_mutual_information(covariance, sources=[0], targets=[2])
                    - gaussian_mutual_information(covariance, sources=[1], targets=[3])
                )
            phi_r = max(0.0, phi_wms + rtr)
            phi_wms_values.append(phi_wms)
            rtr_values.append(rtr)
            phi_r_values.append(phi_r)
            pair_indices.append((left, right))
            pair_counter += 1
        if max_pairs is not None and pair_counter >= max_pairs:
            break

    if not phi_r_values:
        raise ValueError("No valid BOLD region pairs were available for PhiR computation.")

    phi_r_pairwise = np.asarray(phi_r_values, dtype=float)
    phi_wms_pairwise = np.asarray(phi_wms_values, dtype=float)
    rtr_pairwise = np.asarray(rtr_values, dtype=float)
    return {
        "pair_count": int(phi_r_pairwise.size),
        "pair_indices": np.asarray(pair_indices, dtype=int),
        "phi_r_pairwise": phi_r_pairwise,
        "phi_wms_pairwise": phi_wms_pairwise,
        "rtr_pairwise": rtr_pairwise,
        "phi_r_mean": float(np.mean(phi_r_pairwise)),
        "phi_wms_mean": float(np.mean(phi_wms_pairwise)),
        "rtr_mean": float(np.mean(rtr_pairwise)),
        "phiid_source": phiid_source,
        "phiid_source_url": PHIID_SOURCE_URL,
        "phiid_redundancy": redundancy,
        "atom_keys": [
            "rtr", "rtx", "rty", "rts",
            "xtr", "xtx", "xty", "xts",
            "ytr", "ytx", "yty", "yts",
            "str", "stx", "sty", "sts",
        ],
    }


def _regularized_covariance(
    covariance: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, float, float]:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square.")
    sym = 0.5 * (matrix + matrix.T)
    scale = max(float(np.nanmean(np.diag(sym))), 1.0e-12)
    applied_ridge = max(float(ridge), 0.0) * scale
    regularized = sym + applied_ridge * np.eye(sym.shape[0])
    condition_number = float(np.linalg.cond(regularized))
    return regularized, applied_ridge, condition_number


def compute_whole_system_phi_eid_from_gaussian_transition(
    transition_matrix: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    source_covariance: np.ndarray | None = None,
    ridge: float = 0.0,
    log_base: float = np.e,
) -> dict[str, np.ndarray | float]:
    """Compute whole-system Phi^EID for a linear Gaussian transition.

    The source-side intervention is represented by `source_covariance`; by
    default it is the independent standardized maximum-entropy proxy I.
    """

    transition = np.asarray(transition_matrix, dtype=float)
    if transition.ndim != 2:
        raise ValueError("transition_matrix must be 2D.")
    target_dim, source_dim = transition.shape
    if target_dim < 1 or source_dim < 1:
        raise ValueError("transition_matrix must have at least one source and target dimension.")

    if source_covariance is None:
        source_covariance = np.eye(source_dim, dtype=float)
    source_covariance = np.asarray(source_covariance, dtype=float)
    if source_covariance.shape != (source_dim, source_dim):
        raise ValueError(
            f"source_covariance must have shape ({source_dim}, {source_dim}), got {source_covariance.shape}."
        )
    if np.asarray(noise_covariance).shape != (target_dim, target_dim):
        raise ValueError(
            f"noise_covariance must have shape ({target_dim}, {target_dim}), got {np.asarray(noise_covariance).shape}."
        )

    source_cov, source_ridge, source_condition = _regularized_covariance(source_covariance, ridge=ridge)
    noise_cov, noise_ridge, noise_condition = _regularized_covariance(noise_covariance, ridge=ridge)
    target_cov = transition @ source_cov @ transition.T + noise_cov
    target_cov, target_ridge, target_condition = _regularized_covariance(target_cov, ridge=ridge)

    whole_ei = 0.5 * (_safe_logdet(target_cov) - _safe_logdet(noise_cov)) / np.log(log_base)
    whole_ei = max(0.0, float(whole_ei))

    cov_source_target = source_cov @ transition.T
    target_precision = np.linalg.pinv(target_cov)
    conditional_source_cov = source_cov - cov_source_target @ target_precision @ cov_source_target.T
    conditional_source_cov, _, conditional_condition = _regularized_covariance(
        conditional_source_cov,
        ridge=ridge,
    )

    singleton_ei = np.empty(source_dim, dtype=float)
    conditional_variances = np.empty(source_dim, dtype=float)
    for index in range(source_dim):
        var_i = float(source_cov[index, index])
        cov_target_i = transition[:, [index]] * var_i
        conditional_target_cov = target_cov - (cov_target_i @ cov_target_i.T) / max(var_i, 1.0e-12)
        conditional_target_cov, _, _ = _regularized_covariance(conditional_target_cov, ridge=ridge)
        singleton_value = 0.5 * (
            _safe_logdet(target_cov) - _safe_logdet(conditional_target_cov)
        ) / np.log(log_base)
        singleton_ei[index] = max(0.0, float(singleton_value))
        conditional_variances[index] = max(float(conditional_source_cov[index, index]), 1.0e-12)

    raw_phi_eid = float(whole_ei - np.sum(singleton_ei))
    phi_eid = max(0.0, raw_phi_eid)
    conditional_total_correlation = 0.5 * (
        float(np.log(conditional_variances).sum()) - _safe_logdet(conditional_source_cov)
    ) / np.log(log_base)
    conditional_total_correlation = max(0.0, float(conditional_total_correlation))

    return {
        "whole_ei": float(whole_ei),
        "singleton_ei": singleton_ei,
        "singleton_ei_sum": float(np.sum(singleton_ei)),
        "phi_eid": float(phi_eid),
        "raw_phi_eid": raw_phi_eid,
        "conditional_total_correlation": float(conditional_total_correlation),
        "source_ridge": float(source_ridge),
        "noise_ridge": float(noise_ridge),
        "target_ridge": float(target_ridge),
        "source_condition_number": float(source_condition),
        "noise_condition_number": float(noise_condition),
        "target_condition_number": float(target_condition),
        "conditional_source_condition_number": float(conditional_condition),
    }


def estimate_whole_system_phi_eid_from_lagged_samples(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    *,
    ridge: float = 1.0e-6,
    log_base: float = np.e,
) -> dict[str, np.ndarray | float]:
    """Fit a standardized linear Gaussian transition and compute whole-system Phi^EID."""

    source = np.asarray(source_samples, dtype=float)
    target = np.asarray(target_samples, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("source_samples and target_samples must be same-shape 2D arrays.")
    if source.shape[0] <= source.shape[1] + 2:
        raise ValueError("Not enough lagged samples to fit a whole-system transition.")

    source_mean = source.mean(axis=0, keepdims=True)
    source_std = np.maximum(source.std(axis=0, ddof=1, keepdims=True), 1.0e-12)
    target_mean = target.mean(axis=0, keepdims=True)
    target_std = np.maximum(target.std(axis=0, ddof=1, keepdims=True), 1.0e-12)
    source_z = (source - source_mean) / source_std
    target_z = (target - target_mean) / target_std

    coefficient, *_ = np.linalg.lstsq(source_z, target_z, rcond=None)
    transition = coefficient.T
    residual = target_z - source_z @ coefficient
    noise_covariance = np.cov(residual, rowvar=False, bias=False)
    metrics = compute_whole_system_phi_eid_from_gaussian_transition(
        transition,
        noise_covariance,
        source_covariance=np.eye(source.shape[1], dtype=float),
        ridge=ridge,
        log_base=log_base,
    )
    metrics.update(
        {
            "transition_matrix": transition,
            "noise_covariance": noise_covariance,
            "source_dimension": float(source.shape[1]),
            "sample_count": float(source.shape[0]),
        }
    )
    return metrics


def compute_pairwise_phi_metrics_from_lagged_samples(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    *,
    atol: float = 1.0e-10,
    max_pairs: int | None = None,
) -> dict[str, float | int]:
    """Compute pairwise Phi^WMS and Phi^R from explicit lagged source/target samples."""

    source = np.asarray(source_samples, dtype=float)
    target = np.asarray(target_samples, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("source_samples and target_samples must be same-shape 2D arrays.")
    if source.shape[0] < 3 or source.shape[1] < 2:
        raise ValueError("Lagged samples need at least 3 rows and 2 regions.")

    phi_wms_values: list[float] = []
    phi_r_values: list[float] = []
    pair_counter = 0
    for left in range(source.shape[1] - 1):
        for right in range(left + 1, source.shape[1]):
            if max_pairs is not None and pair_counter >= max_pairs:
                break
            lagged = np.column_stack(
                [
                    source[:, left],
                    source[:, right],
                    target[:, left],
                    target[:, right],
                ]
            )
            covariance = np.cov(lagged, rowvar=False, bias=False)
            covariance = 0.5 * (covariance + covariance.T)

            tdmi = gaussian_mutual_information(covariance, sources=[0, 1], targets=[2, 3], atol=atol)
            self_left = gaussian_mutual_information(covariance, sources=[0], targets=[2], atol=atol)
            self_right = gaussian_mutual_information(covariance, sources=[1], targets=[3], atol=atol)
            phi_wms = tdmi - self_left - self_right
            double_redundancy = min(
                gaussian_mutual_information(covariance, sources=[source_index], targets=[target_index], atol=atol)
                for source_index in (0, 1)
                for target_index in (2, 3)
            )
            phi_wms_values.append(float(phi_wms))
            phi_r_values.append(float(max(0.0, phi_wms + double_redundancy)))
            pair_counter += 1
        if max_pairs is not None and pair_counter >= max_pairs:
            break

    return {
        "pair_count": len(phi_wms_values),
        "phi_wms_mean": float(np.mean(phi_wms_values)),
        "phi_r_mean": float(np.mean(phi_r_values)),
        "phi_wms_std": float(np.std(phi_wms_values)),
        "phi_r_std": float(np.std(phi_r_values)),
    }


def bootstrap_pairwise_phi_r(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    *,
    n_bootstrap: int = 32,
    sample_fraction: float = 0.65,
    seed: int = 0,
    max_pairs: int | None = None,
) -> np.ndarray:
    """Bootstrap pairwise PhiR over lagged rows to expose empirical sampling sensitivity."""

    source = np.asarray(source_samples, dtype=float)
    target = np.asarray(target_samples, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("source_samples and target_samples must be same-shape 2D arrays.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1].")

    rng = np.random.default_rng(seed)
    sample_count = max(3, int(round(source.shape[0] * sample_fraction)))
    values = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        row_indices = rng.integers(0, source.shape[0], size=sample_count)
        metrics = compute_pairwise_phi_metrics_from_lagged_samples(
            source[row_indices],
            target[row_indices],
            max_pairs=max_pairs,
        )
        values[index] = float(metrics["phi_r_mean"])
    return values


def compute_average_pairwise_phi_eid_from_lagged_samples(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    *,
    ridge: float = 1.0e-6,
    max_pairs: int | None = None,
) -> dict[str, float | int]:
    """Fallback PEID score: average two-region whole-system Phi^EID across pairs."""

    source = np.asarray(source_samples, dtype=float)
    target = np.asarray(target_samples, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("source_samples and target_samples must be same-shape 2D arrays.")
    if source.shape[1] < 2:
        raise ValueError("At least two regions are required for pairwise Phi^EID.")

    values: list[float] = []
    pair_counter = 0
    for left in range(source.shape[1] - 1):
        for right in range(left + 1, source.shape[1]):
            if max_pairs is not None and pair_counter >= max_pairs:
                break
            pair = [left, right]
            try:
                metrics = estimate_whole_system_phi_eid_from_lagged_samples(
                    source[:, pair],
                    target[:, pair],
                    ridge=ridge,
                )
            except ValueError:
                continue
            values.append(float(metrics["phi_eid"]))
            pair_counter += 1
        if max_pairs is not None and pair_counter >= max_pairs:
            break
    if not values:
        raise ValueError("No valid region pairs were available for pairwise Phi^EID.")
    array = np.asarray(values, dtype=float)
    return {
        "pair_count": int(array.size),
        "phi_eid_mean": float(np.mean(array)),
        "phi_eid_std": float(np.std(array)),
        "phi_eid_min": float(np.min(array)),
        "phi_eid_max": float(np.max(array)),
    }


def load_j_fic_schedule(path: str | Path, *, expected_regions: int | None = None) -> JFICSchedule:
    """Load precomputed J_FIC values for one or more coupling values."""

    values, metadata = load_numeric_array(path)
    if values.ndim == 1:
        if expected_regions is not None and values.shape[0] != expected_regions:
            raise ValueError(
                f"1D J_FIC vector must have {expected_regions} entries, got {values.shape[0]}."
            )
        g_values = None
    elif values.ndim == 2:
        if expected_regions is not None and values.shape[1] != expected_regions:
            raise ValueError(
                "2D J_FIC schedule must have shape "
                f"(n_g, {expected_regions}), got {values.shape}."
            )
        raw_g = metadata.get("G", metadata.get("g_values"))
        g_values = None if raw_g is None else np.asarray(raw_g, dtype=float)
        if g_values is not None and g_values.ndim != 1:
            raise ValueError("Saved J_FIC G values must be a 1D array.")
        if g_values is not None and g_values.shape[0] != values.shape[0]:
            raise ValueError(
                "Saved J_FIC schedule has mismatched G and J_FIC lengths: "
                f"{g_values.shape[0]} vs {values.shape[0]}."
            )
    else:
        raise ValueError(f"J_FIC array must be 1D or 2D, got shape {values.shape}.")

    if not np.isfinite(values).all():
        raise ValueError("J_FIC values contain non-finite numbers.")
    return JFICSchedule(values=np.asarray(values, dtype=float), g_values=g_values)


def resolve_j_fic_vector(
    schedule: JFICSchedule,
    *,
    coupling_g: float,
    index: int,
    expected_regions: int | None = None,
) -> np.ndarray:
    """Select the J_FIC vector for one sweep point."""

    values = np.asarray(schedule.values, dtype=float)
    if values.ndim == 1:
        return values.copy()

    if schedule.g_values is not None:
        matches = np.flatnonzero(np.isclose(schedule.g_values, coupling_g, atol=1.0e-8, rtol=1.0e-6))
        if matches.size == 0:
            raise ValueError(
                f"No J_FIC row in the schedule matches G={coupling_g:.6f}. "
                "Save `G` alongside `j_fic` or align the sweep values."
            )
        row = int(matches[0])
    else:
        if index >= values.shape[0]:
            raise ValueError(
                "J_FIC schedule has fewer rows than requested G values: "
                f"{values.shape[0]} rows for index {index}."
            )
        row = index

    vector = np.asarray(values[row], dtype=float)
    if expected_regions is not None and vector.shape != (expected_regions,):
        raise ValueError(
            f"Resolved J_FIC row must have shape ({expected_regions},), got {vector.shape}."
    )
    return vector


def detect_stabilization_step(
    mean_rate_full_trace_hz: np.ndarray,
    *,
    dt: float,
    min_burn_in: float,
    stabilization_parameters: StabilizationParameters = StabilizationParameters(),
) -> dict[str, float | int | bool]:
    """Detect the first stable averaging window from the mean firing-rate trace."""

    trace = np.asarray(mean_rate_full_trace_hz, dtype=float)
    if trace.ndim != 1 or trace.size == 0:
        raise ValueError("mean_rate_full_trace_hz must be a non-empty 1D array.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")

    window_steps = max(1, int(round(stabilization_parameters.window / dt)))
    confirm_windows = max(1, int(stabilization_parameters.confirm_windows))

    min_step = int(round(min_burn_in / dt))
    min_step = min(max(min_step, 0), trace.size - 1)

    available = trace.size - min_step
    if available < 2 * window_steps:
        return {
            "detected": False,
            "start_step": min_step,
            "start_time_s": min_step * dt,
            "window_steps": window_steps,
            "window_time_s": window_steps * dt,
            "drift_hz": np.nan,
        }

    window_starts = list(range(min_step, trace.size - window_steps + 1, window_steps))
    window_means = np.asarray(
        [trace[start : start + window_steps].mean() for start in window_starts],
        dtype=float,
    )

    stable_run = 0
    for diff_index in range(window_means.size - 1):
        drift_hz = float(abs(window_means[diff_index + 1] - window_means[diff_index]))
        if drift_hz <= stabilization_parameters.tolerance_hz:
            stable_run += 1
            if stable_run >= confirm_windows:
                first_diff_index = diff_index - confirm_windows + 1
                stable_window_index = first_diff_index + 1
                start_step = int(window_starts[stable_window_index])
                return {
                    "detected": True,
                    "start_step": start_step,
                    "start_time_s": start_step * dt,
                    "window_steps": window_steps,
                    "window_time_s": window_steps * dt,
                    "drift_hz": drift_hz,
                }
        else:
            stable_run = 0

    return {
        "detected": False,
        "start_step": min_step,
        "start_time_s": min_step * dt,
        "window_steps": window_steps,
        "window_time_s": window_steps * dt,
        "drift_hz": np.nan,
    }


def simulate_dmf(
    connectivity: np.ndarray,
    coupling_g: float,
    j_fic: np.ndarray,
    parameters: DMFParameters = DMFParameters(),
    stabilization_parameters: StabilizationParameters = StabilizationParameters(),
    *,
    seed: int = 0,
    initial_se: np.ndarray | None = None,
    initial_si: np.ndarray | None = None,
    record_rate_trace: bool = False,
) -> dict[str, np.ndarray | float]:
    """Simulate the E-I DMF dynamics for one global coupling value."""

    n_regions = connectivity.shape[0]
    j_fic = np.asarray(j_fic, dtype=float)
    if j_fic.shape != (n_regions,):
        raise ValueError(f"J_FIC vector must have shape ({n_regions},), got {j_fic.shape}.")

    dt = parameters.dt
    n_steps = int(round(parameters.t_total / dt))
    burn_steps = int(round(parameters.burn_in / dt))

    if burn_steps >= n_steps:
        raise ValueError("burn_in must be shorter than t_total.")

    rng = np.random.default_rng(seed)

    se = (
        np.full(n_regions, parameters.init_se, dtype=float)
        if initial_se is None
        else np.asarray(initial_se, dtype=float).copy()
    )
    si = (
        np.full(n_regions, parameters.init_si, dtype=float)
        if initial_si is None
        else np.asarray(initial_si, dtype=float).copy()
    )

    rate_history = np.empty((n_steps, n_regions), dtype=float)

    for step in range(n_steps):
        input_e = (
            parameters.w_e * parameters.i0
            + parameters.w_plus * parameters.j_nmda * se
            + coupling_g * parameters.j_nmda * (connectivity @ se)
            - j_fic * si
        )
        input_i = parameters.w_i * parameters.i0 + parameters.j_nmda * se - si

        rate_e = transfer_function(
            input_e,
            gain=parameters.gain_e,
            threshold=parameters.threshold_e,
            shape=parameters.shape_e,
        )
        rate_i = transfer_function(
            input_i,
            gain=parameters.gain_i,
            threshold=parameters.threshold_i,
            shape=parameters.shape_i,
        )

        noise_e = parameters.sigma * np.sqrt(dt) * rng.standard_normal(n_regions)
        noise_i = parameters.sigma * np.sqrt(dt) * rng.standard_normal(n_regions)

        dse = dt * (-se / parameters.tau_e + (1.0 - se) * parameters.gamma_e * rate_e) + noise_e
        dsi = dt * (-si / parameters.tau_i + rate_i) + noise_i

        se = np.clip(se + dse, 0.0, 1.0)
        si = np.clip(si + dsi, 0.0, 1.0)

        rate_history[step] = rate_e

    mean_rate_full_trace_hz = rate_history.mean(axis=1)
    stabilization = detect_stabilization_step(
        mean_rate_full_trace_hz,
        dt=dt,
        min_burn_in=parameters.burn_in,
        stabilization_parameters=stabilization_parameters,
    )
    stats_start_step = int(stabilization["start_step"])
    stats_rates = rate_history[stats_start_step:]
    mean_rate_trace = mean_rate_full_trace_hz[stats_start_step:]
    mean_region_rate_hz = stats_rates.mean(axis=0)

    result: dict[str, np.ndarray | float | bool] = {
        "mean_rate_trace_hz": mean_rate_trace,
        "mean_rate_hz": float(mean_rate_trace.mean()),
        "mean_region_rate_hz": mean_region_rate_hz,
        "final_se": se,
        "final_si": si,
        "stabilization_detected": bool(stabilization["detected"]),
        "stabilization_start_step": float(stats_start_step),
        "stabilization_start_time_s": float(stabilization["start_time_s"]),
        "stabilization_window_time_s": float(stabilization["window_time_s"]),
        "stabilization_last_drift_hz": float(stabilization["drift_hz"]),
    }
    if record_rate_trace:
        result["time_s"] = np.arange(n_steps, dtype=float) * dt
        result["mean_rate_full_trace_hz"] = mean_rate_full_trace_hz
        result["region_rate_trace_hz"] = rate_history
    return result


def calibrate_j_fic(
    connectivity: np.ndarray,
    coupling_g: float,
    parameters: DMFParameters = DMFParameters(),
    fic_parameters: FICParameters = FICParameters(),
    stabilization_parameters: StabilizationParameters = StabilizationParameters(),
    *,
    seed: int = 0,
    initial_j_fic: np.ndarray | None = None,
    initial_se: np.ndarray | None = None,
    initial_si: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | bool]:
    """Calibrate one J_FIC value per region to approach the target firing rate."""

    n_regions = connectivity.shape[0]
    if initial_j_fic is None:
        j_fic = np.full(n_regions, fic_parameters.initial_j, dtype=float)
    else:
        j_fic = np.asarray(initial_j_fic, dtype=float).copy()
        if j_fic.shape != (n_regions,):
            raise ValueError(
                f"Initial J_FIC vector must have shape ({n_regions},), got {j_fic.shape}."
            )

    j_fic = np.clip(j_fic, fic_parameters.min_j, fic_parameters.max_j)
    calibration_parameters = replace(parameters, sigma=fic_parameters.calibration_sigma)

    state_se = None if initial_se is None else np.asarray(initial_se, dtype=float).copy()
    state_si = None if initial_si is None else np.asarray(initial_si, dtype=float).copy()

    best_payload: dict[str, np.ndarray | float | bool] | None = None
    best_max_abs_error = np.inf

    for iteration in range(1, fic_parameters.max_iterations + 1):
        result = simulate_dmf(
            connectivity=connectivity,
            coupling_g=coupling_g,
            j_fic=j_fic,
            parameters=calibration_parameters,
            stabilization_parameters=stabilization_parameters,
            seed=seed,
            initial_se=state_se,
            initial_si=state_si,
        )

        mean_region_rate_hz = np.asarray(result["mean_region_rate_hz"], dtype=float)
        rate_error = mean_region_rate_hz - fic_parameters.target_rate_hz
        max_abs_error = float(np.max(np.abs(rate_error)))

        payload: dict[str, np.ndarray | float | bool] = {
            "j_fic": j_fic.copy(),
            "mean_region_rate_hz": mean_region_rate_hz.copy(),
            "max_abs_rate_error_hz": max_abs_error,
            "iterations": float(iteration),
            "converged": max_abs_error <= fic_parameters.tolerance_hz,
            "final_se": np.asarray(result["final_se"], dtype=float).copy(),
            "final_si": np.asarray(result["final_si"], dtype=float).copy(),
        }
        if max_abs_error < best_max_abs_error:
            best_max_abs_error = max_abs_error
            best_payload = payload

        if max_abs_error <= fic_parameters.tolerance_hz:
            return payload

        j_fic = np.clip(
            j_fic + fic_parameters.learning_rate * rate_error,
            fic_parameters.min_j,
            fic_parameters.max_j,
        )

        if fic_parameters.warm_start_state:
            state_se = np.asarray(result["final_se"], dtype=float).copy()
            state_si = np.asarray(result["final_si"], dtype=float).copy()

    if best_payload is None:
        raise RuntimeError("J_FIC calibration did not produce any candidate solution.")
    return best_payload


def sweep_global_coupling(
    g_values: Iterable[float],
    connectivity: np.ndarray,
    parameters: DMFParameters = DMFParameters(),
    fic_parameters: FICParameters = FICParameters(),
    stabilization_parameters: StabilizationParameters = StabilizationParameters(),
    *,
    j_fic_path: str | Path | None = None,
    j_fic_reference_g: float | None = None,
    expected_regions: int | None = None,
    max_regions: int | None = None,
    continuation: bool = True,
    compute_phi: bool = True,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sweep the global coupling G with empirical SC and per-region J_FIC.

    By default, a single fixed J_FIC vector is calibrated once and then reused
    across the full G sweep so that G is the only parameter varied.
    """

    connectivity = validate_connectivity_matrix(
        connectivity,
        expected_regions=expected_regions,
        max_regions=max_regions,
    )
    g_values = np.asarray(list(g_values), dtype=float)
    if g_values.ndim != 1 or g_values.size == 0:
        raise ValueError("g_values must be a non-empty 1D iterable.")

    n_regions = connectivity.shape[0]
    j_fic_schedule = (
        None
        if j_fic_path is None
        else load_j_fic_schedule(j_fic_path, expected_regions=n_regions)
    )

    mean_rates = np.empty_like(g_values)
    phi_wms = np.full(g_values.shape, np.nan, dtype=float)
    phi_r = np.full(g_values.shape, np.nan, dtype=float)
    pair_count = np.zeros(g_values.shape, dtype=int)
    mean_region_rates = np.empty((g_values.size, n_regions), dtype=float)
    j_fic_values = np.empty((g_values.size, n_regions), dtype=float)
    j_fic_converged = np.zeros(g_values.size, dtype=bool)
    j_fic_max_abs_error = np.full(g_values.size, np.nan, dtype=float)
    j_fic_iterations = np.zeros(g_values.size, dtype=int)
    stabilization_detected = np.zeros(g_values.size, dtype=bool)
    stabilization_start_s = np.full(g_values.size, np.nan, dtype=float)

    initial_se = None
    initial_si = None
    fixed_j_fic = None
    fixed_j_fic_reference_g = np.nan
    fixed_j_fic_converged = True
    fixed_j_fic_error = np.nan
    fixed_j_fic_iterations = 0

    if j_fic_schedule is None:
        fixed_j_fic_reference_g = float(g_values[0] if j_fic_reference_g is None else j_fic_reference_g)
        calibration = calibrate_j_fic(
            connectivity=connectivity,
            coupling_g=fixed_j_fic_reference_g,
            parameters=parameters,
            fic_parameters=fic_parameters,
            stabilization_parameters=stabilization_parameters,
            seed=seed,
        )
        fixed_j_fic = np.asarray(calibration["j_fic"], dtype=float)
        fixed_j_fic_converged = bool(calibration["converged"])
        fixed_j_fic_error = float(calibration["max_abs_rate_error_hz"])
        fixed_j_fic_iterations = int(float(calibration["iterations"]))

        if continuation and np.isclose(fixed_j_fic_reference_g, g_values[0]):
            initial_se = np.asarray(calibration["final_se"], dtype=float).copy()
            initial_si = np.asarray(calibration["final_si"], dtype=float).copy()
    elif j_fic_schedule.values.ndim == 1:
        fixed_j_fic = np.asarray(j_fic_schedule.values, dtype=float).copy()

    for index, coupling_g in enumerate(g_values):
        if fixed_j_fic is not None:
            j_fic = fixed_j_fic.copy()
            j_fic_converged[index] = fixed_j_fic_converged
            j_fic_max_abs_error[index] = fixed_j_fic_error
            j_fic_iterations[index] = fixed_j_fic_iterations
        elif j_fic_schedule is not None:
            j_fic = resolve_j_fic_vector(
                j_fic_schedule,
                coupling_g=float(coupling_g),
                index=index,
                expected_regions=n_regions,
            )
            j_fic_converged[index] = True

        result = simulate_dmf(
            connectivity=connectivity,
            coupling_g=float(coupling_g),
            j_fic=j_fic,
            parameters=parameters,
            stabilization_parameters=stabilization_parameters,
            seed=seed + index,
            initial_se=initial_se if continuation else None,
            initial_si=initial_si if continuation else None,
            record_rate_trace=compute_phi,
        )

        mean_rates[index] = float(result["mean_rate_hz"])
        mean_region_rates[index] = np.asarray(result["mean_region_rate_hz"], dtype=float)
        if compute_phi:
            start_step = int(float(result["stabilization_start_step"]))
            rates = np.asarray(result["region_rate_trace_hz"], dtype=float)[start_step:]
            metrics = compute_pairwise_phi_metrics(rates)
            phi_wms[index] = float(metrics["phi_wms_mean"])
            phi_r[index] = float(metrics["phi_r_mean"])
            pair_count[index] = int(metrics["pair_count"])
        j_fic_values[index] = j_fic
        stabilization_detected[index] = bool(result["stabilization_detected"])
        stabilization_start_s[index] = float(result["stabilization_start_time_s"])

        if continuation:
            initial_se = np.asarray(result["final_se"], dtype=float)
            initial_si = np.asarray(result["final_si"], dtype=float)

    derivative = np.gradient(mean_rates, g_values)
    critical_index = int(np.argmax(derivative))

    return {
        "G": g_values,
        "mean_rate_hz": mean_rates,
        "phi_wms": phi_wms,
        "phi_r": phi_r,
        "pair_count": pair_count,
        "mean_region_rate_hz": mean_region_rates,
        "d_rate_dG": derivative,
        "critical_G": np.asarray([g_values[critical_index]], dtype=float),
        "connectivity": connectivity,
        "node_strength": connectivity.sum(axis=1),
        "j_fic": j_fic_values,
        "j_fic_calibration_converged": j_fic_converged,
        "j_fic_calibration_max_abs_error_hz": j_fic_max_abs_error,
        "j_fic_calibration_iterations": j_fic_iterations,
        "j_fic_reference_G": np.asarray([fixed_j_fic_reference_g], dtype=float),
        "stabilization_detected": stabilization_detected,
        "stabilization_start_s": stabilization_start_s,
    }


def plot_fig6b_like(
    sweep_result: dict[str, np.ndarray],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a Fig. 6B-like mean firing rate and integrated-information curve."""

    g_values = sweep_result["G"]
    mean_rate_hz = sweep_result["mean_rate_hz"]
    critical_g = float(sweep_result["critical_G"][0])
    has_phi = "phi_wms" in sweep_result and np.isfinite(sweep_result["phi_wms"]).any()

    if has_phi:
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(6.2, 6.4),
            constrained_layout=True,
            sharex=True,
        )
        axis = axes[0]
    else:
        figure, axis = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        axes = (axis,)

    axis.plot(g_values, mean_rate_hz, color="0.30", lw=1.4, zorder=1)
    axis.scatter(g_values, mean_rate_hz, color="black", s=16, zorder=2)
    axis.set_xlim(g_values.min(), g_values.max())
    axis.set_ylim(0.0, max(20.0, float(mean_rate_hz.max()) * 1.08))
    axis.set_ylabel("Mean firing rate (Hz)")
    axis.set_title("DMF Reproduction of Fig. 6B (Empirical SC + J_FIC)")
    axis.grid(True, color="0.85", lw=0.8)

    axis.text(
        0.98,
        0.05,
        f"max d(rate)/dG near G = {critical_g:.2f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.25"},
    )

    if has_phi:
        phi_axis = axes[1]
        phi_wms = np.asarray(sweep_result["phi_wms"], dtype=float)
        phi_r = np.asarray(sweep_result["phi_r"], dtype=float)
        phi_axis.plot(g_values, phi_wms, color="#2f80c1", lw=1.0)
        phi_axis.scatter(
            g_values,
            phi_wms,
            color="#2f80c1",
            s=14,
            label=r"$\Phi^{WMS}$",
        )
        phi_axis.plot(g_values, phi_r, color="#df2b2b", lw=1.0)
        phi_axis.scatter(
            g_values,
            phi_r,
            color="#df2b2b",
            s=14,
            label=r"$\Phi^R$",
        )
        phi_axis.set_ylabel("Integrated information")
        phi_axis.grid(True, color="0.85", lw=0.8)
        phi_axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        axes[0].text(-0.12, 1.02, "A", transform=axes[0].transAxes, fontsize=14, fontweight="bold")
        axes[1].text(-0.12, 1.02, "B", transform=axes[1].transAxes, fontsize=14, fontweight="bold")

    axes[-1].set_xlabel("Global coupling $G$")

    if output_path is not None:
        output_path = resolve_output_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=220, bbox_inches="tight")

    return figure


def plot_paper_phi_r_like(
    sweep_result: dict[str, np.ndarray],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the paper-focused Fig. 6B reproduction with mean rate and PhiR only."""

    g_values = np.asarray(sweep_result["G"], dtype=float)
    mean_rate_hz = np.asarray(sweep_result["mean_rate_hz"], dtype=float)
    phi_r = np.asarray(sweep_result["phi_r_mean"], dtype=float)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.2, 5.8),
        constrained_layout=True,
        sharex=True,
    )
    rate_axis, phi_axis = axes
    rate_axis.plot(g_values, mean_rate_hz, color="0.25", lw=1.3, zorder=1)
    rate_axis.scatter(g_values, mean_rate_hz, color="black", s=15, zorder=2)
    rate_axis.set_ylabel("Mean firing rate (Hz)")
    rate_axis.set_ylim(0.0, max(20.0, float(np.nanmax(mean_rate_hz)) * 1.08))
    rate_axis.grid(True, color="0.85", lw=0.8)

    phi_axis.plot(g_values, phi_r, color="#df2b2b", lw=1.2, label=r"$\Phi^R$")
    phi_axis.scatter(g_values, phi_r, color="#df2b2b", s=15)
    phi_axis.set_ylabel(r"$\Phi^R$")
    phi_axis.set_xlabel("Global coupling $G$")
    phi_axis.set_ylim(bottom=0.0)
    phi_axis.grid(True, color="0.85", lw=0.8)
    phi_axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    if output_path is not None:
        output_path = resolve_output_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
    return figure


def reproduce_fig6b_paper_phi_r(
    *,
    connectivity: np.ndarray | None = None,
    connectivity_path: str | Path | None = None,
    g_values: Iterable[float] | None = None,
    seed: int = 0,
    continuation: bool = True,
    parameters: DMFParameters = DMFParameters(),
    fic_parameters: FICParameters = FICParameters(),
    stabilization_parameters: StabilizationParameters = StabilizationParameters(),
    hemodynamic_parameters: BalloonWindkesselParameters = BalloonWindkesselParameters(),
    expected_regions: int | None = 83,
    max_regions: int | None = 83,
    phi_tau: int = 1,
    max_phi_pairs: int | None = None,
    results_path: str | Path | None = None,
    figure_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Reproduce the paper Fig. 6B PhiR curve using HCP83, BOLD, and Gaussian MMI PhiID."""

    if connectivity is not None and connectivity_path is not None:
        raise ValueError("Pass either `connectivity` or `connectivity_path`, not both.")
    if connectivity is None:
        connectivity = load_paper_hcp83_connectivity(connectivity_path)
    else:
        connectivity = validate_connectivity_matrix(
            connectivity,
            expected_regions=expected_regions,
            max_regions=max_regions,
        )
        connectivity = 0.5 * (connectivity + connectivity.T)

    if g_values is None:
        g_values = np.linspace(1.0, 3.0, 41)
    g_values = np.asarray(list(g_values), dtype=float)
    if g_values.ndim != 1 or g_values.size == 0:
        raise ValueError("g_values must be a non-empty 1D iterable.")

    n_regions = connectivity.shape[0]
    mean_rates = np.empty(g_values.shape, dtype=float)
    phi_r_mean = np.empty(g_values.shape, dtype=float)
    phi_wms_mean = np.empty(g_values.shape, dtype=float)
    rtr_mean = np.empty(g_values.shape, dtype=float)
    pair_count = np.empty(g_values.shape, dtype=int)
    mean_region_rates = np.empty((g_values.size, n_regions), dtype=float)
    j_fic_values = np.empty((g_values.size, n_regions), dtype=float)
    calibration_errors = np.empty(g_values.shape, dtype=float)
    calibration_iterations = np.empty(g_values.shape, dtype=int)
    calibration_converged = np.empty(g_values.shape, dtype=bool)

    bold_series: list[np.ndarray] = []
    phi_r_pairwise: list[np.ndarray] = []
    phi_wms_pairwise: list[np.ndarray] = []
    phi_pair_indices: np.ndarray | None = None
    phiid_source = ""

    initial_j_fic = None
    initial_se = None
    initial_si = None
    for index, coupling_g in enumerate(g_values):
        calibration = calibrate_j_fic(
            connectivity=connectivity,
            coupling_g=float(coupling_g),
            parameters=parameters,
            fic_parameters=fic_parameters,
            stabilization_parameters=stabilization_parameters,
            seed=seed + index,
            initial_j_fic=initial_j_fic,
            initial_se=initial_se if continuation else None,
            initial_si=initial_si if continuation else None,
        )
        j_fic = np.asarray(calibration["j_fic"], dtype=float)
        simulation = simulate_dmf(
            connectivity=connectivity,
            coupling_g=float(coupling_g),
            j_fic=j_fic,
            parameters=parameters,
            stabilization_parameters=stabilization_parameters,
            seed=seed + index,
            initial_se=np.asarray(calibration["final_se"], dtype=float) if continuation else None,
            initial_si=np.asarray(calibration["final_si"], dtype=float) if continuation else None,
            record_rate_trace=True,
        )
        stats_start_step = int(float(simulation["stabilization_start_step"]))
        rates = np.asarray(simulation["region_rate_trace_hz"], dtype=float)[stats_start_step:]
        bold = transform_rates_to_bold(
            rates,
            dt=parameters.dt,
            hemodynamic_parameters=hemodynamic_parameters,
        )
        phi_metrics = compute_paper_phi_r_metrics(
            bold,
            tau=phi_tau,
            max_pairs=max_phi_pairs,
        )

        mean_rates[index] = float(simulation["mean_rate_hz"])
        mean_region_rates[index] = np.asarray(simulation["mean_region_rate_hz"], dtype=float)
        phi_r_mean[index] = float(phi_metrics["phi_r_mean"])
        phi_wms_mean[index] = float(phi_metrics["phi_wms_mean"])
        rtr_mean[index] = float(phi_metrics["rtr_mean"])
        pair_count[index] = int(phi_metrics["pair_count"])
        j_fic_values[index] = j_fic
        calibration_errors[index] = float(calibration["max_abs_rate_error_hz"])
        calibration_iterations[index] = int(float(calibration["iterations"]))
        calibration_converged[index] = bool(calibration["converged"])
        bold_series.append(bold)
        phi_r_pairwise.append(np.asarray(phi_metrics["phi_r_pairwise"], dtype=float))
        phi_wms_pairwise.append(np.asarray(phi_metrics["phi_wms_pairwise"], dtype=float))
        if phi_pair_indices is None:
            phi_pair_indices = np.asarray(phi_metrics["pair_indices"], dtype=int)
        phiid_source = str(phi_metrics["phiid_source"])

        if continuation:
            initial_j_fic = j_fic
            initial_se = np.asarray(simulation["final_se"], dtype=float)
            initial_si = np.asarray(simulation["final_si"], dtype=float)

    derivative = np.gradient(mean_rates, g_values)
    critical_index = int(np.argmax(derivative))
    metadata = {
        "pipeline": "paper_phi_r_hcp83_bold_gaussian_mmi",
        "phiid_source": phiid_source,
        "phiid_source_url": PHIID_SOURCE_URL,
        "phiid_redundancy": "MMI",
        "phi_tau": int(phi_tau),
        "max_phi_pairs": None if max_phi_pairs is None else int(max_phi_pairs),
        "connectivity_required": "HCP 900 preprocessed Lausanne-83; no local approximation fallback",
    }
    max_pair_count = max(values.size for values in phi_r_pairwise)
    phi_r_pairwise_matrix = np.full((len(phi_r_pairwise), max_pair_count), np.nan, dtype=float)
    phi_wms_pairwise_matrix = np.full((len(phi_wms_pairwise), max_pair_count), np.nan, dtype=float)
    for row, values in enumerate(phi_r_pairwise):
        phi_r_pairwise_matrix[row, : values.size] = values
    for row, values in enumerate(phi_wms_pairwise):
        phi_wms_pairwise_matrix[row, : values.size] = values
    result = {
        "G": g_values,
        "mean_rate_hz": mean_rates,
        "mean_region_rate_hz": mean_region_rates,
        "phi_r_mean": phi_r_mean,
        "phi_wms_mean": phi_wms_mean,
        "rtr_mean": rtr_mean,
        "phi_r_pairwise": phi_r_pairwise_matrix,
        "phi_wms_pairwise": phi_wms_pairwise_matrix,
        "phi_pair_indices": np.empty((0, 2), dtype=int) if phi_pair_indices is None else phi_pair_indices,
        "pair_count": pair_count,
        "bold_timeseries": np.stack(bold_series, axis=0),
        "j_fic": j_fic_values,
        "j_fic_calibration_converged": calibration_converged,
        "j_fic_calibration_max_abs_error_hz": calibration_errors,
        "j_fic_calibration_iterations": calibration_iterations,
        "d_rate_dG": derivative,
        "critical_G": np.asarray([g_values[critical_index]], dtype=float),
        "connectivity": connectivity,
        "node_strength": connectivity.sum(axis=1),
        "metadata": np.asarray(json.dumps(metadata, ensure_ascii=True)),
    }

    if results_path is not None:
        results_path = resolve_output_path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(results_path, **result)
    if figure_path is not None:
        figure = plot_paper_phi_r_like(result, output_path=figure_path)
        plt.close(figure)
    return result


def plot_region_rate_traces(
    simulation_result: dict[str, np.ndarray | float],
    *,
    coupling_g: float,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the excitatory firing-rate trace for every region over time."""

    if "time_s" not in simulation_result or "region_rate_trace_hz" not in simulation_result:
        raise ValueError(
            "simulation_result must include `time_s` and `region_rate_trace_hz`. "
            "Call simulate_dmf(..., record_rate_trace=True)."
        )

    time_s = np.asarray(simulation_result["time_s"], dtype=float)
    region_rate_trace_hz = np.asarray(simulation_result["region_rate_trace_hz"], dtype=float)
    mean_rate_full_trace_hz = np.asarray(simulation_result["mean_rate_full_trace_hz"], dtype=float)
    stabilization_start_time_s = float(simulation_result["stabilization_start_time_s"])
    stabilization_detected = bool(simulation_result["stabilization_detected"])

    figure, (axis_lines, axis_heatmap) = plt.subplots(
        2,
        1,
        figsize=(8.4, 7.2),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.3]},
    )

    axis_lines.plot(time_s, region_rate_trace_hz, color="#4c78a8", alpha=0.16, lw=0.7)
    axis_lines.plot(
        time_s,
        mean_rate_full_trace_hz,
        color="black",
        lw=2.0,
        label="Mean across regions",
        zorder=3,
    )
    axis_lines.set_ylabel("Excitatory rate (Hz)")
    axis_lines.set_title(f"Region-wise excitatory firing rates (G = {coupling_g:.2f})")
    axis_lines.grid(True, color="0.88", lw=0.8)
    axis_lines.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    stability_label = "stable window detected" if stabilization_detected else "fallback to minimum burn-in"
    axis_lines.text(
        0.99,
        0.03,
        f"{stability_label}; stats start = {stabilization_start_time_s:.3f} s",
        transform=axis_lines.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.25"},
    )

    heatmap = axis_heatmap.imshow(
        region_rate_trace_hz.T,
        aspect="auto",
        origin="lower",
        extent=[time_s[0], time_s[-1], 0, region_rate_trace_hz.shape[1] - 1],
        cmap="viridis",
    )
    axis_heatmap.set_xlabel("Time (s)")
    axis_heatmap.set_ylabel("Region index")
    colorbar = figure.colorbar(heatmap, ax=axis_heatmap, pad=0.02)
    colorbar.set_label("Excitatory rate (Hz)")

    if output_path is not None:
        output_path = resolve_output_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200)

    return figure


def reproduce_fig6b_mean_rate_transition(
    *,
    connectivity: np.ndarray | None = None,
    connectivity_path: str | Path | None = None,
    j_fic_path: str | Path | None = None,
    g_values: Iterable[float] | None = None,
    j_fic_reference_g: float | None = None,
    seed: int = 0,
    continuation: bool = True,
    compute_phi: bool = True,
    parameters: DMFParameters = DMFParameters(),
    fic_parameters: FICParameters = FICParameters(),
    stabilization_parameters: StabilizationParameters = StabilizationParameters(),
    expected_regions: int | None = None,
    max_regions: int | None = 100,
    hcp_scale_index: int | None = None,
    hcp_subject_index: int | None = None,
    connectivity_scale: float = DEFAULT_LAUSANNE_CONNECTIVITY_SCALE,
    figure_path: str | Path | None = None,
    results_path: str | Path | None = None,
    trace_coupling_g: float | None = None,
    trace_figure_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """High-level helper for reproducing the mean-rate transition in Fig. 6B.

    If `j_fic_path` is omitted, a single fixed J_FIC vector is calibrated and
    reused for the full sweep.
    """

    if connectivity is not None and connectivity_path is not None:
        raise ValueError("Pass either `connectivity` or `connectivity_path`, not both.")
    if connectivity is None and connectivity_path is None:
        raise ValueError(
            "An empirical structural connectivity matrix is required. "
            "Pass `connectivity` or `connectivity_path`."
        )

    if connectivity_path is not None:
        connectivity = load_connectivity_matrix(
            connectivity_path,
            expected_regions=expected_regions,
            max_regions=max_regions,
            hcp_scale_index=hcp_scale_index,
            hcp_subject_index=hcp_subject_index,
            connectivity_scale=connectivity_scale,
        )
    else:
        connectivity = validate_connectivity_matrix(
            connectivity,
            expected_regions=expected_regions,
            max_regions=max_regions,
        )

    if g_values is None:
        g_values = np.linspace(1.0, 3.0, 21)

    sweep_result = sweep_global_coupling(
        g_values=g_values,
        connectivity=connectivity,
        parameters=parameters,
        fic_parameters=fic_parameters,
        stabilization_parameters=stabilization_parameters,
        j_fic_path=j_fic_path,
        j_fic_reference_g=j_fic_reference_g,
        expected_regions=expected_regions,
        max_regions=max_regions,
        continuation=continuation,
        compute_phi=compute_phi,
        seed=seed,
    )

    if results_path is not None:
        results_path = resolve_output_path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            results_path,
            G=sweep_result["G"],
            mean_rate_hz=sweep_result["mean_rate_hz"],
            phi_wms=sweep_result["phi_wms"],
            phi_r=sweep_result["phi_r"],
            pair_count=sweep_result["pair_count"],
            mean_region_rate_hz=sweep_result["mean_region_rate_hz"],
            d_rate_dG=sweep_result["d_rate_dG"],
            critical_G=sweep_result["critical_G"],
            connectivity=sweep_result["connectivity"],
            node_strength=sweep_result["node_strength"],
            j_fic=sweep_result["j_fic"],
            j_fic_calibration_converged=sweep_result["j_fic_calibration_converged"],
            j_fic_calibration_max_abs_error_hz=sweep_result["j_fic_calibration_max_abs_error_hz"],
            j_fic_calibration_iterations=sweep_result["j_fic_calibration_iterations"],
            j_fic_reference_G=sweep_result["j_fic_reference_G"],
            stabilization_detected=sweep_result["stabilization_detected"],
            stabilization_start_s=sweep_result["stabilization_start_s"],
        )

    if figure_path is not None:
        figure = plot_fig6b_like(sweep_result, output_path=figure_path)
        plt.close(figure)

    if trace_figure_path is not None:
        target_trace_g = float(sweep_result["critical_G"][0] if trace_coupling_g is None else trace_coupling_g)
        trace_index = int(np.argmin(np.abs(sweep_result["G"] - target_trace_g)))
        trace_g = float(sweep_result["G"][trace_index])
        trace_result = simulate_dmf(
            connectivity=connectivity,
            coupling_g=trace_g,
            j_fic=np.asarray(sweep_result["j_fic"][trace_index], dtype=float),
            parameters=parameters,
            stabilization_parameters=stabilization_parameters,
            seed=seed + trace_index,
            record_rate_trace=True,
        )
        trace_figure = plot_region_rate_traces(
            trace_result,
            coupling_g=trace_g,
            output_path=trace_figure_path,
        )
        plt.close(trace_figure)

    return sweep_result


def run_lausanne_count_batch(
    atlas_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    j_fic_path: str | Path | None = None,
    g_values: Iterable[float] | None = None,
    j_fic_reference_g: float | None = None,
    seed: int = 0,
    continuation: bool = True,
    compute_phi: bool = True,
    parameters: DMFParameters = DMFParameters(),
    fic_parameters: FICParameters = FICParameters(),
    stabilization_parameters: StabilizationParameters = StabilizationParameters(),
    expected_regions: int | None = None,
    max_regions: int | None = 100,
    connectivity_scale: float = DEFAULT_LAUSANNE_CONNECTIVITY_SCALE,
    trace_coupling_g: float | None = None,
) -> dict[str, object]:
    """Run the Fig. 6B workflow for every Lausanne `count` matrix in the atlas pickle."""

    atlas_path = resolve_input_path(atlas_path)
    entries = load_lausanne_atlas_entries(atlas_path)
    row_labels = load_lausanne_region_labels(atlas_path)

    if output_dir is None:
        resolved_output_dir = atlas_path.parent / "result"
    else:
        resolved_output_dir = resolve_output_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        connectivity = prepare_lausanne_count_connectivity(
            entry["count"],
            region_labels=row_labels,
            connectivity_scale=connectivity_scale,
            expected_regions=expected_regions,
            max_regions=max_regions,
        )
        kept_labels = [label for label in row_labels if label.casefold() != "unknown"]
        stem = f"count_{index:02d}"
        connectivity_csv = save_labeled_connectivity_csv(
            resolved_output_dir / f"{stem}_connectivity.csv",
            connectivity,
            row_labels=kept_labels,
        )
        figure_path = resolved_output_dir / f"{stem}_fig6b_mean_rate.png"
        results_path = resolved_output_dir / f"{stem}_fig6b_mean_rate.npz"
        trace_figure_path = (
            resolved_output_dir / f"{stem}_rate_traces.png"
            if trace_coupling_g is not None
            else None
        )

        sweep_result = reproduce_fig6b_mean_rate_transition(
            connectivity=connectivity,
            j_fic_path=j_fic_path,
            g_values=g_values,
            j_fic_reference_g=j_fic_reference_g,
            seed=seed + index,
            continuation=continuation,
            compute_phi=compute_phi,
            parameters=parameters,
            fic_parameters=fic_parameters,
            stabilization_parameters=stabilization_parameters,
            expected_regions=expected_regions,
            max_regions=max_regions,
            connectivity_scale=connectivity_scale,
            figure_path=figure_path,
            results_path=results_path,
            trace_coupling_g=trace_coupling_g,
            trace_figure_path=trace_figure_path,
        )

        calibration_error = np.asarray(
            sweep_result["j_fic_calibration_max_abs_error_hz"],
            dtype=float,
        )
        summary_rows.append(
            {
                "matrix_index": index,
                "critical_G": float(sweep_result["critical_G"][0]),
                "max_mean_rate_hz": float(np.max(sweep_result["mean_rate_hz"])),
                "stabilization_detected_any": bool(np.any(sweep_result["stabilization_detected"])),
                "earliest_stabilization_start_s": (
                    float(np.nanmin(sweep_result["stabilization_start_s"]))
                    if np.isfinite(sweep_result["stabilization_start_s"]).any()
                    else np.nan
                ),
                "max_j_fic_error_hz": (
                    float(np.nanmax(calibration_error))
                    if np.isfinite(calibration_error).any()
                    else np.nan
                ),
                "connectivity_csv": str(connectivity_csv),
                "results_npz": str(results_path),
                "mean_rate_figure": str(figure_path),
                "trace_figure": "" if trace_figure_path is None else str(trace_figure_path),
            }
        )
        print(
            f"[{index + 1}/{len(entries)}] Saved batch outputs for `{stem}` to "
            f"{resolved_output_dir}"
        )

    summary_path = resolved_output_dir / "batch_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "matrix_index",
                "critical_G",
                "max_mean_rate_hz",
                "stabilization_detected_any",
                "earliest_stabilization_start_s",
                "max_j_fic_error_hz",
                "connectivity_csv",
                "results_npz",
                "mean_rate_figure",
                "trace_figure",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "atlas_path": atlas_path,
        "output_dir": resolved_output_dir,
        "summary_path": summary_path,
        "rows": summary_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Fig. 6B mean firing rate transition with empirical SC and J_FIC."
    )
    default_connectivity = DEFAULT_LAUSANNE_ATLAS_PATH
    parser.add_argument(
        "--connectivity",
        type=Path,
        default=default_connectivity,
        help=(
            "Path to the empirical structural connectivity matrix "
            "(.mat/.npy/.npz/.csv/.txt/.pkl/.pkl.gz). Defaults to the full "
            "Lausanne2008-33 zip archive under `data/`, so the script "
            "batch-runs all 19 `count` matrices. Each `count` matrix is cropped "
            "to 83x83 by removing the `Unknown` parcel and normalized by its "
            "global maximum before simulation. Batch outputs are written to "
            "`exp/brain/result`. "
            "For CSV inputs, the first column stores Lausanne-83 region names and "
            "the remaining 83x83 block stores the structural connectivity matrix. "
            "Relative paths are resolved from this script's directory."
        ),
    )
    parser.add_argument(
        "--paper-phi-r",
        action="store_true",
        help=(
            "Run the paper-focused HCP Lausanne-83 + BOLD + Gaussian MMI PhiID PhiR "
            "reproduction path instead of the existing Lausanne count/proxy path."
        ),
    )
    parser.add_argument(
        "--paper-connectivity",
        type=Path,
        default=None,
        help=(
            "Optional exact HCP Lausanne-83 83x83 connectivity matrix for --paper-phi-r. "
            "If omitted, the script looks for data/external/hcp_lausanne83_connectivity.*."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help=(
            "Batch-output directory used when `--connectivity` points to the full "
            "Lausanne atlas `.zip`/`.pkl`/`.pkl.gz`. Defaults to `exp/brain/result`."
        ),
    )
    parser.add_argument(
        "--j-fic",
        type=Path,
        default=None,
        help=(
            "Optional path to a saved J_FIC vector or schedule. "
            "A 1D vector is reused for every G; a 2D array is treated as a "
            "per-G schedule. If omitted, one fixed J_FIC vector is calibrated."
        ),
    )
    parser.add_argument(
        "--j-fic-reference-g",
        type=float,
        default=None,
        help=(
            "Coupling value used to calibrate the fixed J_FIC vector when "
            "`--j-fic` is omitted. Defaults to the first G in the sweep."
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=SCRIPT_DIR / "fig6b_mean_rate.png",
        help="Where to save the output figure.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=SCRIPT_DIR / "fig6b_mean_rate.npz",
        help="Where to save the numerical sweep results.",
    )
    parser.add_argument(
        "--trace-figure",
        type=Path,
        default=SCRIPT_DIR / "fig6b_rate_traces.png",
        help=(
            "Where to save the region-wise excitatory firing-rate trace figure. "
            "Use an empty value in Python calls to disable it."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--t-total",
        type=float,
        default=DMFParameters().t_total,
        help="Total simulation time (s) for each coupling value.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DMFParameters().dt,
        help="Euler integration step size (s).",
    )
    parser.add_argument(
        "--burn-in",
        type=float,
        default=DMFParameters().burn_in,
        help=(
            "Minimum warm-up time (s) before stabilization checks begin. "
            "Statistics start at the first detected stable window after this time."
        ),
    )
    parser.add_argument(
        "--expected-regions",
        type=int,
        default=None,
        help="Optional exact region count to enforce for the empirical SC.",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=100,
        help="Maximum allowed region count for the empirical SC. Defaults to 100.",
    )
    parser.add_argument(
        "--connectivity-scale",
        type=float,
        default=DEFAULT_LAUSANNE_CONNECTIVITY_SCALE,
        help=(
            "Scale applied after max-normalizing Lausanne count matrices. "
            "The default 0.2 matches the paper-scale Fig. 6B firing-rate range; "
            "use 1.0 to recover the raw max-normalized connectivity."
        ),
    )
    parser.add_argument(
        "--hcp-scale-index",
        type=int,
        default=None,
        help=(
            "For HCP `.mat` connectivity, optionally force a specific scale index "
            "instead of auto-selecting the largest scale within `--max-regions`."
        ),
    )
    parser.add_argument(
        "--hcp-subject-index",
        type=int,
        default=None,
        help=(
            "For HCP `.mat` connectivity, optionally load one subject-specific SC. "
            "If omitted, the script averages the selected HCP scale across subjects."
        ),
    )
    parser.add_argument(
        "--g-min",
        type=float,
        default=1.0,
        help="Minimum global coupling value.",
    )
    parser.add_argument(
        "--g-max",
        type=float,
        default=3.0,
        help="Maximum global coupling value.",
    )
    parser.add_argument(
        "--g-count",
        type=int,
        default=21,
        help="Number of coupling values to evaluate.",
    )
    parser.add_argument(
        "--phi-tau",
        type=int,
        default=1,
        help="Lag, in BOLD samples, used for Gaussian MMI PhiID in --paper-phi-r mode.",
    )
    parser.add_argument(
        "--max-phi-pairs",
        type=int,
        default=None,
        help="Optional limit on region pairs for quick --paper-phi-r pilot runs.",
    )
    parser.add_argument(
        "--trace-g",
        type=float,
        default=None,
        help=(
            "Optional coupling value used for the region-wise rate trace figure. "
            "Defaults to the detected critical G; the nearest swept G is used."
        ),
    )
    parser.add_argument(
        "--stabilization-window",
        type=float,
        default=StabilizationParameters().window,
        help="Window size (s) used to detect stabilization from the mean firing-rate trace.",
    )
    parser.add_argument(
        "--stabilization-tolerance",
        type=float,
        default=StabilizationParameters().tolerance_hz,
        help="Maximum allowed mean-rate drift (Hz) between adjacent windows to mark the trace as stable.",
    )
    parser.add_argument(
        "--stabilization-confirm-windows",
        type=int,
        default=StabilizationParameters().confirm_windows,
        help="Number of consecutive stable window comparisons required before statistics begin.",
    )
    parser.add_argument(
        "--j-fic-target-rate",
        type=float,
        default=3.0,
        help="Target post-burn mean excitatory rate per region during J_FIC calibration.",
    )
    parser.add_argument(
        "--j-fic-tolerance",
        type=float,
        default=0.05,
        help="Maximum allowed absolute regional rate error (Hz) for J_FIC calibration.",
    )
    parser.add_argument(
        "--j-fic-max-iters",
        type=int,
        default=12,
        help="Maximum number of J_FIC calibration iterations per G.",
    )
    parser.add_argument(
        "--j-fic-learning-rate",
        type=float,
        default=0.025,
        help="Step size used to update J_FIC from regional rate errors.",
    )
    parser.add_argument(
        "--j-fic-calibration-sigma",
        type=float,
        default=0.0,
        help="Noise level used during J_FIC calibration. Use 0 for stable deterministic fitting.",
    )
    parser.add_argument(
        "--independent-restarts",
        action="store_true",
        help="Restart each G from the same initial state instead of continuation.",
    )
    parser.add_argument(
        "--skip-phi",
        action="store_true",
        help="Skip pairwise integrated-information proxy metrics and draw only firing rates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parameters = replace(
        DMFParameters(),
        t_total=args.t_total,
        burn_in=args.burn_in,
        dt=args.dt,
    )
    fic_parameters = FICParameters(
        target_rate_hz=args.j_fic_target_rate,
        tolerance_hz=args.j_fic_tolerance,
        max_iterations=args.j_fic_max_iters,
        learning_rate=args.j_fic_learning_rate,
        calibration_sigma=args.j_fic_calibration_sigma,
    )
    stabilization_parameters = StabilizationParameters(
        window=args.stabilization_window,
        tolerance_hz=args.stabilization_tolerance,
        confirm_windows=args.stabilization_confirm_windows,
    )
    g_count = 41 if args.paper_phi_r and args.g_count == 21 else args.g_count
    g_values = np.linspace(args.g_min, args.g_max, g_count)
    connectivity_path = resolve_input_path(args.connectivity)

    if args.paper_phi_r:
        sweep_result = reproduce_fig6b_paper_phi_r(
            connectivity_path=args.paper_connectivity,
            g_values=g_values,
            seed=args.seed,
            continuation=not args.independent_restarts,
            parameters=parameters,
            fic_parameters=fic_parameters,
            stabilization_parameters=stabilization_parameters,
            expected_regions=83,
            max_regions=83,
            phi_tau=args.phi_tau,
            max_phi_pairs=args.max_phi_pairs,
            figure_path=args.figure,
            results_path=args.results,
        )
        print(f"Saved paper PhiR figure to: {args.figure}")
        print(f"Saved paper PhiR numerical results to: {args.results}")
        print(f"Connectivity regions: {sweep_result['connectivity'].shape[0]}")
        print(f"Estimated transition point: G ~ {float(sweep_result['critical_G'][0]):.3f}")
        print(
            "Mean PhiR values: "
            + ", ".join(
                f"G={g:.2f}:{value:.4g}"
                for g, value in zip(sweep_result["G"], sweep_result["phi_r_mean"])
            )
        )
        return

    if is_lausanne_archive_path(connectivity_path):
        batch_result = run_lausanne_count_batch(
            atlas_path=connectivity_path,
            output_dir=args.output_dir,
            j_fic_path=args.j_fic,
            g_values=g_values,
            j_fic_reference_g=args.j_fic_reference_g,
            seed=args.seed,
            continuation=not args.independent_restarts,
            compute_phi=not args.skip_phi,
            parameters=parameters,
            fic_parameters=fic_parameters,
            stabilization_parameters=stabilization_parameters,
            expected_regions=args.expected_regions,
            max_regions=args.max_regions,
            connectivity_scale=args.connectivity_scale,
            trace_coupling_g=args.trace_g,
        )
        print(f"Processed atlas batch: {batch_result['atlas_path']}")
        print(f"Saved batch outputs to: {batch_result['output_dir']}")
        print(f"Saved batch summary to: {batch_result['summary_path']}")
        print(f"Processed matrices: {len(batch_result['rows'])}")
        return

    sweep_result = reproduce_fig6b_mean_rate_transition(
        connectivity_path=connectivity_path,
        j_fic_path=args.j_fic,
        g_values=g_values,
        j_fic_reference_g=args.j_fic_reference_g,
        seed=args.seed,
        continuation=not args.independent_restarts,
        compute_phi=not args.skip_phi,
        parameters=parameters,
        fic_parameters=fic_parameters,
        stabilization_parameters=stabilization_parameters,
        expected_regions=args.expected_regions,
        max_regions=args.max_regions,
        hcp_scale_index=args.hcp_scale_index,
        hcp_subject_index=args.hcp_subject_index,
        connectivity_scale=args.connectivity_scale,
        figure_path=args.figure,
        results_path=args.results,
        trace_coupling_g=args.trace_g,
        trace_figure_path=args.trace_figure,
    )

    critical_g = float(sweep_result["critical_G"][0])
    mean_rates = sweep_result["mean_rate_hz"]
    calibration_error = sweep_result["j_fic_calibration_max_abs_error_hz"]

    print(f"Saved figure to: {args.figure}")
    print(f"Saved numerical results to: {args.results}")
    print(f"Saved region-rate trace figure to: {args.trace_figure}")
    print(f"Connectivity regions: {sweep_result['connectivity'].shape[0]}")
    print(f"Estimated transition point: G ~ {critical_g:.3f}")
    if sweep_result["stabilization_detected"].any():
        first_detected = float(np.nanmin(sweep_result["stabilization_start_s"]))
        print(f"Earliest detected stabilization start: t ~ {first_detected:.3f} s")
    if np.isfinite(calibration_error).any():
        print(
            "Worst J_FIC regional fitting error (Hz): "
            f"{np.nanmax(calibration_error):.3f}"
        )
    print(
        "Mean firing rates (Hz): "
        + ", ".join(
            f"G={g:.2f}:{rate:.2f}" for g, rate in zip(sweep_result["G"], mean_rates)
        )
    )


if __name__ == "__main__":
    main()
