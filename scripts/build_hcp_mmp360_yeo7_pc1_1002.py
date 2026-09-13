#!/usr/bin/env python3
"""Build Yeo7-PC1 task time series for score-complete HCP subjects.

The reduction follows the repository's frozen task-evoked convention:

1. fit one PC1 per Yeo7 network on ``taskRetained - taskRegressed``;
2. use only the first 75% of time points for PCA fitting; and
3. project the complete ``taskRetained`` signal with the fitted PCA models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.io import loadmat, savemat
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "hcp_s1200_all_available_task_lr_mmp360_schaefer100_200_feat_timeseries"
)
DEFAULT_BEHAVIOR = ROOT / "data" / "unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "hcp_s1200_1002_complete_scores_task_lr_mmp360_yeo7_pc1_timeseries"
)
MAPPING_URL = (
    "https://raw.githubusercontent.com/Jingfeng-Tang/"
    "HCP360-Glasser-to-Yeo-7-Network-Mapping/main/"
    "HCP360_to_Yeo7_Mapping.csv"
)
MAPPING_SHA256 = "b7188bffe67b52d3cd9c4939e22ea3bd6c91221d5cb97c3ea841af459bad5179"
PIPELINE_VERSION = "hcp_mmp360_yeo7_task_evoked_pc1_v1"
TASKS = (
    "EMOTION_LR",
    "GAMBLING_LR",
    "LANGUAGE_LR",
    "MOTOR_LR",
    "RELATIONAL_LR",
    "SOCIAL_LR",
    "WM_LR",
)
EXPECTED_FRAMES = {
    "EMOTION_LR": 176,
    "GAMBLING_LR": 253,
    "LANGUAGE_LR": 316,
    "MOTOR_LR": 284,
    "RELATIONAL_LR": 232,
    "SOCIAL_LR": 274,
    "WM_LR": 405,
}
NETWORK_NAMES = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)
EXPECTED_NETWORK_COUNTS = {1: 62, 2: 52, 3: 46, 4: 50, 5: 24, 6: 44, 7: 82}

# These are the exact inputs used by the repository's frozen seven-task endpoint
# contract. Social d-prime needs the full response table to infer trial counts.
SCORE_FIELDS = {
    "EMOTION": ("Emotion_Task_Face_Median_RT", "Emotion_Task_Shape_Median_RT"),
    "GAMBLING": ("DDisc_AUC_200", "DDisc_AUC_40K"),
    "LANGUAGE": ("Language_Task_Math_Avg_Difficulty_Level",),
    "MOTOR": ("Endurance_AgeAdj", "Dexterity_AgeAdj", "Strength_AgeAdj"),
    "RELATIONAL": ("Relational_Task_Acc",),
    "SOCIAL": (
        "Social_Task_TOM_Perc_Random",
        "Social_Task_TOM_Perc_TOM",
        "Social_Task_TOM_Perc_Unsure",
        "Social_Task_TOM_Perc_NLR",
        "Social_Task_Random_Perc_Random",
        "Social_Task_Random_Perc_TOM",
        "Social_Task_Random_Perc_Unsure",
        "Social_Task_Random_Perc_NLR",
        "Social_Task_Perc_Random",
        "Social_Task_Perc_TOM",
        "Social_Task_Perc_Unsure",
        "Social_Task_Perc_NLR",
    ),
    "WM": ("WM_Task_Acc",),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def development_end_for_length(length: int) -> int:
    """Match the existing repository's round-half-up 75% split."""
    return int(np.floor(0.75 * int(length) + 0.5))


def discover_complete_imaging_subjects(input_root: Path) -> list[str]:
    expected = set(TASKS)
    subjects = []
    for folder in sorted(Path(input_root).glob("sub-*")):
        if folder.is_dir() and {path.stem for path in folder.glob("*.mat")} == expected:
            subjects.append(folder.name.removeprefix("sub-"))
    return subjects


def _finite_number(value: str | None) -> bool:
    if value is None or value.strip() in {"", "NA", "NaN", "nan"}:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except ValueError:
        return False


def infer_social_trials(row: Mapping[str, str]) -> int:
    fields = SCORE_FIELDS["SOCIAL"]
    condition = np.asarray([float(row[field]) for field in fields[:8]]) / 100.0
    overall = np.asarray([float(row[field]) for field in fields[8:]]) / 100.0
    for trials in range(5, 101, 5):
        condition_error = np.max(np.abs(condition * trials - np.round(condition * trials)))
        overall_error = np.max(np.abs(overall * 2 * trials - np.round(overall * 2 * trials)))
        if max(float(condition_error), float(overall_error)) < 0.011:
            return trials
    raise ValueError(f"Could not infer SOCIAL trial count for subject {row['Subject']}")


def select_score_complete_subjects(
    imaging_subjects: Sequence[str], behavior_path: Path
) -> tuple[list[str], dict[str, Any]]:
    with Path(behavior_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = {str(row["Subject"]).removeprefix("sub-"): row for row in csv.DictReader(handle)}

    included: list[str] = []
    missing_by_task: dict[str, list[str]] = {task: [] for task in SCORE_FIELDS}
    missing_rows: list[str] = []
    for subject in imaging_subjects:
        row = rows.get(subject)
        if row is None:
            missing_rows.append(subject)
            continue
        complete = True
        for task, fields in SCORE_FIELDS.items():
            if not all(_finite_number(row.get(field)) for field in fields):
                missing_by_task[task].append(subject)
                complete = False
        if complete:
            emotion_rt = [float(row[field]) for field in SCORE_FIELDS["EMOTION"]]
            if any(value <= 0 for value in emotion_rt):
                missing_by_task["EMOTION"].append(subject)
                complete = False
            try:
                infer_social_trials(row)
            except ValueError:
                missing_by_task["SOCIAL"].append(subject)
                complete = False
        if complete:
            included.append(subject)

    excluded = sorted(set(imaging_subjects).difference(included))
    selected_rows = [rows[subject] for subject in included]
    component_sds = {
        task: {
            field: float(
                np.std([float(row[field]) for row in selected_rows], ddof=1)
            )
            for field in SCORE_FIELDS[task]
        }
        for task in ("GAMBLING", "MOTOR")
    }
    if any(
        not np.isfinite(value) or value <= 1.0e-12
        for task in component_sds.values()
        for value in task.values()
    ):
        raise ValueError("A composite-score component is constant or non-finite.")
    audit = {
        "behavior_path": str(Path(behavior_path).resolve()),
        "complete_imaging_subjects": len(imaging_subjects),
        "score_complete_subjects": len(included),
        "excluded_subjects": excluded,
        "missing_behavior_rows": missing_rows,
        "missing_by_task": missing_by_task,
        "score_fields": {key: list(value) for key, value in SCORE_FIELDS.items()},
        "derived_endpoint_inputs_valid": True,
        "composite_component_standard_deviations": component_sds,
    }
    return included, audit


def fetch_mapping() -> tuple[bytes, np.ndarray, np.ndarray, dict[str, list[int]]]:
    with urllib.request.urlopen(MAPPING_URL, timeout=30) as response:
        payload = response.read()
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != MAPPING_SHA256:
        raise ValueError(
            f"Mapping SHA-256 mismatch: {actual_sha256} != {MAPPING_SHA256}"
        )
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    ids = np.asarray([int(row["HCP_ID"]) for row in rows], dtype=np.int16)
    networks = np.asarray([int(row["Yeo_Network"]) for row in rows], dtype=np.int8)
    confidence = np.asarray([float(row["Confidence"]) for row in rows], dtype=np.float32)
    if not np.array_equal(ids, np.arange(1, 361)):
        raise ValueError("Mapping must contain consecutive HCP IDs 1..360.")
    counts = {index: int(np.count_nonzero(networks == index)) for index in range(1, 8)}
    if counts != EXPECTED_NETWORK_COUNTS:
        raise ValueError(f"Unexpected Yeo7 network counts: {counts}")
    groups = {
        name: np.flatnonzero(networks == index).astype(int).tolist()
        for index, name in enumerate(NETWORK_NAMES, start=1)
    }
    return payload, networks, confidence, groups


def reduce_task(
    input_path: Path,
    output_path: Path,
    groups: Mapping[str, Sequence[int]],
    mapping_networks: np.ndarray,
    mapping_confidence: np.ndarray,
    source_manifest: Mapping[tuple[str, str], Mapping[str, str]],
) -> dict[str, Any]:
    subject = input_path.parent.name.removeprefix("sub-")
    task = input_path.stem
    expected_frames = EXPECTED_FRAMES[task]
    payload = loadmat(
        input_path,
        variable_names=["MMP360_taskRetained", "MMP360_taskRegressed"],
        verify_compressed_data_integrity=True,
    )
    retained = np.asarray(payload["MMP360_taskRetained"], dtype=float)
    regressed = np.asarray(payload["MMP360_taskRegressed"], dtype=float)
    expected_shape = (expected_frames, 360)
    if retained.shape != expected_shape or regressed.shape != expected_shape:
        raise ValueError(
            f"{input_path}: expected paired {expected_shape}, got "
            f"{retained.shape} and {regressed.shape}"
        )
    if not np.isfinite(retained).all() or not np.isfinite(regressed).all():
        raise ValueError(f"{input_path}: non-finite input values")

    fitting = retained - regressed
    development_end = development_end_for_length(expected_frames)
    series = np.empty((expected_frames, 7), dtype=np.float32)
    explained = np.empty(7, dtype=np.float64)
    loadings = np.zeros((360, 7), dtype=np.float32)
    fitting_mean = np.empty(360, dtype=np.float32)
    for network_index, (network, indices) in enumerate(groups.items()):
        selected = np.asarray(indices, dtype=int)
        model = PCA(n_components=1, svd_solver="full").fit(
            fitting[:development_end, selected]
        )
        series[:, network_index] = model.transform(retained[:, selected])[:, 0]
        explained[network_index] = float(model.explained_variance_ratio_[0])
        loadings[selected, network_index] = model.components_[0].astype(np.float32)
        fitting_mean[selected] = model.mean_.astype(np.float32)

    if not np.isfinite(series).all() or not np.isfinite(explained).all():
        raise ValueError(f"{input_path}: PCA produced non-finite output")
    if np.any(np.std(series, axis=0, ddof=1) <= 1.0e-12):
        raise ValueError(f"{input_path}: PCA produced a constant network series")
    reconstructed = (retained.astype(np.float32) - fitting_mean) @ loadings
    reconstruction_error = float(np.max(np.abs(reconstructed - series)))
    if reconstruction_error > 1.0e-2:
        raise ValueError(
            f"{input_path}: stored PCA parameters reconstruct with max error "
            f"{reconstruction_error:.6g}"
        )

    manifest_row = source_manifest[(subject, task)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, prefix=output_path.stem + ".", suffix=".mat", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        savemat(
            temporary,
            {
                "Yeo7_taskRetainedPC1": series,
                "Yeo7_network_names": np.asarray(NETWORK_NAMES, dtype=object),
                "Yeo7_pc1_explained_variance_ratio": explained,
                "MMP360_pc1_loadings": loadings,
                "MMP360_pca_fitting_mean": fitting_mean,
                "MMP360_to_Yeo7_network": mapping_networks.astype(np.int8),
                "MMP360_to_Yeo7_confidence": mapping_confidence.astype(np.float32),
                "development_end": np.asarray(development_end, dtype=np.int32),
                "pipeline_version": PIPELINE_VERSION,
                "source_mat_sha256": manifest_row["mat_sha256"],
            },
            do_compression=True,
            oned_as="row",
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "subject": subject,
        "task": task.removesuffix("_LR"),
        "direction": "LR",
        "status": "ok",
        "source_mat": str(input_path.resolve()),
        "source_mat_bytes": int(input_path.stat().st_size),
        "source_mat_sha256": manifest_row["mat_sha256"],
        "output_mat": str(output_path.resolve()),
        "output_mat_bytes": int(output_path.stat().st_size),
        "output_mat_sha256": sha256_file(output_path),
        "frames": expected_frames,
        "dimensions": 7,
        "development_end": development_end,
        "mean_pc1_explained_variance_ratio": float(explained.mean()),
        "min_pc1_explained_variance_ratio": float(explained.min()),
        "max_pc1_explained_variance_ratio": float(explained.max()),
        "stored_parameter_reconstruction_max_abs_error": reconstruction_error,
    }


def load_source_manifest(input_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = Path(input_root) / "logs" / "manifest_all.tsv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    manifest = {
        (str(row["subject"]), f"{row['task']}_{row['direction']}"): row for row in rows
    }
    if len(manifest) != len(rows):
        raise ValueError(f"Duplicate keys in {path}")
    return manifest


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", mode="w", encoding="utf-8", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", mode="w", newline="", encoding="utf-8", delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_outputs(output_root: Path, subjects: Sequence[str]) -> dict[str, Any]:
    errors: list[str] = []
    files = sorted(Path(output_root).glob("sub-*/*.mat"))
    expected_count = len(subjects) * len(TASKS)
    if len(files) != expected_count:
        errors.append(f"expected {expected_count} MAT files, found {len(files)}")
    subject_set = {f"sub-{subject}" for subject in subjects}
    unexpected_subjects = sorted({path.parent.name for path in files}.difference(subject_set))
    if unexpected_subjects:
        errors.append(f"unexpected output subjects: {unexpected_subjects}")
    for path in files:
        task = path.stem
        try:
            payload = loadmat(
                path,
                variable_names=[
                    "Yeo7_taskRetainedPC1",
                    "Yeo7_pc1_explained_variance_ratio",
                    "MMP360_pc1_loadings",
                    "MMP360_pca_fitting_mean",
                    "MMP360_to_Yeo7_network",
                    "MMP360_to_Yeo7_confidence",
                ],
                verify_compressed_data_integrity=True,
            )
            series = np.asarray(payload["Yeo7_taskRetainedPC1"])
            explained = np.asarray(payload["Yeo7_pc1_explained_variance_ratio"])
            loadings = np.asarray(payload["MMP360_pc1_loadings"])
            fitting_mean = np.asarray(payload["MMP360_pca_fitting_mean"])
            if series.shape != (EXPECTED_FRAMES[task], 7) or not np.isfinite(series).all():
                errors.append(f"{path}: invalid series {series.shape}")
            if explained.size != 7 or not np.isfinite(explained).all():
                errors.append(f"{path}: invalid explained variance")
            if loadings.shape != (360, 7) or not np.isfinite(loadings).all():
                errors.append(f"{path}: invalid loadings {loadings.shape}")
            if fitting_mean.size != 360 or not np.isfinite(fitting_mean).all():
                errors.append(f"{path}: invalid PCA fitting mean")
        except Exception as error:  # validation must aggregate all failures
            errors.append(f"{path}: {type(error).__name__}: {error}")
    return {
        "status": "ok" if not errors else "error",
        "pipeline_version": PIPELINE_VERSION,
        "subjects": len(subjects),
        "tasks_per_subject": len(TASKS),
        "mat_files": len(files),
        "expected_mat_files": expected_count,
        "all_shapes_valid": not errors,
        "all_values_finite": not errors,
        "errors": errors,
    }


def write_readme(output_root: Path, score_audit: Mapping[str, Any]) -> None:
    text = f"""# HCP MMP360 → Yeo7 PC1 task time series

This directory contains {score_audit['score_complete_subjects']} subjects with all seven LR task runs and complete inputs for the repository's frozen seven-task behavioral endpoint contract.

Each `sub-*/TASK_LR.mat` contains `Yeo7_taskRetainedPC1` with shape `[time, 7]`. Columns follow: `{', '.join(NETWORK_NAMES)}`.

PCA is fitted separately for every subject, task, and network on the first 75% of `MMP360_taskRetained - MMP360_taskRegressed`, using full-SVD PC1. The fitted models are then applied to the complete `MMP360_taskRetained` signal. `MMP360_pc1_loadings`, `MMP360_pca_fitting_mean`, and `Yeo7_pc1_explained_variance_ratio` preserve the fitted reductions.

The MMP360→Yeo7 assignment is a maximum-volume-overlap mapping downloaded from:
{MAPPING_URL}

Its frozen SHA-256 is `{MAPPING_SHA256}`. Mapping confidence is retained in every MAT file and in `HCP360_to_Yeo7_Mapping.csv`.

See `score_completeness.json`, `manifest.tsv`, and `validation_report.json` for audit details.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--behavior", type=Path, default=DEFAULT_BEHAVIOR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only: process the first N subjects.")
    parser.add_argument("--expected-imaging-subjects", type=int, default=1009)
    parser.add_argument("--expected-score-complete-subjects", type=int, default=1002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imaging_subjects = discover_complete_imaging_subjects(args.input_root)
    subjects, score_audit = select_score_complete_subjects(imaging_subjects, args.behavior)
    if len(imaging_subjects) != args.expected_imaging_subjects:
        raise ValueError(
            f"Expected {args.expected_imaging_subjects} complete imaging subjects, "
            f"found {len(imaging_subjects)}"
        )
    if len(subjects) != args.expected_score_complete_subjects:
        raise ValueError(
            f"Expected {args.expected_score_complete_subjects} score-complete subjects, "
            f"found {len(subjects)}"
        )
    if args.limit is not None:
        subjects = subjects[: args.limit]
        score_audit = dict(score_audit)
        score_audit["processed_subjects_smoke_test"] = len(subjects)

    mapping_payload, mapping_networks, mapping_confidence, groups = fetch_mapping()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "HCP360_to_Yeo7_Mapping.csv").write_bytes(mapping_payload)
    (output_root / "subjects.txt").write_text("".join(f"{subject}\n" for subject in subjects), encoding="utf-8")
    atomic_json(output_root / "score_completeness.json", score_audit)
    source_manifest = load_source_manifest(args.input_root)

    jobs = [
        (
            Path(args.input_root) / f"sub-{subject}" / f"{task}.mat",
            output_root / f"sub-{subject}" / f"{task}.mat",
        )
        for subject in subjects
        for task in TASKS
    ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                reduce_task,
                input_path,
                output_path,
                groups,
                mapping_networks,
                mapping_confidence,
                source_manifest,
            ): (input_path, output_path)
            for input_path, output_path in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 100 == 0 or completed == len(futures):
                print(f"[{completed}/{len(futures)}] task files complete", flush=True)

    rows.sort(key=lambda row: (row["subject"], row["task"]))
    write_manifest(output_root / "manifest.tsv", rows)
    validation = validate_outputs(output_root, subjects)
    validation.update(
        {
            "mapping_url": MAPPING_URL,
            "mapping_sha256": MAPPING_SHA256,
            "mapping_network_counts": EXPECTED_NETWORK_COUNTS,
            "mapping_confidence_min": float(mapping_confidence.min()),
            "mapping_confidence_median": float(np.median(mapping_confidence)),
            "mapping_confidence_below_0_6": int(np.count_nonzero(mapping_confidence < 0.6)),
            "mean_pc1_explained_variance_ratio": float(
                np.mean([row["mean_pc1_explained_variance_ratio"] for row in rows])
            ),
            "stored_parameter_reconstruction_max_abs_error": float(
                max(row["stored_parameter_reconstruction_max_abs_error"] for row in rows)
            ),
            "output_bytes": int(sum(row["output_mat_bytes"] for row in rows)),
        }
    )
    atomic_json(output_root / "validation_report.json", validation)
    write_readme(output_root, score_audit)
    if validation["status"] != "ok":
        raise RuntimeError(f"Output validation failed: {validation['errors'][:10]}")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
