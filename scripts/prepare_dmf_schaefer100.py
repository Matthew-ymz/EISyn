#!/usr/bin/env python3
"""Prepare the 93-subject Schaefer100 group connectome for DMF experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_dmf_83_region_oracle_phi_eid import load_dmf_module, resolve_path


DEFAULT_INPUT = ROOT / "data" / "neuromodulator_receptor_sc_100"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "dmf_schaefer100"
DEFAULT_REFERENCE = ROOT / "data" / "iid_fig6_data" / "input_data" / "DTI_fiber_consensus_HCP.csv"


SCHAEFER100_CENTROIDS = """
7Networks_LH_Vis_1,-26,-34,-17
7Networks_LH_Vis_2,-26,-77,-14
7Networks_LH_Vis_3,-17,-60,-7
7Networks_LH_Vis_4,-27,-95,-4
7Networks_LH_Vis_5,-5,-92,-2
7Networks_LH_Vis_6,-12,-67,7
7Networks_LH_Vis_7,-47,-71,11
7Networks_LH_Vis_8,-25,-88,20
7Networks_LH_Vis_9,-6,-82,26
7Networks_LH_SomMot_1,-53,-23,8
7Networks_LH_SomMot_2,-37,-21,16
7Networks_LH_SomMot_3,-54,-12,13
7Networks_LH_SomMot_4,-55,-8,34
7Networks_LH_SomMot_5,-39,-23,59
7Networks_LH_SomMot_6,-6,-29,70
7Networks_LH_DorsAttn_Post_1,-47,-58,-13
7Networks_LH_DorsAttn_Post_2,-57,-25,39
7Networks_LH_DorsAttn_Post_3,-24,-68,49
7Networks_LH_DorsAttn_Post_4,-42,-34,48
7Networks_LH_DorsAttn_Post_5,-6,-60,56
7Networks_LH_DorsAttn_Post_6,-22,-51,66
7Networks_LH_DorsAttn_PrCv_1,-48,6,28
7Networks_LH_DorsAttn_FEF_1,-26,-3,59
7Networks_LH_SalVentAttn_ParOper_1,-59,-38,29
7Networks_LH_SalVentAttn_FrOperIns_1,-41,-1,-7
7Networks_LH_SalVentAttn_FrOperIns_2,-38,12,6
7Networks_LH_SalVentAttn_PFCl_1,-30,44,30
7Networks_LH_SalVentAttn_Med_1,-5,20,34
7Networks_LH_SalVentAttn_Med_2,-11,-34,45
7Networks_LH_SalVentAttn_Med_3,-6,4,62
7Networks_LH_Limbic_OFC_1,-14,32,-20
7Networks_LH_Limbic_TempPole_1,-32,2,-37
7Networks_LH_Limbic_TempPole_2,-57,-33,-21
7Networks_LH_Cont_Par_1,-37,-53,46
7Networks_LH_Cont_PFCl_1,-43,33,21
7Networks_LH_Cont_pCun_1,-9,-73,38
7Networks_LH_Cont_Cing_1,-4,-26,33
7Networks_LH_Default_Temp_1,-55,-4,-20
7Networks_LH_Default_Temp_2,-58,-32,-1
7Networks_LH_Default_Par_1,-57,-50,12
7Networks_LH_Default_Par_2,-48,-63,35
7Networks_LH_Default_PFC_1,-35,21,-11
7Networks_LH_Default_PFC_2,-47,33,-3
7Networks_LH_Default_PFC_3,-6,47,0
7Networks_LH_Default_PFC_4,-24,61,-1
7Networks_LH_Default_PFC_5,-9,48,41
7Networks_LH_Default_PFC_6,-41,14,48
7Networks_LH_Default_PFC_7,-25,20,51
7Networks_LH_Default_pCunPCC_1,-11,-56,13
7Networks_LH_Default_pCunPCC_2,-6,-53,33
7Networks_RH_Vis_1,32,-31,-22
7Networks_RH_Vis_2,27,-66,-12
7Networks_RH_Vis_3,49,-60,-11
7Networks_RH_Vis_4,22,-96,-5
7Networks_RH_Vis_5,8,-76,5
7Networks_RH_Vis_6,17,-57,5
7Networks_RH_Vis_7,36,-82,16
7Networks_RH_Vis_8,13,-86,29
7Networks_RH_SomMot_1,53,-16,7
7Networks_RH_SomMot_2,40,-15,15
7Networks_RH_SomMot_3,57,-4,11
7Networks_RH_SomMot_4,58,-5,31
7Networks_RH_SomMot_5,47,-11,48
7Networks_RH_SomMot_6,41,-22,60
7Networks_RH_SomMot_7,30,-37,64
7Networks_RH_SomMot_8,6,-26,70
7Networks_RH_DorsAttn_Post_1,50,-62,16
7Networks_RH_DorsAttn_Post_2,50,-24,42
7Networks_RH_DorsAttn_Post_3,38,-45,49
7Networks_RH_DorsAttn_Post_4,27,-67,51
7Networks_RH_DorsAttn_Post_5,14,-52,66
7Networks_RH_DorsAttn_PrCv_1,49,10,27
7Networks_RH_DorsAttn_FEF_1,28,-3,59
7Networks_RH_SalVentAttn_TempOccPar_1,58,-42,13
7Networks_RH_SalVentAttn_TempOccPar_2,61,-26,27
7Networks_RH_SalVentAttn_FrOperIns_1,40,8,1
7Networks_RH_SalVentAttn_Med_1,11,-31,45
7Networks_RH_SalVentAttn_Med_2,7,6,52
7Networks_RH_Limbic_OFC_1,12,35,-20
7Networks_RH_Limbic_TempPole_1,38,1,-38
7Networks_RH_Cont_Par_1,57,-39,44
7Networks_RH_Cont_Par_2,45,-63,46
7Networks_RH_Cont_PFCl_1,30,58,-3
7Networks_RH_Cont_PFCl_2,45,39,15
7Networks_RH_Cont_PFCl_3,32,46,29
7Networks_RH_Cont_PFCl_4,43,16,45
7Networks_RH_Cont_Cing_1,5,-27,33
7Networks_RH_Cont_PFCmp_1,6,28,30
7Networks_RH_Cont_pCun_1,9,-66,43
7Networks_RH_Default_Par_1,55,-51,31
7Networks_RH_Default_Temp_1,62,-23,-19
7Networks_RH_Default_Temp_2,51,7,-18
7Networks_RH_Default_Temp_3,57,-26,-2
7Networks_RH_Default_PFCv_1,35,26,-15
7Networks_RH_Default_PFCv_2,51,28,0
7Networks_RH_Default_PFCdPFCm_1,7,48,1
7Networks_RH_Default_PFCdPFCm_2,11,50,39
7Networks_RH_Default_PFCdPFCm_3,26,24,50
7Networks_RH_Default_pCunPCC_1,12,-54,14
7Networks_RH_Default_pCunPCC_2,7,-52,31
""".strip()


def schaefer_metadata() -> tuple[list[str], np.ndarray]:
    rows = [line.split(",") for line in SCHAEFER100_CENTROIDS.splitlines()]
    labels = [row[0] for row in rows]
    coordinates = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=float)
    if len(labels) != 100 or coordinates.shape != (100, 3) or len(set(labels)) != 100:
        raise RuntimeError("Embedded Schaefer100 metadata is malformed.")
    return labels, coordinates


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spectral_radius(matrix: np.ndarray) -> float:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    return float(np.max(np.abs(np.linalg.eigvalsh(symmetric))))


def edge_vector(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=float)[np.triu_indices(matrix.shape[0], 1)]


def load_subject_connectomes(input_dir: Path) -> tuple[list[Path], np.ndarray]:
    files = sorted((input_dir / "CON_SC_1mio").glob("sub-*.csv"))
    if len(files) != 93:
        raise ValueError(f"Expected 93 subject connectomes, found {len(files)}.")
    matrices = np.stack([np.loadtxt(path, delimiter=",") for path in files])
    if matrices.shape != (93, 100, 100):
        raise ValueError(f"Expected shape (93, 100, 100), got {matrices.shape}.")
    if not np.isfinite(matrices).all() or np.any(matrices < 0.0):
        raise ValueError("Subject connectomes must be finite and non-negative.")
    if not np.allclose(matrices, matrices.transpose(0, 2, 1), atol=1.0e-12):
        raise ValueError("Subject connectomes must be symmetric.")
    if not np.allclose(np.diagonal(matrices, axis1=1, axis2=2), 0.0, atol=1.0e-12):
        raise ValueError("Subject connectome diagonals must be zero.")
    return files, matrices


def ordering_audit(group_mean: np.ndarray, reference_path: Path, *, permutations: int, seed: int) -> dict[str, float]:
    reference = np.loadtxt(reference_path, delimiter=",")
    if reference.shape != group_mean.shape:
        raise ValueError(f"Reference matrix has incompatible shape {reference.shape}.")
    observed = float(np.corrcoef(edge_vector(group_mean), edge_vector(reference))[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        order = rng.permutation(group_mean.shape[0])
        permuted = reference[np.ix_(order, order)]
        null[index] = float(np.corrcoef(edge_vector(group_mean), edge_vector(permuted))[0, 1])
    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
    return {
        "same_order_edge_correlation": observed,
        "permutation_count": int(permutations),
        "permutation_null_mean": float(null.mean()),
        "permutation_null_sd": float(null.std(ddof=1)),
        "permutation_p_one_sided": p_value,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-connectivity", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--scale-mode", choices=("native", "spectral-match", "strength-match"), default="native")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--audit-seed", type=int, default=20260722)
    parser.add_argument("--dmf-seed", type=int, default=0)
    parser.add_argument("--fic-max-iterations", type=int, default=30)
    parser.add_argument("--g-start", type=float, default=1.0)
    parser.add_argument("--g-stop", type=float, default=3.0)
    parser.add_argument("--g-step", type=float, default=0.1)
    parser.add_argument("--skip-dmf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels, coordinates = schaefer_metadata()
    files, matrices = load_subject_connectomes(input_dir)
    native_mean = matrices.mean(axis=0)
    old_source = ROOT / "exp" / "brain" / "result_lausanne_fig6" / "count_00_fig6b_mean_rate.npz"
    with np.load(old_source) as archive:
        old_connectivity = np.asarray(archive["connectivity"], dtype=float)
    native_radius = spectral_radius(native_mean)
    old_radius = spectral_radius(old_connectivity)
    native_strength = float(native_mean.sum(axis=1).mean())
    old_strength = float(old_connectivity.sum(axis=1).mean())
    if args.scale_mode == "native":
        scale = 1.0
    elif args.scale_mode == "spectral-match":
        scale = old_radius / native_radius
    else:
        scale = old_strength / native_strength
    connectivity = native_mean * scale

    group_edges = edge_vector(native_mean)
    subject_edge_correlation = np.asarray(
        [np.corrcoef(edge_vector(matrix), group_edges)[0, 1] for matrix in matrices], dtype=float,
    )
    subject_spectral_radius = np.asarray([spectral_radius(matrix) for matrix in matrices], dtype=float)
    subject_mean_strength = matrices.sum(axis=2).mean(axis=1)
    subject_zero_fraction = np.mean(matrices == 0.0, axis=(1, 2))
    ordering = ordering_audit(
        native_mean,
        resolve_path(args.reference_connectivity),
        permutations=int(args.permutations),
        seed=int(args.audit_seed),
    )

    labels_path = output_dir / "schaefer100_labels.txt"
    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    prep_cache = output_dir / f"group_mean_{args.scale_mode.replace('-', '_')}.npz"
    np.savez_compressed(
        prep_cache,
        connectivity=connectivity,
        native_group_mean=native_mean,
        scale_factor=np.asarray(scale),
        labels=np.asarray(labels),
        centroid_ras=coordinates,
        subject_ids=np.asarray([path.stem for path in files]),
        subject_edge_correlation=subject_edge_correlation,
        subject_spectral_radius=subject_spectral_radius,
        subject_mean_strength=subject_mean_strength,
        subject_zero_fraction=subject_zero_fraction,
    )
    summary = {
        "input": {
            "subject_count": len(files),
            "shape": [100, 100],
            "subject_sha256": {path.name: sha256(path) for path in files},
            "receptor_table_sha256": sha256(input_dir / "average.csv"),
        },
        "aggregation": {
            "method": "elementwise arithmetic mean across 93 subject matrices",
            "scale_mode": str(args.scale_mode),
            "scale_factor": float(scale),
            "native_spectral_radius": native_radius,
            "old83_spectral_radius": old_radius,
            "native_mean_strength": native_strength,
            "old83_mean_strength": old_strength,
            "density": float(np.mean(edge_vector(native_mean) > 0.0)),
            "maximum": float(native_mean.max()),
        },
        "subject_robustness": {
            "edge_correlation_to_group_mean": {
                "min": float(subject_edge_correlation.min()),
                "median": float(np.median(subject_edge_correlation)),
                "max": float(subject_edge_correlation.max()),
            },
            "spectral_radius": {
                "min": float(subject_spectral_radius.min()),
                "median": float(np.median(subject_spectral_radius)),
                "max": float(subject_spectral_radius.max()),
            },
            "zero_fraction": {
                "min": float(subject_zero_fraction.min()),
                "median": float(np.median(subject_zero_fraction)),
                "max": float(subject_zero_fraction.max()),
            },
        },
        "roi_order_audit": {
            **ordering,
            "label_source": "Schaefer2018 100Parcels 7Networks official ordering and RAS centroids",
            "status": "inferred; upstream archive did not include explicit ROI labels",
        },
        "outputs": {
            "prep_cache": str(prep_cache.relative_to(ROOT)),
            "labels": str(labels_path.relative_to(ROOT)),
        },
    }

    if not args.skip_dmf:
        dmf = load_dmf_module()
        count = int(round((float(args.g_stop) - float(args.g_start)) / float(args.g_step))) + 1
        g_values = float(args.g_start) + float(args.g_step) * np.arange(count)
        source_results = output_dir / "source" / f"group_mean_{args.scale_mode.replace('-', '_')}_mean_rate.npz"
        result = dmf.reproduce_fig6b_mean_rate_transition(
            connectivity=connectivity,
            g_values=g_values,
            j_fic_reference_g=1.0,
            seed=int(args.dmf_seed),
            continuation=True,
            compute_phi=False,
            fic_parameters=dmf.FICParameters(max_iterations=int(args.fic_max_iterations)),
            expected_regions=100,
            max_regions=100,
            results_path=source_results,
        )
        summary["dmf_source"] = {
            "path": str(source_results.relative_to(ROOT)),
            "G": np.asarray(result["G"], dtype=float).tolist(),
            "critical_G_by_rate_derivative": float(np.asarray(result["critical_G"]).item()),
            "j_fic_converged": bool(np.all(result["j_fic_calibration_converged"])),
            "j_fic_max_abs_error_hz": float(np.nanmax(result["j_fic_calibration_max_abs_error_hz"])),
            "stabilization_detected_fraction": float(np.mean(result["stabilization_detected"])),
        }
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
