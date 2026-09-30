#!/usr/bin/env python3
"""Compare seven REST split-half fingerprints after half-specific group centering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_ped_oinfo_57 import (
    PED_TOLERANCE_BITS,
    compute_window,
    triplets,
)
from scripts.run_hcp_57_rest_split_fingerprint import (
    BOOTSTRAPS,
    SEED,
    atomic_json,
    atomic_npz,
    match,
)
from scripts.run_hcp_57_rest_split_phiid_comparison import load_base, load_or_compute


OUTPUT = ROOT / "results/hcp_57_rest_split_fingerprint"
CACHE = OUTPUT / "oinfo_ped_features.npz"
SUMMARY = OUTPUT / "group_centered_comparison_summary.json"
FIGURE = OUTPUT / "rest_split_group_centered_comparison.png"
METHODS = (
    "fc_signed21", "peid_syn21", "phiid_red21", "phiid_syn21",
    "oinfo35", "ped_red35", "ped_syn35",
)
LABELS = (
    "FC 21", "PEID Syn 21", "PhiID Red 21", "PhiID Syn 21",
    "O-information 35", "PED Red 35", "PED Syn 35",
)
COLORS = (
    "#55738f", "#c56944", "#6a9b70", "#814e82",
    "#9a7444", "#4d8b92", "#ba687f",
)


def compute_triplets(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    subjects = base["subjects"].astype(str)
    n = len(subjects)
    result = {key: np.empty((2, n, 35), dtype=np.float64)
              for key in ("oinfo35", "ped_red35", "ped_syn35")}
    minimum = np.inf
    tolerance_negative_count = 0
    for half in range(2):
        for index, series in enumerate(base["pc1_fit"][half]):
            red, syn, oinfo, atom_min, negative_count = compute_window(series)
            result["ped_red35"][half, index] = red
            result["ped_syn35"][half, index] = syn
            result["oinfo35"][half, index] = oinfo
            minimum = min(minimum, atom_min)
            tolerance_negative_count += negative_count
        print(f"O-information/PED computed: half {half + 1}/2, {n} subjects", flush=True)
    result.update({
        "subjects": subjects,
        "triplets": np.asarray(["+".join(names) for names in triplets()]),
        "fit_length": np.asarray(450),
        "ped_atom_minimum_bits": np.asarray(minimum),
        "ped_tolerance_negative_atom_count": np.asarray(tolerance_negative_count),
        "ped_tolerance_bits": np.asarray(PED_TOLERANCE_BITS),
    })
    return result


def load_triplets(base: dict[str, np.ndarray], *, recompute: bool, save: bool) -> dict[str, np.ndarray]:
    if save and CACHE.exists() and not recompute:
        with np.load(CACHE, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        print("Loaded reusable O-information/PED feature cache", flush=True)
    else:
        arrays = compute_triplets(base)
    expected_triplets = ["+".join(names) for names in triplets()]
    if (arrays["subjects"].astype(str).tolist() != base["subjects"].astype(str).tolist()
            or arrays["triplets"].astype(str).tolist() != expected_triplets
            or int(arrays["fit_length"]) != 450
            or float(arrays["ped_tolerance_bits"]) != PED_TOLERANCE_BITS):
        raise ValueError("Triplet feature cache cohort, order, or settings differ")
    for key in ("oinfo35", "ped_red35", "ped_syn35"):
        value = arrays[key]
        if value.shape != (2, len(base["subjects"]), 35) or not np.isfinite(value).all():
            raise ValueError(f"Invalid triplet feature {key}: {value.shape}")
    if float(arrays["ped_atom_minimum_bits"]) < -PED_TOLERANCE_BITS:
        raise ValueError("PED cache contains a significant negative partial atom")
    if save and (recompute or not CACHE.exists()):
        atomic_npz(CACHE, arrays)
    return arrays


def paired_differences(scores: dict[str, np.ndarray]) -> dict[str, dict]:
    n = len(scores["fc_signed21"])
    draws = np.random.default_rng(SEED).integers(0, n, size=(BOOTSTRAPS, n))
    output = {}
    for key in METHODS[1:]:
        delta = 100 * (scores[key] - scores["fc_signed21"])
        means = delta[draws].mean(axis=1)
        output[key] = {
            "mean_percentage_points": float(delta.mean()),
            "conditional_95pct_interval_percentage_points":
                np.quantile(means, [0.025, 0.975]).astype(float).tolist(),
            "subjects_positive_delta": int((delta > 0).sum()),
            "subjects_negative_delta": int((delta < 0).sum()),
            "subjects_tied": int((delta == 0).sum()),
        }
    return output


def plot(summary: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    values = [100 * summary["results"][key]["mean_accuracy"] for key in METHODS]
    positions = np.arange(len(METHODS))
    for index, (value, color) in enumerate(zip(values, COLORS, strict=True)):
        ax.scatter(value, index, s=76, color=color, zorder=3)
        ax.text(value + 0.65, index, f"{value:.2f}%", va="center", fontsize=9,
                color="#333333")
    ax.axvline(100 / summary["n_subjects"], color="#777777", lw=1,
               ls="--", zorder=1)
    ax.set_yticks(positions, LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) + 5)
    ax.set_xlabel("Top-1 identification (%)")
    ax.grid(axis="x", color="#e5e5e5", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, metavar="N", help="First N subjects; save nothing")
    parser.add_argument("--recompute", action="store_true", help="Recompute O-information/PED")
    args = parser.parse_args()
    if args.smoke is not None and not 2 <= args.smoke <= 57:
        parser.error("--smoke must be between 2 and 57")
    base = load_base(args.smoke)
    phiid = load_or_compute(base, recompute=False, save=args.smoke is None)
    higher = load_triplets(base, recompute=args.recompute, save=args.smoke is None)
    features = {
        "fc_signed21": base["fc_signed21"],
        "peid_syn21": base["syn21"],
        "phiid_red21": phiid["phiid_red21"],
        "phiid_syn21": phiid["phiid_syn21"],
        "oinfo35": higher["oinfo35"],
        "ped_red35": higher["ped_red35"],
        "ped_syn35": higher["ped_syn35"],
    }
    results = {}
    scores = {}
    for key in METHODS:
        results[key], scores[key] = match(features[key], group_centered=True)
    summary = {
        "n_subjects": len(base["subjects"]),
        "gallery_size": len(base["subjects"]),
        "chance_accuracy": 1 / len(base["subjects"]),
        "windows": "Two independently fitted 450-frame Yeo7 PC1 windows from one REST1_LR scan",
        "matching": "Half-specific across-subject coordinate centering, then row-wise Pearson and bidirectional Top-1",
        "feature_dimensions": {key: int(features[key].shape[2]) for key in METHODS},
        "triplet_methods": {
            "oinfo35": "Signed Gaussian-copula O-information with finite-sample bias correction",
            "ped_red35": "Binary shared-exclusion PED redundancy, unnormalized",
            "ped_syn35": "Binary shared-exclusion PED synergy, unnormalized",
        },
        "results": results,
        "difference_from_signed_fc": paired_differences(scores),
        "uncertainty": {"bootstrap_replicates": BOOTSTRAPS, "seed": SEED,
                        "interpretation": "paired probe-subject bootstrap with fixed gallery"},
        "ped_atom_quality": {
            "minimum_bits": float(higher["ped_atom_minimum_bits"]),
            "nonnegative_tolerance_bits": PED_TOLERANCE_BITS,
            "tolerance_negative_count": int(higher["ped_tolerance_negative_atom_count"]),
            "significant_violation_count": 0,
        },
    }
    print(json.dumps({"results": results, "ped_atom_quality": summary["ped_atom_quality"]},
                     indent=2))
    if args.smoke is None:
        atomic_json(SUMMARY, summary)
        plot(summary)


if __name__ == "__main__":
    main()
