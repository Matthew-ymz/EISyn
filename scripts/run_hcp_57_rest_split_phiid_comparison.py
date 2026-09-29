#!/usr/bin/env python3
"""Compare FC, PEID Syn, and Gaussian-MMI PhiID atoms in REST split halves."""

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

from scripts.phiid_gaussian_mmi import phiid_pair_features
from scripts.run_hcp_57_rest_split_fingerprint import (
    BOOTSTRAPS,
    CACHE as REST_CACHE,
    SEED,
    atomic_json,
    atomic_npz,
    match,
)


OUTPUT = ROOT / "results/hcp_57_rest_split_fingerprint"
PHIID_CACHE = OUTPUT / "phiid_features.npz"
SUMMARY = OUTPUT / "phiid_comparison_summary.json"
FIGURE = OUTPUT / "fc_syn_phiid_comparison.png"
ESTIMATOR = "Gaussian MMI, no bias correction, z-score within half"
METHODS = ("fc_signed21", "peid_syn21", "phiid_red21", "phiid_syn21")
LABELS = {
    "fc_signed21": "FC 21",
    "peid_syn21": "PEID Syn 21",
    "phiid_red21": "PhiID Red 21",
    "phiid_syn21": "PhiID Syn 21",
}
COLORS = {
    "fc_signed21": "#55738f",
    "peid_syn21": "#c56944",
    "phiid_red21": "#6a9b70",
    "phiid_syn21": "#814e82",
}


def load_base(limit: int | None) -> dict[str, np.ndarray]:
    with np.load(REST_CACHE, allow_pickle=False) as archive:
        required = ("subjects", "pc1_fit", "fc_signed21", "fc_abs21", "syn21", "fit_length")
        if any(key not in archive for key in required):
            raise ValueError("REST feature cache lacks network PC1 fitting windows; rerun with --recompute")
        arrays = {key: archive[key] for key in required}
    if limit is not None:
        arrays = {key: (value[:, :limit] if key not in ("subjects", "fit_length")
                        else value[:limit] if key == "subjects" else value)
                  for key, value in arrays.items()}
    n = len(arrays["subjects"])
    if ((limit is None and n != 57) or n < 2
            or arrays["pc1_fit"].shape != (2, n, 450, 7)
            or int(arrays["fit_length"]) != 450):
        raise ValueError("Expected two independent 450-frame, seven-network windows")
    for key in ("fc_signed21", "fc_abs21", "syn21"):
        if arrays[key].shape != (2, n, 21) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid baseline feature {key}")
    return arrays


def compute_phiid(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    subjects = base["subjects"].astype(str)
    red = np.empty((2, len(subjects), 21), dtype=np.float64)
    syn = np.empty_like(red)
    for half in range(2):
        for subject_index in range(len(subjects)):
            red[half, subject_index], syn[half, subject_index] = phiid_pair_features(
                base["pc1_fit"][half, subject_index], tau=1
            )
        print(f"PhiID computed: half {half + 1}/2, {len(subjects)} subjects", flush=True)
    if not np.isfinite(red).all() or not np.isfinite(syn).all():
        raise ValueError("PhiID produced a non-finite atom")
    return {
        "subjects": subjects,
        "phiid_red21": red,
        "phiid_syn21": syn,
        "tau": np.asarray(1),
        "fit_length": np.asarray(450),
        "estimator": np.asarray(ESTIMATOR),
    }


def load_or_compute(base: dict[str, np.ndarray], *, recompute: bool,
                    save: bool) -> dict[str, np.ndarray]:
    if save and PHIID_CACHE.exists() and not recompute:
        with np.load(PHIID_CACHE, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        print("Loaded reusable PhiID feature cache", flush=True)
    else:
        arrays = compute_phiid(base)
    n = len(base["subjects"])
    if (arrays["subjects"].astype(str).tolist() != base["subjects"].astype(str).tolist()
            or int(arrays["tau"]) != 1 or int(arrays["fit_length"]) != 450
            or str(arrays["estimator"]) != ESTIMATOR):
        raise ValueError("PhiID cache cohort or estimator settings differ")
    for key in ("phiid_red21", "phiid_syn21"):
        if arrays[key].shape != (2, n, 21) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid PhiID feature matrix {key}")
    if save and (recompute or not PHIID_CACHE.exists()):
        atomic_npz(PHIID_CACHE, arrays)
    return arrays


def bootstrap_differences(scores: dict[str, np.ndarray], reference: str) -> dict[str, dict]:
    n = len(scores[reference])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n, size=(BOOTSTRAPS, n))
    output = {}
    for key in ("peid_syn21", "phiid_red21", "phiid_syn21"):
        delta = 100 * (scores[key] - scores[reference])
        means = delta[draws].mean(axis=1)
        output[key] = {
            "mean_percentage_points": float(delta.mean()),
            "conditional_95pct_interval_percentage_points":
                np.quantile(means, [0.025, 0.975]).astype(float).tolist(),
            "subjects_positive_delta": int(np.sum(delta > 0)),
            "subjects_negative_delta": int(np.sum(delta < 0)),
            "subjects_tied": int(np.sum(delta == 0)),
        }
    return output


def plot(summary: dict) -> None:
    fig, (ax, difference_ax) = plt.subplots(
        1, 2, figsize=(10.6, 4.1), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1.08]},
    )
    for index, key in enumerate(METHODS):
        raw = 100 * summary["raw"][key]["mean_accuracy"]
        centered = 100 * summary["group_centered_sensitivity"][key]["mean_accuracy"]
        ax.scatter(raw, index - 0.12, s=66, color=COLORS[key], zorder=3,
                   label="Raw" if index == 0 else None)
        ax.scatter(centered, index + 0.12, s=66, facecolor="white",
                   edgecolor=COLORS[key], linewidth=1.8, zorder=3,
                   label="Group centered" if index == 0 else None)
    ax.set_yticks(range(len(METHODS)), [LABELS[key] for key in METHODS])
    ax.invert_yaxis()
    ax.axvline(100 / summary["n_subjects"], ls="--", lw=0.9, color="#656565")
    ax.set_xlabel("Top-1 identification (%)")
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    ax.text(0, 1.20, "a  Four matched 21-feature maps", transform=ax.transAxes,
            fontweight="bold", va="bottom")

    other = ("peid_syn21", "phiid_red21", "phiid_syn21")
    for index, key in enumerate(other):
        row = summary["difference_from_signed_fc"][key]
        point = row["mean_percentage_points"]
        low, high = row["conditional_95pct_interval_percentage_points"]
        difference_ax.errorbar(point, index, xerr=[[point - low], [high - point]],
                               fmt="o", color=COLORS[key], capsize=3,
                               markersize=7, lw=1.5)
    difference_ax.set_yticks(range(len(other)),
                             [f"{LABELS[key]} − FC 21" for key in other])
    difference_ax.invert_yaxis()
    difference_ax.axvline(0, lw=0.9, color="#656565")
    difference_ax.set_xlabel("Raw accuracy difference (percentage points)")
    difference_ax.grid(axis="x", alpha=0.18)
    difference_ax.set_axisbelow(True)
    difference_ax.text(0, 1.20, "b  Paired difference (95% interval)",
                       transform=difference_ax.transAxes, fontweight="bold", va="bottom")
    for axis in (ax, difference_ax):
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, metavar="N", help="First N subjects only; no files saved")
    parser.add_argument("--recompute", action="store_true", help="Recalculate PhiID atoms")
    args = parser.parse_args()
    if args.smoke is not None and not 2 <= args.smoke <= 57:
        parser.error("--smoke must be between 2 and 57")
    base = load_base(args.smoke)
    phiid = load_or_compute(base, recompute=args.recompute, save=args.smoke is None)
    features = {
        "fc_signed21": base["fc_signed21"],
        "fc_abs21": base["fc_abs21"],
        "peid_syn21": base["syn21"],
        "phiid_red21": phiid["phiid_red21"],
        "phiid_syn21": phiid["phiid_syn21"],
    }
    raw = {}
    centered = {}
    scores = {}
    for key, array in features.items():
        raw[key], scores[key] = match(array, group_centered=False)
        centered[key], _ = match(array, group_centered=True)
    phid_red = phiid["phiid_red21"]
    phid_syn = phiid["phiid_syn21"]
    summary = {
        "n_subjects": len(base["subjects"]),
        "gallery_size": len(base["subjects"]),
        "chance_accuracy": 1 / len(base["subjects"]),
        "feature_count_each": 21,
        "data": "Same 57 REST1_LR subjects, two independently fitted 450-frame Yeo7-PC1 windows",
        "phiid_estimator": "Gaussian MMI without bias correction, z-score per network and half, tau=1",
        "phiid_atom_source": "brainets/hoi AtomsPhiID, commit 4db2fbd701d40d33fe2375f3df5a4a4dc94c6b83",
        "raw": raw,
        "group_centered_sensitivity": centered,
        "difference_from_signed_fc": bootstrap_differences(scores, "fc_signed21"),
        "difference_from_absolute_fc": bootstrap_differences(scores, "fc_abs21"),
        "uncertainty": {"bootstrap_replicates": BOOTSTRAPS, "seed": SEED,
                        "interpretation": "paired probe-subject bootstrap with fixed gallery"},
        "phiid_atom_quality": {
            "red_min_bits": float(phid_red.min()),
            "red_negative_count": int(np.sum(phid_red < 0)),
            "syn_min_bits": float(phid_syn.min()),
            "syn_negative_count": int(np.sum(phid_syn < 0)),
            "note": "PhiID MMI atoms are signed and were not clipped",
        },
    }
    print(json.dumps({"raw": raw, "difference_from_signed_fc":
                      summary["difference_from_signed_fc"],
                      "phiid_atom_quality": summary["phiid_atom_quality"]}, indent=2))
    if args.smoke is None:
        atomic_json(SUMMARY, summary)
        plot(summary)


if __name__ == "__main__":
    main()
