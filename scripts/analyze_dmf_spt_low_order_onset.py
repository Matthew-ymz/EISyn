#!/usr/bin/env python3
"""Cache-only low-order dominance and frozen-pair localization near G=0.5.

No new simulations, partition searches, or nonlinear estimator fits. Direct
Gaussian ROI-pair Syn uses the SAME cached conditional covariance as the SPT.
Discovery: freeze the top three ROI pairs in seed 4 at G=0.6; inspect the seven
other simulation seeds descriptively. This is a post-hoc, exploratory analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_dmf_spt_order_distribution import INPUT, SEEDS, STYLE, TOLERANCE, load_analysis, selected_nodes
from scripts.plot_dmf_spt_max_syn import BLUE, ORANGE, NATS_PER_BIT, low_order_dominance, max_by_order
from scripts.brain_surface_plot import draw_brain_map_four_views
from scripts.plot_dmf_schaefer100_summary import load_schaefer100_surface_map

FIGURE = ROOT / "fig/dmf_schaefer100/dmf_spt_low_order_onset.png"
SUMMARY = ROOT / "results/dmf_schaefer100/spt_order_distribution/low_order_onset_summary.json"
SURFACE = ROOT / "results/dmf_schaefer100/schaefer100_fsaverage5_surface.npz"
PAIRS = np.array(np.triu_indices(100, 1)).T
DISCOVERY_SEED, DISCOVERY_G, FIXED_PAIR_COUNT = 4, .6, 3
CUTOFFS = (5, 10, 20)
EARLY_MAX_G = 1.0
PAIR_COLORS = ("#71509A", "#138B8B", "#D56A28")


def audit_syn(raw, label):
    """Native bits; numerical-zero treatment is explicit and counted."""
    raw = np.asarray(raw, dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError(f"{label}: nonfinite Syn")
    bad = raw < -TOLERANCE
    if bad.any():
        raise ValueError(f"{label}: Syn violation: minimum={raw.min():.12g}, "
                         f"threshold={-TOLERANCE:.12g}, affected_count={int(bad.sum())}")
    treated = raw.copy()
    treated[treated < 0] = 0.0
    return treated, dict(minimum_raw_syn_bits=float(raw.min()), tolerance_zero_count=int(np.sum(raw < 0)),
                         significant_negative_count=0, evaluated_pair_count=raw.size, syn_tolerance_bits=TOLERANCE)


def gaussian_pair_syn(covariance):
    """Block conditional TC: [log|C_i|+log|C_j|-log|C_ij|]/(2 ln 2)."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (200, 200) or not np.isfinite(covariance).all():
        raise ValueError("Expected a finite 200-dimensional E/I covariance")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
        raise ValueError("Conditional covariance must be symmetric")
    roi = np.column_stack([np.arange(100), np.arange(100) + 100])
    sources = np.column_stack([PAIRS[:, 0], PAIRS[:, 0] + 100, PAIRS[:, 1], PAIRS[:, 1] + 100])
    # Cholesky, rather than determinant sign alone, requires positive definiteness.
    roi_chol = np.linalg.cholesky(covariance[roi[:, :, None], roi[:, None, :]])
    pair_chol = np.linalg.cholesky(covariance[sources[:, :, None], sources[:, None, :]])
    roi_logdet = 2 * np.log(np.diagonal(roi_chol, axis1=-2, axis2=-1)).sum(axis=1)
    pair_logdet = 2 * np.log(np.diagonal(pair_chol, axis1=-2, axis2=-1)).sum(axis=1)
    return (roi_logdet[PAIRS[:, 0]] + roi_logdet[PAIRS[:, 1]] - pair_logdet) / (2 * np.log(2))


def load_pair_analysis(data, input_dir):
    gs = data["G"][data["G"] <= EARLY_MAX_G]
    paths = {}
    for p in (input_dir / "shards").glob("seeds*/covariance/seed*.npz"):
        # Use only shard caches, never overlapping root-directory copies.
        if p.name in paths:
            raise ValueError(f"Duplicate covariance condition: {p.name}")
        paths[p.name] = p
    values = np.empty((len(SEEDS), len(gs), len(PAIRS)))
    fingerprints = hashlib.sha256()
    source_hashes = {}
    tree_pair_error, checked_tree_pairs = 0., 0
    winner_members = []
    for si, seed in enumerate(SEEDS):
        for gi, g in enumerate(gs):
            name = f"seed{seed:02d}_G{g:.2f}.npz"
            if name not in paths:
                raise ValueError(f"Missing paired conditional covariance: {name}")
            path = paths[name]
            fingerprints.update(str(path.relative_to(input_dir)).encode() + b"\0" + path.read_bytes())
            with np.load(path) as cache:
                config = json.loads(str(cache["config_json"].item()))
                if config != data["simulation"] or int(cache["seed"]) != seed or not np.isclose(float(cache["G"]), g):
                    raise ValueError("Covariance/tree simulation contract mismatch")
                digest = str(cache["input_sha256"].item())
                if source_hashes.setdefault(int(seed), digest) != digest:
                    raise ValueError("Intervention samples differ across G within a seed")
                values[si, gi] = gaussian_pair_syn(cache["conditional_covariance"])
            tree_path = path.parent.parent / "trees" / name.replace(".npz", ".json")
            nodes = selected_nodes(json.loads(tree_path.read_text())["tree"])
            for node in nodes:
                if len(node["indices"]) == 2:
                    i, j = sorted(node["indices"])
                    index = int(np.flatnonzero(np.all(PAIRS == [i, j], axis=1))[0])
                    tree_pair_error = max(tree_pair_error, abs(values[si, gi, index] - node["syn_bits_raw"]))
                    checked_tree_pairs += 1
            if np.isclose(g, DISCOVERY_G):
                winner = max(nodes, key=lambda n:n["syn_bits_raw"])
                winner_members.append(dict(seed=int(seed), members=winner["indices"], order=len(winner["indices"])))
    treated, audit = audit_syn(values, "Direct Gaussian pairs")
    if tree_pair_error > TOLERANCE:
        raise ValueError(f"Direct Syn disagrees with cached tree pairs: {tree_pair_error}")
    audit.update(maximum_direct_tree_pair_error_bits=tree_pair_error, checked_tree_pair_count=checked_tree_pairs)
    si, gi = int(np.flatnonzero(SEEDS == DISCOVERY_SEED)[0]), int(np.flatnonzero(np.isclose(gs, DISCOVERY_G))[0])
    frozen_indices = np.argsort(-treated[si, gi], kind="stable")[:FIXED_PAIR_COUNT]
    frozen_values = treated[:, :, frozen_indices]
    # Competition ranking, exact ties share a rank. No sorted-column tie artifacts.
    ranks = 1 + np.sum(treated[:, :, None, :] > frozen_values[:, :, :, None], axis=-1)
    return dict(G=gs, frozen_indices=frozen_indices, pairs=PAIRS[frozen_indices], values=frozen_values,
                ranks=ranks, maximum=treated.max(axis=2), p95=np.quantile(treated, .95, axis=2),
                audit=audit, covariance_fingerprint=fingerprints.hexdigest(), source_hashes=source_hashes,
                tree_winner_members=winner_members)


def decorate(ax, title, *, title_pad=28, ylabel=None):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10, pad=title_pad)
    ax.set(xlim=(-.02, 1.02), xlabel=r"Global coupling, $G$", xticks=np.arange(0, 1.01, .2))
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E8EAED", lw=.5)
    ax.axvspan(.5, .6, color="#D5D9DE", alpha=.25, linewidth=0)


def outside_key(ax, **kwargs):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False,
              borderaxespad=0, fontsize=8, **kwargs)


def plot(data, dominance, pairs, labels, output):
    early = data["G"] <= EARLY_MAX_G
    gs = data["G"][early]
    core_rois = set(pairs["pairs"].ravel().tolist())
    roi_values = np.full(100, np.nan)
    for i in core_rois:
        roi_values[i] = 1 if "_Vis_" in labels[i] else 2
    left, right, lv, rv = load_schaefer100_surface_map(SURFACE, labels, roi_values)
    with mpl.rc_context({**STYLE, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(12, 8.5), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.1), width_ratios=(1, 1))
        ax = fig.add_subplot(grid[0, 0])
        for si, values in enumerate(dominance["dominance"][:, early]):
            ax.plot(gs, values, color="#A1A7AD", lw=.9, alpha=.7,
                    label="Individual seeds" if si == 0 else None)
        mean, sd = dominance["mean"][early], dominance["sd"][early]
        ax.fill_between(gs, mean - sd, mean + sd, color=BLUE, alpha=.15, linewidth=0)
        ax.plot(gs, mean, color=BLUE, lw=1.8, marker="o", ms=3.5, label="8-seed mean ± SD")
        ax.axhline(0, color="#666666", lw=.8, ls="--")
        ax.set(ylim=(-.85, .38))
        decorate(ax, r"a  Low-order dominance, $D_{10}=(L-H)/(L+H)$", ylabel=r"Low-order dominance, $D_{10}$")
        outside_key(ax, ncol=2)

        ax = fig.add_subplot(grid[0, 1])
        for key, color, label in (("low", BLUE, "Low-order maximum (2–10 ROI)"),
                                  ("high", ORANGE, "High-order maximum (11–100 ROI)")):
            values = dominance[key][:, early] * NATS_PER_BIT
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.fill_between(gs, mean - sd, mean + sd, color=color, alpha=.13, linewidth=0)
            ax.plot(gs, mean, color=color, lw=1.8, marker="o", ms=3.5, label=label)
        decorate(ax, "b  Growing low-order strength crosses the background", title_pad=37,
                 ylabel="Maximum selected node Syn (nats)")
        ax.set_ylim(0, .315)
        outside_key(ax, ncol=1, labelspacing=.25)

        ax = fig.add_subplot(grid[1, 0])
        names = [str(labels[i]).removeprefix("7Networks_") + "–" + str(labels[j]).split("_", 2)[-1]
                 for i, j in pairs["pairs"]]
        for pi, (name, color) in enumerate(zip(names, PAIR_COLORS, strict=True)):
            values = pairs["values"][:, :, pi] * NATS_PER_BIT
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.fill_between(gs, mean - sd, mean + sd, color=color, alpha=.10, linewidth=0)
            ax.plot(gs, mean, color=color, lw=1.8, label=name)
        ax.plot(gs, pairs["p95"].mean(0) * NATS_PER_BIT, color="#777777", ls="--", lw=1,
                label="95th percentile of all 4,950 pairs")
        decorate(ax, "c  Fixed pairs: direct Syn, independent of the tree", title_pad=61,
                 ylabel="Fixed-pair Syn (nats)")
        ax.set_ylim(0, .315)
        outside_key(ax, ncol=1, labelspacing=.2)

        # Exact atlas name matching; colors mark the five ROIs in the frozen pairs.
        sub = grid[1, 1].subgridspec(4, 2, height_ratios=(.3, 1, 1, .09), hspace=0, wspace=0)
        host = fig.add_subplot(sub[0, :])
        host.set_axis_off()
        host.set_title("d  Localized posterior visual–Default combinations", loc="left", fontweight="bold", fontsize=10, pad=10)
        host.text(0, .7, "3 fixed pairs; rank ≤3 in all 7 other seeds at G=0.6\nAtlas row order remains inferred",
                  transform=host.transAxes, fontsize=8, va="top", color="#444444")
        axes = [fig.add_subplot(sub[1, 0], projection="3d"), fig.add_subplot(sub[1, 1], projection="3d"),
                fig.add_subplot(sub[2, 0], projection="3d"), fig.add_subplot(sub[2, 1], projection="3d")]
        cb_ax = fig.add_subplot(sub[3, :])
        draw_brain_map_four_views(axes, cb_ax, left, right, lv, rv,
                                 cmap=ListedColormap(["#71509A", "#539327"]), vmin=.5, vmax=2.5,
                                 background_color="#DEDEDE", view_labels=True, zoom=1.7)
        cb_ax.set_xticks([1, 2], labels=["Visual: Vis_3 / Vis_6", "Default: pCunPCC_1"])
        cb_ax.tick_params(length=0, labelsize=8)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    data = load_analysis(args.input_dir)
    statistics = max_by_order(data)
    dominance = low_order_dominance(statistics["maxima"])
    sensitivities = {str(k):low_order_dominance(statistics["maxima"], k) for k in CUTOFFS}
    pairs = load_pair_analysis(data, args.input_dir)
    with np.load(ROOT / "results/dmf_schaefer100/full/critical_yeo7.npz", allow_pickle=True) as cache:
        labels = cache["region_labels"].astype(str)
    with np.load(ROOT / "results/dmf_schaefer100/source/group_mean_native_mean_rate.npz") as cache:
        sc, rate_g, rates = cache["connectivity"], cache["G"], cache["mean_rate_hz"]
    if not np.allclose(sc, sc.T, atol=1e-12, rtol=0):
        raise ValueError("Structural ranking assumes the existing symmetric SC")
    plot(data, dominance, pairs, labels, args.figure)
    jj = [int(np.flatnonzero(np.isclose(data["G"], g))[0]) for g in (.5, .6)]
    pj = int(np.flatnonzero(np.isclose(pairs["G"], DISCOVERY_G))[0])
    validation = SEEDS != DISCOVERY_SEED
    all_sc = sc[PAIRS[:, 0], PAIRS[:, 1]]
    pair_records = []
    for pi, (i, j) in enumerate(pairs["pairs"]):
        pair_records.append(dict(indices=[int(i), int(j)], labels=labels[[i, j]].tolist(),
            structural_weight=float(sc[i, j]), structural_rank=int(1 + np.sum(all_sc > sc[i, j])),
            seed_ranks_at_G06=pairs["ranks"][:, pj, pi].tolist(),
            validation_top3_count=int(np.sum(pairs["ranks"][validation, pj, pi] <= 3)),
            per_seed_syn_nats=(pairs["values"][:, :, pi] * NATS_PER_BIT).tolist(),
            mean_syn_nats=(pairs["values"][:, :, pi].mean(0) * NATS_PER_BIT).tolist(),
            sd_syn_nats=(pairs["values"][:, :, pi].std(0, ddof=1) * NATS_PER_BIT).tolist()))
    brackets = {}
    for k, entry in sensitivities.items():
        rows = []
        for si, values in enumerate(entry["dominance"]):
            gi = int(np.flatnonzero(values > 0)[0])
            rows.append(dict(seed=int(SEEDS[si]), last_nonpositive_sample_G=float(data["G"][gi-1]),
                             first_positive_sample_G=float(data["G"][gi])))
        brackets[k] = rows
    payload = dict(
        status="Exploratory cache-only analysis; no new dynamics or searches",
        metric_definition="D10=(L-H)/(L+H), L=max selected Syn at orders 2..10, H=max at orders 11..100; signed dimensionless strength comparison, not Syn or budget share",
        low_order_cutoff_reason="Reuse existing C10 scale, not optimize a cutoff against the observed crossover",
        G=data["G"].tolist(), seeds=SEEDS.tolist(), per_seed_dominance=dominance["dominance"].tolist(),
        mean_dominance=dominance["mean"].tolist(), sd_dominance=dominance["sd"].tolist(),
        positive_seed_count=dominance["positive_seed_count"].tolist(),
        mean_low_max_syn_nats=(dominance["low"].mean(0) * NATS_PER_BIT).tolist(),
        mean_high_max_syn_nats=(dominance["high"].mean(0) * NATS_PER_BIT).tolist(),
        low_order_budget_share_mean=data["share"][:, :, :9].sum(axis=2).mean(axis=0).tolist(),
        low_order_budget_share_sd=data["share"][:, :, :9].sum(axis=2).std(axis=0, ddof=1).tolist(),
        first_positive_sample_brackets=brackets,
        G05_to_G06=dict(per_seed_dominance_change=(dominance["dominance"][:, jj[1]] - dominance["dominance"][:, jj[0]]).tolist(),
            positive_change_count=int(np.sum(dominance["dominance"][:, jj[1]] > dominance["dominance"][:, jj[0]]))),
        frozen_pair_selection=dict(seed=DISCOVERY_SEED, G=DISCOVERY_G, rule="Top 3 of all 4,950 direct Gaussian ROI pairs",
            validation_seeds=SEEDS[validation].tolist(), evidence_boundary="Retrospective descriptive seed check; onset and metric chosen after inspecting the scan"),
        early_G=pairs["G"].tolist(), frozen_pairs=pair_records,
        pair_p95_mean_nats=(pairs["p95"].mean(0) * NATS_PER_BIT).tolist(),
        tree_winner_members_at_G06=pairs["tree_winner_members"],
        mean_rate_G=rate_g.tolist(), mean_rate_hz=rates.tolist(),
        tree_audit=data["audit"], direct_pair_audit=pairs["audit"],
        simulation=data["simulation"], search=data["search"],
        input_tree_fingerprint_sha256=data["fingerprint"],
        input_covariance_fingerprint_sha256=pairs["covariance_fingerprint"],
        paired_source_hashes=pairs["source_hashes"],
        figure=str(args.figure),
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(G05_to_G06=payload["G05_to_G06"],
        frozen_pairs=[{k:v for k,v in r.items() if k not in ("per_seed_syn_nats", "mean_syn_nats", "sd_syn_nats")} for r in pair_records],
        direct_pair_audit=payload["direct_pair_audit"]), indent=2))
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
