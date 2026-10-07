#!/usr/bin/env python3
"""Paired input diagnostics, with fixed legacy dynamics; never changes main estimates."""
from __future__ import annotations

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import periodogram
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_hcp_timeseries_abc_979 import (
    AUDIT, LABELS, NETWORKS, LN2, TOL, atomic_json, correlation_columns,
    decomposition, fit_delta_history_phi, load_yeo7_groups, sha,
)

OUT = ROOT / "results/hcp_timeseries_abc_replication_979/rest_diagnostic"
OLD = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_57_brain"


def reduce(raw, groups):
    scores, models, rows = [], [], []
    for name, ids in groups.items():
        a = raw[:, ids]
        model = PCA(n_components=1, svd_solver="full").fit(a[:900])
        s = model.transform(a)[:, 0]
        weights = model.components_[0]
        z = (a - a.mean(0)) / a.std(0)
        pair = np.corrcoef(a, rowvar=False)[np.triu_indices(len(ids), 1)]
        f, power = periodogram(s, fs=1., detrend="constant")
        rows.append(dict(network=name, n_parcels=len(ids),
            pc1_explained=float(model.explained_variance_ratio_[0]),
            pc1_lag1=float(np.corrcoef(s[:-1], s[1:])[0, 1]),
            pc1_power_above_0p1_cycles_per_frame=float(power[f > .1].sum() / power.sum()),
            pc1_effective_parcels=float(1 / np.sum(weights ** 4)),
            pc1_largest_loading_energy=float(np.max(weights ** 2)),
            roi_median_sd=float(np.median(a.std(0))),
            roi_mean_pairwise_correlation=float(pair.mean()),
            roi_median_lag1=float(np.median(np.sum(z[:-1]*z[1:], axis=0) /
                np.sqrt(np.sum(z[:-1]**2, axis=0)*np.sum(z[1:]**2, axis=0)))),
            top_loading_parcel_indices_1based=[int(ids[j] + 1) for j in np.argsort(weights**2)[-5:][::-1]]))
        scores.append(s)
        models.append(model)
    return np.column_stack(scores), models, rows


def evaluate(scores):
    fitted = fit_delta_history_phi(scores, alpha=1., order=3, development_end=900)
    d = decomposition(fitted["transition"], fitted["noise_covariance"])
    root = d["nodes"][0]
    return dict(system_xi_nats=d["system_xi_bits"]*LN2,
        joint_ei_nats=d["joint_ei_bits"]*LN2,
        scalar_ei_sum_nats=sum(d["scalar_ei_bits"])*LN2,
        cross_xi_nats=d["cross_network_xi_bits"]*LN2,
        root7_syn_nats=d["order_mass_bits"][-1]*LN2,
        root_children=root["children"],
        root_isolates_limbic=["Limbic"] in root["children"],
        network_percent=d["network_percent"],
        noise_diagonal=np.diag(fitted["noise_covariance"]).tolist(),
        heldout=fitted["heldout"],
        minimum_candidate_syn_bits=d["numerical_audit"]["all_candidate_syn"]["minimum_bits"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = json.loads(AUDIT.read_text())["old_cohort_subjects"]
    groups = load_yeo7_groups(LABELS, expected_parcels=1000)
    frozen = json.loads((OUT.parent / "rest_input_verification.json").read_text())
    reference = {r["subject"]: r for r in frozen["rows"]}
    rows = []
    for k, subject in enumerate(ids):
        op = OLD / f"sub-{subject}/sub-{subject}_hcp_s1200_rfMRI_REST1_LR_schaefer500-1000_yeo7.mat"
        npth = ROOT / f"data/timeseries/{subject}.mat"
        old = loadmat(op, variable_names=["Schaefer1000"])["Schaefer1000"].astype(float)
        new = loadmat(npth, variable_names=["Schaefer1000_NoGSR"])["Schaefer1000_NoGSR"].astype(float)
        if old.shape != (1200, 1000) or new.shape != old.shape:
            raise ValueError(f"Input shape mismatch: {subject}")
        oscore, omodel, orows = reduce(old, groups)
        nscore, nmodel, nrows = reduce(new, groups)
        # Exchange only spatial loadings. Dynamics, intervention, sample lengths,
        # and native Syn tolerance stay fixed; these are diagnostic representations.
        new_old_basis = np.column_stack([model.transform(new[:, ix])[:, 0]
            for model, ix in zip(omodel, groups.values())])
        old_new_basis = np.column_stack([model.transform(old[:, ix])[:, 0]
            for model, ix in zip(nmodel, groups.values())])
        variants = dict(old_native=evaluate(oscore), new_native=evaluate(nscore),
            new_with_old_loadings=evaluate(new_old_basis), old_with_new_loadings=evaluate(old_new_basis))
        for name, key in (("old_native", "old_rest_xi_bits"), ("new_native", "new_rest_xi_bits")):
            error = abs(variants[name]["system_xi_nats"] / LN2 - reference[subject][key])
            if error > TOL:
                raise ValueError(f"Input/calibration mismatch: {subject}/{name}: {error} bits")
        corr = correlation_columns(old, new)
        pc_corr = np.abs(correlation_columns(oscore, nscore))
        x = old-old.mean(0)
        y = new-new.mean(0)
        diff = y - x * np.sum(x*y, axis=0)/np.sum(x*x, axis=0)
        # Fixed 7-component SVD diagnostic of the affine residual: no tuned filter.
        u, s, _ = np.linalg.svd(diff, full_matrices=False)
        result = dict(subject=subject, old_path=str(op.resolve()), new_path=str(npth),
            old_sha256=sha(op), new_sha256=sha(npth), old=orows, new=nrows,
            roi_correlations_by_network=[float(np.median(corr[ix])) for ix in groups.values()],
            own_pc_score_abs_correlations=pc_corr.tolist(),
            affine_residual_relative_norm=float(np.linalg.norm(diff)/np.linalg.norm(y)),
            affine_residual_top1_variance=float(s[0]**2 / np.sum(s*s)),
            affine_residual_top7_variance=float(np.sum(s[:7]**2) / np.sum(s*s)),
            affine_residual_pc1_abs_corr_old_global=float(abs(np.corrcoef(u[:, 0], x.mean(1))[0, 1])),
            variants=variants)
        # Same-time parcel matching is a spot check, not atlas provenance validation.
        if k in (0, len(ids)//2, len(ids)-1):
            zo=x/x.std(0); zn=y/y.std(0)
            correspondence=np.abs(zo.T@zn/len(zo))
            result["parcel_same_index_best_match_fraction"] = float(np.mean(
                np.argmax(correspondence, axis=1) == np.arange(1000)))
        rows.append(result)
        if (k+1)%10==0 or k==len(ids)-1:
            print(f"Paired REST diagnostics {k+1}/{len(ids)}", flush=True)
    measures = [key for key in rows[0]["old"][0] if key not in (
        "network", "n_parcels", "top_loading_parcel_indices_1based")]
    summary = {side:{key:np.mean([[r[side][i][key] for i in range(7)] for r in rows], axis=0).tolist()
        for key in measures} for side in ("old", "new")}
    variant_summary = {}
    for variant in rows[0]["variants"]:
        v=[r["variants"][variant] for r in rows]
        variant_summary[variant] = {key:float(np.mean([a[key] for a in v])) for key in (
            "system_xi_nats", "joint_ei_nats", "scalar_ei_sum_nats", "cross_xi_nats", "root7_syn_nats")}
        variant_summary[variant]["root_isolates_limbic_count"] = sum(a["root_isolates_limbic"] for a in v)
        variant_summary[variant]["mean_network_percent"] = np.mean([a["network_percent"] for a in v], axis=0).tolist()
        variant_summary[variant]["mean_noise_diagonal"] = np.mean([a["noise_diagonal"] for a in v], axis=0).tolist()
        variant_summary[variant]["root_splits"] = dict(Counter(" | ".join("+".join(c) for c in a["root_children"]) for a in v))
    summary.update(variants=variant_summary,
        mean_own_pc_score_abs_correlations=np.mean([r["own_pc_score_abs_correlations"] for r in rows], axis=0).tolist(),
        mean_roi_correlations_by_network=np.mean([r["roi_correlations_by_network"] for r in rows], axis=0).tolist(),
        new_limbic_pc1_lag1_below_0p2_count=sum(r["new"][4]["pc1_lag1"] < .2 for r in rows),
        new_limbic_largest_loading_above_0p5_count=sum(r["new"][4]["pc1_largest_loading_energy"] > .5 for r in rows),
        mean_affine_residual_top1_variance=float(np.mean([r["affine_residual_top1_variance"] for r in rows])),
        mean_affine_residual_top7_variance=float(np.mean([r["affine_residual_top7_variance"] for r in rows])),
        mean_affine_residual_pc1_abs_corr_old_global=float(np.mean([r["affine_residual_pc1_abs_corr_old_global"] for r in rows])))
    atomic_json(OUT / "diagnostic.json", dict(n_subjects=len(rows), purpose="Paired REST input diagnosis only, not a separate population replication", networks=NETWORKS,
        methods=dict(pca="covariance PC1 fitted on first 900 frames, matching frozen analysis", dynamics="p=3 delta-Ridge alpha=1; independent unit Gaussian source; affine Gaussian TM", tolerance_bits=TOL,
            power_units="cycles/frame; TR unavailable; fixed power cutoff 0.1 cycles/frame", transfer="spatial loadings exchanged between matched old/new subjects, no parameter tuning", aggregates="arithmetic subject means; no hypothesis tests"),
        manuscript=dict(parent="P6UJCVG8", main="DXGC7JEA", supplement="MWIWKSVG", version="Only available attached full texts; explicit draft date/version unavailable; rechecked 2026-10-07", locations=["SI S1.2 Eqs S7-S11", "SI S5 Eqs S49-S50", "SI S12.2.2 pp23-25"]), summary=summary, rows=rows))
    print(json.dumps(summary, indent=2))


def plot():
    import matplotlib.pyplot as plt
    p = json.loads((OUT / "diagnostic.json").read_text())
    rows, s = p["rows"], p["summary"]
    names = ("Visual", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Control", "Default")
    colors = ("#4877A0", "#CF7C43")
    style = {"font.family":"sans-serif", "font.sans-serif":["Arial", "DejaVu Sans"],
        "font.size":9, "axes.spines.top":False, "axes.spines.right":False,
        "axes.labelsize":10, "xtick.labelsize":8, "ytick.labelsize":8}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(10.1, 6.4), layout="constrained")
        a, b, c, d = axes.ravel()
        old=np.array([r["variants"]["old_native"]["system_xi_nats"] for r in rows])
        new=np.array([r["variants"]["new_native"]["system_xi_nats"] for r in rows])
        for x, y in zip(old, new):
            a.plot([0, 1], [x, y], color=".8", lw=.6, alpha=.7, zorder=1)
        rng=np.random.default_rng(20261007)
        for i, values in enumerate((old, new)):
            a.scatter(i+rng.uniform(-.055, .055, len(values)), values,
                s=13, color=colors[i], alpha=.65, edgecolors="none", zorder=2)
            a.scatter(i, values.mean(), marker="D", facecolor="white", edgecolor=".15", s=48, zorder=3)
        a.set(xticks=[0,1], xticklabels=["Old REST", "New REST"], xlim=(-.25,1.25),
            ylim=(0,None), ylabel="System $\\Xi$ (nats)")
        for ax, field, factor, ylabel in ((b, "pc1_explained", 100, "PC1 explained variance (%)"),
            (c, "pc1_lag1", 1, "PC1 lag-one correlation")):
            x=np.arange(7)
            for i, side in enumerate(("old", "new")):
                values=np.array([[r[side][j][field] for j in range(7)] for r in rows])*factor
                ax.errorbar(x+(i-.5)*.15, values.mean(0), yerr=values.std(0,ddof=1)/np.sqrt(len(rows)),
                    color=colors[i], marker=("o", "s")[i], markersize=4, ls="none", capsize=2,
                    label=("Old REST", "New REST")[i])
            ax.set(xticks=x, xticklabels=names, ylabel=ylabel, ylim=(0,60 if factor==100 else 1.03))
            ax.tick_params(axis="x", rotation=30)
            for label in ax.get_xticklabels(): label.set_ha("right")
        matrix=np.array([[s["variants"]["old_native"]["system_xi_nats"],
            s["variants"]["old_with_new_loadings"]["system_xi_nats"]],
            [s["variants"]["new_with_old_loadings"]["system_xi_nats"],
            s["variants"]["new_native"]["system_xi_nats"]]])
        im=d.imshow(matrix, cmap="Blues", vmin=0, vmax=5, aspect="auto")
        for i in range(2):
            for j in range(2):
                d.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",
                    color="white" if matrix[i,j]>3 else "#12304A",fontsize=15)
        d.set(xticks=[0,1], xticklabels=["Old loadings", "New loadings"],
            yticks=[0,1], yticklabels=["Old signal", "New signal"], xlabel="Network PCA spatial weights")
        fig.colorbar(im, ax=d, label="Mean system $\\Xi$ (nats)", shrink=.82, pad=.03)
        fig.legend(*b.get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
        for label, ax in zip("abcd", axes.ravel()):
            ax.text(-.12,1.03,label,transform=ax.transAxes,fontweight="bold",fontsize=12)
        fig.savefig(OUT / "rest_input_diagnostic.png", dpi=260, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot()
    else:
        main()
        plot()
