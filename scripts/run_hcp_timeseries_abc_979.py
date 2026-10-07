#!/usr/bin/env python3
"""Frozen 979-person HCP A-C replication and one centered REST length control.

Native bits are retained; figures/statistical summaries convert to nats.
No tuning, behavioral analysis, subset report, or Syn projection is performed.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_hcp_schaefer500_yeo7_network_attribution import NETWORK_ORDER, exact_shapley
from scripts.run_hcp_schaefer500_all_tasks_phi import development_end_for_length
from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import load_yeo7_groups
from scripts.run_hcp_schaefer500_yeo7_pc1_phi_null import fit_delta_history_phi
from scripts.spt import (SPTConfig, TableXiOracle, all_nonempty_subsets,
                         build_spt, flatten_nodes, nontrivial_bipartitions)

STATES = ("REST", "EMOTION", "LANGUAGE", "MOTOR", "WM")
TASKS = STATES[1:]
LENGTHS = dict(REST=1200, EMOTION=176, LANGUAGE=316, MOTOR=284, WM=405)
VARIANTS = (*STATES, *(f"REST_MATCH_{task}" for task in TASKS))
NETWORKS = tuple(NETWORK_ORDER)
SUBSETS = all_nonempty_subsets(NETWORKS)
TOL = 1e-4
FLOOR = 1e-6
LN2 = math.log(2)
SEED = 20261006
CONTRACT = ROOT / "docs/log/hcp_timeseries_abc_replication_contract_20261006.json"
AUDIT = ROOT / "docs/log/hcp_timeseries_integrity_20261006.json"
LABELS = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_30/_atlas_labels/Schaefer2018_1000Parcels_7Networks_order.txt"
OLD_TASK = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_task_lr_feat_timeseries_57_brain"
OLD_ARRAYS = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full/k1_p3_a1/arrays.npz"
DEFAULT_OUT = ROOT / "results/hcp_timeseries_abc_replication_979"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def append_rows(path, rows):
    with Path(path).open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def file_identity(path):
    p = Path(path)
    s = p.stat()
    return dict(path=str(p.resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                sha256=sha(p))


def prepare(out):
    contract = json.loads(CONTRACT.read_text())
    subjects = contract["subjects"]
    if len(subjects) != 979 or len(set(subjects)) != 979:
        raise ValueError("The approved cohort must contain exactly 979 unique IDs")
    methods = {k: contract[k] for k in ("representation", "transition_model", "units",
               "numerical_validity", "spt", "length_sensitivity", "statistics")}
    implementation = {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__), ROOT / "scripts/spt.py",
        ROOT / "scripts/run_hcp_schaefer500_yeo7_pc1_phi_null.py",
        ROOT / "scripts/run_hcp_schaefer500_all_tasks_phi.py")}
    config_hash = digest(dict(methods=methods, implementation=implementation, labels=sha(LABELS)))
    inputs = {}
    print("Verifying immutable input/configuration fingerprints", flush=True)
    for subject in subjects:
        data_path = Path(contract["data_root"]) / f"{subject}.mat"
        designs = {task: file_identity(Path(contract["design_root"]) / f"sub-{subject}/{task}_LR_design.tsv") for task in TASKS}
        inp = dict(data=file_identity(data_path), designs=designs)
        inp["fingerprint"] = digest(dict(config=config_hash, inputs=inp))
        inputs[subject] = inp
    manifest = dict(subjects=subjects, states=STATES, variants=VARIANTS, config_hash=config_hash,
                    inputs=inputs, methods=methods, implementation=implementation,
                    labels=file_identity(LABELS), native_units="bits", display_units="nats",
                    no_subset_report=True, input_verification="full-file SHA256 before cache reuse")
    atomic_json(out / "manifest.json", manifest)
    return manifest


def load_records(out, manifest):
    rows = {}
    path = out / "records.jsonl"
    if not path.exists():
        return rows
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                # Recover a write interrupted before fsync; never accept a partial record.
                path.write_text("\n".join(lines[:index]) + "\n")
                break
            raise
        subject = row["subject"]
        if (subject in manifest["inputs"] and
            row["fingerprint"] == manifest["inputs"][subject]["fingerprint"]):
            key = subject, row["variant"]
            if key in rows:
                raise ValueError(f"Duplicate matching checkpoint {key}")
            rows[key] = row
    return rows


def audit_nonnegative(values, context):
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"Nonfinite {context}")
    count = int(np.sum(a < -TOL))
    if count:
        raise ValueError(f"Syn violation [{context}]: minimum={a.min():.12g} bits; "
                         f"threshold={-TOL:.12g}; affected_count={count}")
    return dict(minimum_bits=float(a.min()), near_zero_negative_count=int(np.sum(a < 0)),
                significant_negative_count=count, checked_count=int(a.size), tolerance_bits=TOL)


def decomposition(transition, noise):
    # Shared full target/channel and independent unit-variance source intervention.
    # Marginalizing omitted sources puts their response covariance in conditional noise.
    full_cov = transition @ transition.T + noise
    floor_counts = []

    def logdet(a):
        eigenvalues = np.linalg.eigvalsh((a + a.T) / 2)
        if not np.isfinite(eigenvalues).all():
            raise ValueError("Nonfinite covariance eigenvalues")
        floor_counts.append(int(np.sum(eigenvalues < FLOOR)))
        return float(np.log(np.maximum(eigenvalues, FLOOR)).sum())

    full_logdet = logdet(full_cov)

    def ei(indices):
        omitted = [i for i in range(21) if i not in set(indices)]
        conditional = noise.copy()
        if omitted:
            conditional += transition[:, omitted] @ transition[:, omitted].T
        return (full_logdet - logdet(conditional)) / (2 * LN2)

    module_indices = {name: [lag * 7 + i for lag in range(3)] for i, name in enumerate(NETWORKS)}
    table = {subset: ei([i for name in subset for i in module_indices[name]]) for subset in SUBSETS}
    scalar = np.asarray([ei([i]) for i in range(21)])
    scalar_by_network = {name: float(scalar[module_indices[name]].sum()) for name in NETWORKS}
    within = {name: float(table[(name,)] - scalar_by_network[name]) for name in NETWORKS}
    oracle = TableXiOracle(table, NETWORKS)
    cross_values = [oracle.xi(subset) for subset in SUBSETS]
    candidate_values = [oracle.xi(c) - oracle.xi(l) - oracle.xi(r)
                        for c in SUBSETS if len(c) > 1 for l, r in nontrivial_bipartitions(c)]
    system = float(table[NETWORKS] - scalar.sum())
    qc = dict(coalition_cross_xi=audit_nonnegative(cross_values, "all coalition Xi"),
              all_candidate_syn=audit_nonnegative(candidate_values, "all 966 bipartitions"),
              within_network_xi=audit_nonnegative(list(within.values()), "within-network Xi"),
              system_xi=audit_nonnegative([system], "system Xi"))
    tree = build_spt(NETWORKS, oracle, config=SPTConfig(syn_tolerance=TOL, complete_to_singletons=True))
    nodes = []
    order_mass = np.zeros(6)
    for node in flatten_nodes(tree.root):
        nodes.append(dict(sources=list(node.sources), xi_bits=node.xi_value,
                          syn_bits=node.syn_value, depth=node.depth, split_kind=node.split_kind,
                          children=[list(x.sources) for x in node.children]))
        if node.children:
            order_mass[node.order - 2] += node.syn_value
    shapley = exact_shapley(NETWORKS, lambda subset: oracle.xi(subset))
    contribution = np.asarray([within[name] + shapley[name] for name in NETWORKS])
    qc["attribution"] = audit_nonnegative(contribution, "network attribution")
    errors = dict(order_mass_minus_cross=float(order_mass.sum() - oracle.xi(NETWORKS)),
                  within_plus_cross_minus_system=float(sum(within.values()) + oracle.xi(NETWORKS) - system),
                  attribution_minus_system=float(contribution.sum() - system),
                  shapley_minus_cross=float(sum(shapley.values()) - oracle.xi(NETWORKS)))
    if max(abs(v) for v in errors.values()) > TOL:
        raise ValueError(f"Information closure failed {errors}")
    if system <= TOL or oracle.xi(NETWORKS) <= TOL:
        raise ValueError("Cannot normalize a zero/near-zero system or cross-network Xi")
    return dict(joint_ei_bits=table[NETWORKS], system_xi_bits=system,
                cross_network_xi_bits=oracle.xi(NETWORKS),
                module_ei_bits={name: table[(name,)] for name in NETWORKS},
                scalar_ei_bits=scalar.tolist(), within_network_xi_bits=within,
                coalition_ei_bits={"+".join(c): v for c, v in table.items()},
                network_contribution_bits=contribution.tolist(),
                network_percent=(100 * contribution / system).tolist(),
                order_mass_bits=order_mass.tolist(),
                order_share=(order_mass / oracle.xi(NETWORKS)).tolist(),
                nodes=nodes, spt_audit=asdict(tree.audit), numerical_audit=qc,
                identity_errors_bits=errors,
                covariance_floor=dict(threshold=FLOOR, affected_eigenvalue_count=sum(floor_counts),
                                      covariance_evaluations=len(floor_counts)))


def project(fitting, raw, groups):
    end = development_end_for_length(len(raw))
    scores, means, components, explained = [], [], [], []
    for indices in groups.values():
        model = PCA(n_components=1, svd_solver="full").fit(fitting[:end, indices])
        scores.append(model.transform(raw[:, indices]))
        means.append(model.mean_)
        components.append(model.components_[0])
        explained.append(float(model.explained_variance_ratio_[0]))
    return np.concatenate(scores, axis=1), np.concatenate(means), np.concatenate(components), explained


def fit_series(raw, fitting, groups, subject, variant, fingerprint):
    scores, means, components, explained = project(fitting, raw, groups)
    end = development_end_for_length(len(raw))
    fitted = fit_delta_history_phi(scores, alpha=1., order=3, development_end=end)
    row = dict(subject=subject, variant=variant, fingerprint=fingerprint,
               n_timepoints=len(raw), development_end=end, pca_explained=explained,
               heldout=fitted["heldout"], noise_condition=float(np.linalg.cond(fitted["noise_covariance"])),
               **decomposition(fitted["transition"], fitted["noise_covariance"]))
    arrays = {f"{variant}__{name}": value for name, value in dict(scores=scores,
              pca_mean=means, pca_components=components, transition=fitted["transition"],
              noise=fitted["noise_covariance"]).items()}
    return row, arrays


def task_component(raw, design_identity):
    design = np.loadtxt(design_identity["path"], ndmin=2)
    if len(design) != len(raw) or not np.isfinite(design).all():
        raise ValueError("Task design shape/finite-value mismatch")
    design = np.column_stack([np.ones(len(design)), design])
    component = design @ np.linalg.lstsq(design, raw, rcond=None)[0]
    return component


def read_data(identity, variants):
    keys = set()
    for v in variants:
        keys.add("Schaefer1000_NoGSR" if v.startswith("REST") else f"{v}_LR_Schaefer1000_NoGSR")
    data = loadmat(identity["path"], variable_names=sorted(keys), verify_compressed_data_integrity=True)
    result = {}
    for key in keys:
        raw = np.asarray(data[key], dtype=float)
        state = "REST" if key == "Schaefer1000_NoGSR" else key.split("_LR_")[0]
        if raw.shape != (LENGTHS[state], 1000) or not np.isfinite(raw).all():
            raise ValueError(f"Input mismatch {identity['path']} {key}: {raw.shape}")
        result[state] = raw
    return result


def save_subject_cache(out, subject, arrays, fingerprint):
    path = Path(out) / "pc1_cache" / f"{subject}.npz"
    merged = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as a:
            if str(a["fingerprint"]) == fingerprint:
                merged = {k: a[k] for k in a.files}
    merged.update(arrays)
    merged["fingerprint"] = np.asarray(fingerprint)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **merged)
    os.replace(tmp, path)


def subject_worker(subject, inp, variants, out):
    groups = load_yeo7_groups(LABELS, expected_parcels=1000)
    data = read_data(inp["data"], variants)
    rows, arrays = [], {}
    for variant in variants:
        if variant.startswith("REST"):
            raw = data["REST"]
            start = 0
            if variant != "REST":
                task = variant.removeprefix("REST_MATCH_")
                start = (1200 - LENGTHS[task]) // 2
                raw = raw[start:start + LENGTHS[task]]
            fitting = raw
        else:
            raw = data[variant]
            fitting = task_component(raw, inp["designs"][variant])
            start = 0
        row, cache = fit_series(raw, fitting, groups, subject, variant, inp["fingerprint"])
        row["rest_window_start"] = start if variant.startswith("REST") else None
        rows.append(row)
        arrays.update(cache)
    save_subject_cache(out, subject, arrays, inp["fingerprint"])
    return rows


def correlation_columns(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return np.sum(a * b, axis=0) / np.sqrt(np.sum(a*a, axis=0) * np.sum(b*b, axis=0))


def calibration_worker(subject, inp, out, frozen):
    groups = load_yeo7_groups(LABELS, expected_parcels=1000)
    data = read_data(inp["data"], TASKS)
    records, calibration, arrays = [], [], {}
    for task in TASKS:
        old = loadmat(OLD_TASK / f"sub-{subject}/{task}_LR.mat",
                      variable_names=["Schaefer1000_taskRetained", "Schaefer1000_taskRegressed"])
        retained = np.asarray(old["Schaefer1000_taskRetained"], float)
        regressed = np.asarray(old["Schaefer1000_taskRegressed"], float)
        component = task_component(retained, inp["designs"][task])
        error = float(np.linalg.norm(retained - component - regressed) / np.linalg.norm(regressed))
        if error > 1e-6:
            raise ValueError(f"Old task-regression reconstruction mismatch {subject}/{task}: {error}")
        old_row, old_cache = fit_series(retained, retained-regressed, groups, subject, task, inp["fingerprint"])
        cache_error = float(old_row["system_xi_bits"] - frozen[task])
        if abs(cache_error) > TOL:
            raise ValueError(f"Legacy metric reproduction mismatch {subject}/{task}: {cache_error} bits")
        raw = data[task]
        new_row, new_cache = fit_series(raw, task_component(raw, inp["designs"][task]),
                                        groups, subject, task, inp["fingerprint"])
        x, y = raw-raw.mean(axis=0), retained-retained.mean(axis=0)
        slopes = np.sum(x*y, axis=0) / np.sum(x*x, axis=0)
        affine_error = float(np.linalg.norm(y-x*slopes) / np.linalg.norm(y))
        if affine_error > 1e-4:
            raise ValueError(f"New/old parcel ordering or signal correspondence failed {subject}/{task}: {affine_error}")
        pcs = np.abs(correlation_columns(new_cache[f"{task}__scores"], old_cache[f"{task}__scores"]))
        calibration.append(dict(subject=subject, task=task, regression_relative_error=error,
            old_metric_cache_error_bits=cache_error, roi_affine_error=affine_error,
            roi_scaling_range=[float(slopes.min()), float(slopes.max())],
            pca_score_abs_correlations=pcs.tolist(),
            system_xi_new_minus_old_bits=new_row["system_xi_bits"]-old_row["system_xi_bits"],
            network_percent_new_minus_old=(np.asarray(new_row["network_percent"])-old_row["network_percent"]).tolist()))
        records.append(new_row)
        arrays.update(new_cache)
    save_subject_cache(out, subject, arrays, inp["fingerprint"])
    return records, calibration


def calibrate(out, manifest, workers):
    previous = out / "calibration.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        if saved.get("config_hash") == manifest["config_hash"] and saved.get("input_hash") == digest(manifest["inputs"]):
            print("57-person calibration reused after matching full input fingerprints", flush=True)
            return
    old_ids = json.loads(AUDIT.read_text())["old_cohort_subjects"]
    with np.load(OLD_ARRAYS, allow_pickle=False) as a:
        ids = [str(s).removeprefix("sub-") for s in a["subjects"]]
        states = a["states"].astype(str).tolist()
        frozen = {s: {t: float(a["system_xi"][states.index(t), ids.index(s)]) for t in TASKS} for s in old_ids}
    if len(old_ids) != 57 or not set(old_ids) <= set(manifest["subjects"]):
        raise ValueError("Calibration cohort mismatch")
    all_rows, all_checks = [], []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(calibration_worker, s, manifest["inputs"][s], str(out), frozen[s]) for s in old_ids]
        for future in as_completed(futures):
            records, checks = future.result()
            all_rows.extend(records)
            all_checks.extend(checks)
    current = load_records(out, manifest)
    append_rows(out / "records.jsonl", [r for r in all_rows if (r["subject"], r["variant"]) not in current])
    summary = dict(status="passed", n_subjects=57, n_task_checks=len(all_checks),
        elapsed_seconds=time.monotonic()-started, config_hash=manifest["config_hash"],
        input_hash=digest(manifest["inputs"]),
        maximum_regression_relative_error=max(r["regression_relative_error"] for r in all_checks),
        maximum_legacy_metric_error_bits=max(abs(r["old_metric_cache_error_bits"]) for r in all_checks),
        maximum_roi_affine_error=max(r["roi_affine_error"] for r in all_checks),
        minimum_pca_score_abs_correlation=min(min(r["pca_score_abs_correlations"]) for r in all_checks),
        rows=all_checks)
    atomic_json(previous, summary)
    print(f"Calibration passed: 57 people / 228 tasks; {summary['elapsed_seconds']:.1f}s", flush=True)


def compute(out, manifest, subjects, variants, workers, label):
    current = load_records(out, manifest)
    jobs = [(s, [v for v in variants if (s, v) not in current]) for s in subjects]
    jobs = [(s, v) for s, v in jobs if v]
    count = sum(len(v) for s, v in jobs)
    print(f"{label}: {count} uncached models across {len(jobs)} people", flush=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(subject_worker, s, manifest["inputs"][s], v, str(out)): s for s, v in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            append_rows(out / "records.jsonl", rows)
            if i == 1 or i % 25 == 0 or i == len(jobs):
                print(f"{label}: saved {i}/{len(jobs)} subject batches; {time.monotonic()-started:.1f}s", flush=True)
    elapsed = time.monotonic()-started
    return dict(new_models=count, elapsed_seconds=elapsed,
                aggregate_seconds_per_model=elapsed/count if count else None)


def bh(values):
    p = np.asarray(values)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def paired(left, right, seed, unit="nats"):
    delta = np.asarray(left)-np.asarray(right)
    p = float(wilcoxon(delta, alternative="two-sided", method="approx").pvalue) if np.any(delta) else 1.
    rng = np.random.default_rng(seed)
    # Participants, not frames, are the resampling unit. Family IDs remain unavailable.
    means = np.empty(5000)
    for start in range(0, 5000, 250):
        idx = rng.integers(0, len(delta), size=(min(250,5000-start), len(delta)))
        means[start:start+len(idx)] = delta[idx].mean(axis=1)
    return dict(mean_difference=float(delta.mean()), median_difference=float(np.median(delta)),
                mean_difference_ci95=np.quantile(means,[.025,.975]).tolist(),
                positive_pair_fraction=float(np.mean(delta>0)), p=p, unit=unit)


def collect(out, manifest):
    rows = load_records(out, manifest)
    expected = {(s,v) for s in manifest["subjects"] for v in VARIANTS}
    if set(rows) != expected:
        raise ValueError(f"Missing/extra result records: missing={len(expected-set(rows))}, extra={len(set(rows)-expected)}")
    arrays = {key: np.asarray([[rows[s,v][field] for s in manifest["subjects"]] for v in VARIANTS])
              for key,field in dict(system_xi_bits="system_xi_bits", cross_xi_bits="cross_network_xi_bits",
                  order_mass_bits="order_mass_bits", order_share="order_share", network_percent="network_percent").items()}
    for a in arrays.values():
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite final arrays")
    if np.max(np.abs(arrays["network_percent"].sum(axis=2)-100)) > 1e-6:
        raise ValueError("Network percent does not close to 100")
    arrays.update(subjects=np.asarray(manifest["subjects"]), variants=np.asarray(VARIANTS), networks=np.asarray(NETWORKS))
    np.savez_compressed(out / "arrays.npz", **arrays)
    summary = dict(status="complete", n_subjects=979, native_units="bits", figure_units="nats",
                   fit_count=len(rows), full_length={}, length_matched={}, by_variant={},
                   limitations=["New REST run identifier and denoising provenance unresolved",
                                "Assumes new-only parcel order matches the supplied Schaefer1000 order",
                                "Family dependence and motion not controlled in participant-level inference",
                                "OLS task preprocessing uses the whole run; holdout is prediction diagnostic only",
                                "One fixed central REST window; not a 12-window control",
                                "Linear Gaussian approximation does not establish irreducible physical high-order coupling"])
    for i,v in enumerate(VARIANTS):
        mass = arrays["order_mass_bits"][i]
        peaks = np.argmax(mass,axis=1)+2
        summary["by_variant"][v] = dict(mean_system_xi_nats=float(arrays["system_xi_bits"][i].mean()*LN2),
            mean_system_xi_bits=float(arrays["system_xi_bits"][i].mean()),
            mean_order_mass_nats=(mass.mean(axis=0)*LN2).tolist(),
            mean_order_share=arrays["order_share"][i].mean(axis=0).tolist(),
            group_peak_order=int(np.argmax(mass.mean(axis=0))+2),
            participant_peak_order_fractions={str(k):float(np.mean(peaks==k)) for k in range(2,8)},
            mean_order6_minus_order7_nats=float((mass[:,4]-mass[:,5]).mean()*LN2),
            mean_network_percent=arrays["network_percent"][i].mean(axis=0).tolist())
    for mode in ("full_length","length_matched"):
        panel_a, panel_b, secondary, panel_c = [], [], [], []
        for j, task in enumerate(TASKS,1):
            ri = 0 if mode=="full_length" else VARIANTS.index(f"REST_MATCH_{task}")
            panel_a.append(dict(task=task, **paired(arrays["system_xi_bits"][ri]*LN2, arrays["system_xi_bits"][j]*LN2, SEED+j)))
            panel_b.append(dict(task=task, **paired(arrays["order_mass_bits"][ri,:,5]*LN2, arrays["order_mass_bits"][j,:,5]*LN2, SEED+10+j)))
            secondary.append(dict(task=task, **paired(arrays["order_share"][ri,:,5], arrays["order_share"][j,:,5], SEED+20+j, "fraction")))
            for network in ("Cont","Limbic"):
                k = NETWORKS.index(network)
                panel_c.append(dict(task=task, network=network, contrast="task minus REST",
                    **paired(arrays["network_percent"][j,:,k], arrays["network_percent"][ri,:,k], SEED+100+10*j+k, "percentage points")))
        for family in (panel_a,panel_b,secondary,panel_c):
            for row,q in zip(family,bh([r["p"] for r in family])):
                row["q"] = float(q)
        summary[mode] = dict(panel_A=panel_a, panel_B_order7=panel_b,
                             panel_B_normalized_order7=secondary, panel_C_prespecified=panel_c)
    category_names = list(next(iter(rows.values()))["numerical_audit"])
    summary["numerical_audit"] = {cat:dict(
        minimum_bits=min(r["numerical_audit"][cat]["minimum_bits"] for r in rows.values()),
        near_zero_negative_count=sum(r["numerical_audit"][cat]["near_zero_negative_count"] for r in rows.values()),
        checked_count=sum(r["numerical_audit"][cat]["checked_count"] for r in rows.values()),
        significant_negative_count=0, tolerance_bits=TOL) for cat in category_names}
    summary["maximum_identity_error_bits"] = max(abs(v) for r in rows.values() for v in r["identity_errors_bits"].values())
    summary["covariance_floor_affected_count"] = sum(r["covariance_floor"]["affected_eigenvalue_count"] for r in rows.values())
    summary["heldout_diagnostics"] = {v:dict(
        median_skill_ratio=float(np.median([rows[s,v]["heldout"]["skill_ratio"] for s in manifest["subjects"]])),
        fraction_better_than_persistence=float(np.mean([rows[s,v]["heldout"]["skill_ratio"]<1 for s in manifest["subjects"]]))) for v in VARIANTS}
    atomic_json(out / "summary.json", summary)
    from scripts.plot_hcp_timeseries_abc_979 import render_results
    render_results(out, arrays, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage", choices=("calibrate","smoke","run","report"), default="run")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    (out / "pc1_cache").mkdir(exist_ok=True)
    started = time.monotonic()
    status = dict(state="running", stage=args.stage, pid=os.getpid(), expected_models=8811,
                  report_cohort=979, workers=args.workers)
    atomic_json(out / "run_status.json",status)
    try:
        manifest = prepare(out)
        if args.stage=="report":
            collect(out,manifest)
        else:
            calibrate(out,manifest,args.workers)
            if args.stage!="calibrate":
                smoke_ids = manifest["subjects"][:20]
                smoke = compute(out,manifest,smoke_ids,VARIANTS,args.workers,"20-person smoke")
                atomic_json(out / "smoke.json",dict(status="passed",subjects=smoke_ids,**smoke))
                if args.stage=="run":
                    status["stage"]="main"
                    atomic_json(out / "run_status.json",status)
                    compute(out,manifest,manifest["subjects"],STATES,args.workers,"Main five states")
                    status["stage"]="length_control"
                    atomic_json(out / "run_status.json",status)
                    compute(out,manifest,manifest["subjects"],VARIANTS[5:],args.workers,"Fixed REST length control")
                    status["stage"]="analysis_and_figures"
                    atomic_json(out / "run_status.json",status)
                    collect(out,manifest)
        status.update(state="complete",elapsed_seconds=time.monotonic()-started)
        atomic_json(out / "run_status.json",status)
        print(f"Requested stage complete in {status['elapsed_seconds']:.1f}s",flush=True)
    except BaseException as exc:
        status.update(state="failed",error=f"{type(exc).__name__}: {exc}",elapsed_seconds=time.monotonic()-started)
        atomic_json(out / "run_status.json",status)
        raise


if __name__=="__main__":
    main()
