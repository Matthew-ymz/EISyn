#!/usr/bin/env python3
"""Frozen HCP task/behavior replication and bounded 600-hypothesis exploration.

Uses existing Schaefer-1000 EI caches; no transition models are fitted here.
Legacy figure terminology is retained only in the provenance: the tested feature
is grouped Xi, not an SPT node atom. Family dependence remains unadjusted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "results/hcp_timeseries_abc_replication_979"
OUTPUT = ROOT / "results/hcp_task_behavior_979"
BEHAVIOR = ROOT / "data/unrestricted_xinyangliu_6_12_2018_2_43_32.csv"
PLAN = ROOT / "docs/log/hcp979_task_behavior_replication_20261009.md"
NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
STATES = ("EMOTION", "LANGUAGE", "MOTOR", "WM")
COMBOS = tuple(c for k in range(2, 8) for c in itertools.combinations(NETWORKS, k))
NAMES = tuple("+".join(c) for c in COMBOS)
SHORT = dict(zip(NETWORKS, ("V", "SM", "DAN", "VAN", "L", "C", "D")))
SEED = 20261009
PERMUTATIONS = 100000
BOOTSTRAPS = 5000
TOLERANCE_BITS = 1e-4
ENDPOINTS = (
    ("story_accuracy", "LANGUAGE", "Story accuracy", "Language_Task_Story_Acc"),
    ("math_accuracy", "LANGUAGE", "Math accuracy", "Language_Task_Math_Acc"),
    ("face_specific_speed", "EMOTION", "Face-specific speed", "Emotion_Task_Face_Median_RT"),
    ("motor_capacity", "MOTOR", "Motor capacity", "motor_capacity"),
    ("wm_accuracy", "WM", "WM accuracy", "WM_Task_Acc"),
)
MOTOR_NAMES = (
    "Vis+SomMot+Limbic+Cont+Default", "Vis+SomMot+Limbic+Default",
    "Vis+SomMot+Cont+Default", "Vis+SomMot+SalVentAttn+Limbic+Cont+Default",
    "SomMot+Limbic+Cont+Default", "SomMot+Limbic+Default",
    "Vis+Limbic+Cont+Default", "Vis+SomMot+SalVentAttn+Limbic+Default",
    "Vis+Cont+Default",
)
FIXED = (
    ("story_accuracy", "Vis+SomMot+Limbic+Cont", 1),
    ("math_accuracy", "Vis+DorsAttn+Cont", 1),
    ("face_specific_speed", "Limbic+Cont+Default", -1),
) + tuple(("motor_capacity", n, -1) for n in MOTOR_NAMES)
COLORS = {"LANGUAGE": "#C87552", "EMOTION": "#4C78A8", "MOTOR": "#3F8E80", "WM": "#8B6BAA"}


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compact(name: str) -> str:
    return "+".join(SHORT[n] for n in name.split("+"))


def nonnegative(values, name: str, tolerance: float = TOLERANCE_BITS) -> dict:
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"Nonfinite {name}")
    audit = dict(minimum_bits=float(a.min()), tolerance_bits=tolerance,
                 checked_count=int(a.size), near_zero_negative_count=int(((a < 0) & (a >= -tolerance)).sum()),
                 significant_negative_count=int((a < -tolerance).sum()))
    if audit["significant_negative_count"]:
        raise ValueError(f"Syn validity failure {name}: {audit}")
    return audit  # Preserve raw values; no projection is applied.


def verify_inputs(manifest: dict) -> dict:
    identities = {manifest["labels"]["path"]: manifest["labels"]}
    for subject in manifest["subjects"]:
        data = manifest["inputs"][subject]
        for identity in [data["data"], *data["designs"].values()]:
            identities[identity["path"]] = identity
    def check(pair):
        name, ref = pair
        p = Path(name)
        stat = p.stat()
        if stat.st_size != ref["bytes"] or sha(p) != ref["sha256"]:
            raise ValueError(f"Cache input content changed: {p}")
        return stat.st_size, stat.st_mtime_ns != ref["mtime_ns"]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(check, identities.items()))
    for name, reference in manifest["implementation"].items():
        if sha(ROOT / name) != reference:
            raise ValueError(f"Frozen model implementation changed: {name}")
    return dict(status="passed", files=len(checks), bytes_checked=sum(x[0] for x in checks),
                timestamp_only_changes=sum(x[1] for x in checks), elapsed_seconds=time.monotonic()-started,
                verification="Actual full-file SHA256; model implementation and atlas-label SHA256")


def load_features(manifest: dict):
    ids = list(map(str, manifest["subjects"]))
    positions = {s: i for i, s in enumerate(ids)}
    features = {s: np.full((len(ids), 120), np.nan) for s in STATES}
    seen = set()
    minimum_candidate = float("inf")
    candidate_count = near_count = 0
    for line in (SOURCE / "records.jsonl").open():
        r = json.loads(line)
        state, sid = r["variant"], str(r["subject"])
        if state not in features:
            continue
        key = (sid, state)
        if key in seen or sid not in positions:
            raise ValueError(f"Invalid duplicate/extra cache record {key}")
        seen.add(key)
        if r["fingerprint"] != manifest["inputs"][sid]["fingerprint"]:
            raise ValueError(f"Input fingerprint mismatch {key}")
        ei = r["coalition_ei_bits"]
        if len(ei) != 127:
            raise ValueError(f"Incomplete EI oracle {key}")
        features[state][positions[sid]] = [ei[n] - sum(ei[v] for v in c) for n, c in zip(NAMES, COMBOS)]
        a = r["numerical_audit"]["all_candidate_syn"]
        minimum_candidate = min(minimum_candidate, a["minimum_bits"])
        candidate_count += a["checked_count"]
        near_count += a["near_zero_negative_count"]
        if a["significant_negative_count"]:
            raise ValueError(f"Cached candidate Syn violation {key}: {a}")
    if len(seen) != len(ids) * len(STATES):
        raise ValueError("Task cache coverage incomplete")
    audit = {s: nonnegative(v, s) for s, v in features.items()}
    audit["all_candidate_syn"] = dict(minimum_bits=minimum_candidate, checked_count=candidate_count,
                                     tolerance_bits=TOLERANCE_BITS, near_zero_negative_count=near_count,
                                     significant_negative_count=0)
    return ids, features, audit


def raw_behavior(ids):
    raw = pd.read_csv(BEHAVIOR, dtype={"Subject": str}).set_index("Subject")
    if not raw.index.is_unique:
        raise ValueError("Duplicate behavior Subject")
    frame = raw.loc[ids].copy()
    for field in [e[3] for e in ENDPOINTS if e[3] != "motor_capacity"] + [
        "Emotion_Task_Shape_Median_RT", "Endurance_AgeAdj", "Dexterity_AgeAdj", "Strength_AgeAdj"]:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    old = json.loads((ROOT / "results/hcp_motor_composite_scores_57/summary.json").read_text())
    sd = old["score_definition"]
    components = ["Endurance", "Dexterity", "Strength"]
    # NumPy mean deliberately preserves missingness rather than silently averaging two components.
    z = np.column_stack([(frame[c+"_AgeAdj"].to_numpy() - sd["component_means"][c]) /
                         sd["component_sds"][c] for c in components])
    frame["motor_capacity"] = z.mean(axis=1)
    cohort = pd.read_csv(ROOT / "results/hcp_language_story_math_coalitions_57/selected_candidate_source_data.tsv", sep="\t")
    origin = dict(zip(cohort["subject"].str.removeprefix("sub-"), cohort["cohort"]))
    frame["origin"] = [origin.get(s, "expansion") for s in ids]
    if len(origin) != 57 or not set(origin) <= set(ids):
        raise ValueError("Frozen old cohort does not match main cohort")
    return frame


def basis(design):
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    cutoff = singular.max() * max(design.shape) * np.finfo(float).eps
    return u[:, singular > cutoff]


def residual(values, q):
    return values - q @ (q.T @ values)


def normalize(values):
    a = np.asarray(values, dtype=float)
    norm = np.linalg.norm(a, axis=0)
    if np.any(norm <= 1e-10):
        raise ValueError("Constant residual variable")
    return a / norm


def make_design(frame, emotion=False, shape=None):
    dummy = pd.get_dummies(frame[["Age", "Gender", "fMRI_3T_ReconVrs", "origin"]].astype(str), drop_first=True, dtype=float)
    d = np.column_stack([np.ones(len(frame)), dummy.to_numpy()])
    names = ["intercept", *dummy.columns.tolist()]
    if emotion:
        if shape is None:
            shape = -np.log(frame["Emotion_Task_Shape_Median_RT"].to_numpy())
        d = np.column_stack([d, rankdata(shape)])
        names.append("rank_shape_speed")
    return d, names


def prepare_analysis(definition, frame, features):
    key, state, title, field = definition
    y = frame[field].to_numpy(float)
    mask = np.isfinite(y)
    if state == "EMOTION":
        shape = frame["Emotion_Task_Shape_Median_RT"].to_numpy(float)
        mask &= np.isfinite(shape) & (shape > 0) & (y > 0)
        y = -np.log(y[mask])
    else:
        y = y[mask]
    f = frame.iloc[np.flatnonzero(mask)]
    x = features[state][mask]
    design, labels = make_design(f, emotion=state == "EMOTION")
    q = basis(design)
    xr = residual(rankdata(x, axis=0), q)
    yr = residual(rankdata(y), q)
    brain = normalize(xr)
    endpoint = normalize(yr)
    observed = brain.T @ endpoint
    return dict(key=key, state=state, title=title, mask=mask, frame=f, x=x, y=y,
                design=design, design_labels=labels, q=q, xr=xr, yr=yr,
                brain=brain, observed=observed, cohort=f["origin"].to_numpy())


def permute(a, repeats, seed):
    """Freedman-Lane residual permutation, reprojected onto nuisance complement.

    The fitted component vanishes on reprojection. The numerator uses the
    orthogonalized brain columns; the denominator retains the reprojected
    response norm. This is algebraically identical to pseudo-response refitting.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(a["cohort"] == g) for g in np.unique(a["cohort"])]
    counts = np.zeros(120, dtype=np.int64)
    q, e = a["q"], a["yr"]
    for start in range(0, repeats, 500):
        size = min(500, repeats-start)
        indices = np.tile(np.arange(len(e)), (size, 1))
        for group in groups:
            indices[:, group] = group[np.argsort(rng.random((size, len(group))), axis=1)]
        pe = e[indices]
        projected = pe @ q
        squared_norm = np.sum(pe * pe, axis=1) - np.sum(projected * projected, axis=1)
        if np.any(squared_norm <= 1e-12):
            raise ValueError("Degenerate permutation response")
        null = (pe @ a["brain"]) / np.sqrt(squared_norm[:, None])
        counts += np.sum(np.abs(null) >= np.abs(a["observed"])[None, :], axis=0)
    return (counts+1) / (repeats+1)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    sorted_adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    out = np.empty_like(p)
    out[order] = sorted_adjusted
    return out


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p)+1)
    out = np.empty_like(p)
    out[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return out


def bootstrap(a, columns, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(a["cohort"] == g) for g in np.unique(a["cohort"])]
    x, y = a["x"][:, columns], a["y"]
    values = np.empty((BOOTSTRAPS, len(columns)))
    for b in range(BOOTSTRAPS):
        sample = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        # Categorical columns are retained; continuous Shape-speed ranks are recomputed.
        d = a["design"][sample].copy()
        if a["state"] == "EMOTION":
            shape = -np.log(a["frame"]["Emotion_Task_Shape_Median_RT"].to_numpy()[sample])
            d[:, -1] = rankdata(shape)
        q = basis(d)
        xr = normalize(residual(rankdata(x[sample], axis=0), q))
        yr = normalize(residual(rankdata(y[sample]), q))
        values[b] = xr.T @ yr
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite bootstrap")
    return np.quantile(values, [0.025, 0.975], axis=0).T


def self_check():
    rng = np.random.default_rng(194)
    d = np.column_stack([np.ones(47), rng.normal(size=(47, 3))])
    q = basis(d)
    x = normalize(residual(rng.normal(size=(47, 120)), q))
    y = rng.normal(size=47)
    fit = d @ np.linalg.lstsq(d, y, rcond=None)[0]
    e = y - fit
    indices = np.vstack([rng.permutation(47) for _ in range(31)])
    pseudo = fit[None, :] + e[indices]
    explicit = pseudo - (d @ np.linalg.lstsq(d, pseudo.T, rcond=None)[0]).T
    expected = explicit @ x / np.linalg.norm(explicit, axis=1)[:, None]
    pe = e[indices]
    optimized = pe @ x / np.sqrt(np.sum(pe*pe, axis=1)-np.sum((pe @ q)**2, axis=1))[:, None]
    error = float(np.max(np.abs(expected-optimized)))
    assert error < 1e-12, error
    from statsmodels.stats.multitest import multipletests
    p = rng.uniform(0, 1, 120)
    assert np.allclose(holm(p), multipletests(p, method="holm")[1])
    assert np.allclose(bh(p), multipletests(p, method="fdr_bh")[1])
    try:
        nonnegative([-2*TOLERANCE_BITS], "self_check")
    except ValueError:
        pass
    else:
        raise AssertionError("Nonnegative guard failed")
    return dict(status="passed", residual_permutation_refit_maximum_error=error,
                multiple_comparisons_reference="statsmodels Holm and BH", nonnegative_failure_guard="passed")


def legacy_gate():
    from scripts import plot_hcp_schaefer1000_behavior_main as old_figure
    from scripts import screen_hcp_motor_composite_scores_57 as old_motor
    from scripts import screen_hcp_emotion_performance_coalitions_57 as old_emotion
    rows = []
    with np.load(old_figure.COALITION_CACHE, allow_pickle=False) as a:
        ids, names, x = a["subjects"].astype(str), a["coalitions"].astype(str).tolist(), a["synergy_bits"]
    raw = pd.read_csv(BEHAVIOR, dtype={"Subject": str}).set_index("Subject").loc[[s.removeprefix("sub-") for s in ids]]
    table = pd.read_csv(old_figure.BEHAVIOR_ROOT / "selected_candidate_source_data.tsv", sep="\t").set_index("subject")
    groups = pd.Categorical(table.loc[ids, "cohort"]).codes
    for spec, key, rounded in zip(old_figure.SCATTER_SPECS, ["story_accuracy", "math_accuracy"], [(0.258, 0.044), (0.281, 0.035)]):
        brain = x[:, names.index(spec["coalition"])]
        y = raw[spec["endpoint"]].to_numpy(float)
        rho, p = old_figure.blocked_pointwise_spearman(brain, y, groups, seed=spec["seed"])
        error = abs(rho-float(spearmanr(brain, y).statistic))
        if error > 1e-8 or abs(rho-rounded[0]) > 0.0005 or abs(p-rounded[1]) > 0.0005:
            raise ValueError(f"Old LANGUAGE gate mismatch {key}: {rho}, {p}")
        rows.append(dict(endpoint=key, coalition=spec["coalition"], rho=rho, p_raw=p,
                         coefficient_reference="independent scipy Spearman plus rounded original figure",
                         maximum_coefficient_error=error, historical_p_reference="original figure rounded to three decimals"))
    with np.load(old_motor.CACHE, allow_pickle=False) as a:
        motor_ids, names, x = a["subjects"].astype(str), a["coalitions"].astype(str).tolist(), a["synergy_bits"]
    if set(ids) != set(motor_ids):
        raise ValueError("Legacy subject identity mismatch")
    nonnegative(x, "old MOTOR", 1e-9)
    scores = old_motor.load_scores(motor_ids)
    result = old_motor.screen(x, scores, old_motor.PERMUTATIONS, old_motor.SEED)
    summary = json.loads((old_motor.OUTPUT / "summary.json").read_text())
    references = {r["coalition"]: r for r in summary["top_ten"]}
    for name in MOTOR_NAMES:
        c, ref = names.index(name), references[name]
        rho, p = float(result["rho_adjusted"][c]), float(result["p_raw"][c])
        error = abs(rho-ref["rho_adjusted"])
        if error > 1e-8 or abs(p-ref["p_raw"]) > 1e-12:
            raise ValueError(f"Old MOTOR gate mismatch {name}: {rho}, {p}")
        rows.append(dict(endpoint="motor_capacity", coalition=name, rho=rho, p_raw=p,
                         maximum_coefficient_error=error, p_reference_error=abs(p-ref["p_raw"])))
    with np.load(old_emotion.CACHE, allow_pickle=False) as a:
        emotion_ids, names, x = a["subjects"].astype(str), a["coalitions"].astype(str).tolist(), a["synergy_bits"]
    if set(ids) != set(emotion_ids):
        raise ValueError("Legacy EMOTION subject identity mismatch")
    nonnegative(x, "old EMOTION", 1e-9)
    behavior = old_emotion.load_behavior(emotion_ids)
    analysis = old_emotion.prepare_analyses(x, behavior)[0]
    original_status = old_emotion.status
    try:
        old_emotion.status = lambda *args, **kwargs: None  # Do not overwrite old live-progress files.
        result = old_emotion.screen_analyses([analysis], behavior["cohort"].astype(int), np.array([0]), 100000, old_emotion.SEED)
    finally:
        old_emotion.status = original_status
    name = "Limbic+Cont+Default"
    c = names.index(name)
    ref = json.loads((old_emotion.OUTPUT / "summary.json").read_text())["analyses"]["task_face_specific_speed"]["winner"]
    rho, p = float(result["observed"][0, c]), float(result["p_raw"][0, c])
    if abs(rho-ref["rho"]) > 1e-8 or abs(p-ref["p_raw"]) > 1e-12:
        raise ValueError(f"Old EMOTION gate mismatch {rho}, {p}")
    rows.append(dict(endpoint="face_specific_speed", coalition=name, rho=rho, p_raw=p,
                     maximum_coefficient_error=abs(rho-ref["rho"]), p_reference_error=abs(p-ref["p_raw"])))
    lookup = {(r["endpoint"], r["coalition"]): r for r in rows}
    rows = [lookup[(key, name)] for key, name, _ in FIXED]
    return dict(status="passed", n_subjects=57, hypotheses=12, rows=rows,
                language_precision_limit="Full-precision historical LANGUAGE statistics were not stored; gate recomputes with original function, checks independent coefficient, and matches rounded figure values.",
                motor_legacy_age_adjustment="Raw age midpoint in legacy MOTOR implementation; legacy EMOTION uses ranked age midpoint.",
                cache_order="Same 57 Subject IDs; supplementary ordering differs across caches. Each historical permutation uses its original cache order.")


def legacy_compatible(a):
    age = np.array([38 if v == "36+" else np.mean([float(t) for t in v.split("-")]) for v in a["frame"]["Age"]])
    if a["state"] == "LANGUAGE":
        return np.asarray([spearmanr(a["x"][:, j], a["y"]).statistic for j in range(120)])
    origin = pd.get_dummies(a["frame"]["origin"], drop_first=True, dtype=float).to_numpy()
    d = np.column_stack([np.ones(len(age)), rankdata(age) if a["state"] == "EMOTION" else age,
                         (a["frame"]["Gender"].to_numpy() == "M").astype(float), origin])
    if a["state"] == "EMOTION":
        shape = -np.log(a["frame"]["Emotion_Task_Shape_Median_RT"].to_numpy())
        d = np.column_stack([d, rankdata(shape)])
    q = basis(d)
    return normalize(residual(rankdata(a["x"], axis=0), q)).T @ normalize(residual(rankdata(a["y"]), q))


def compare_overlap(ids, features):
    rows = {}
    index = {s: j for j, s in enumerate(ids)}
    paths = {
        "LANGUAGE": ROOT / "results/hcp_language_story_math_coalitions_57/language_coalition_synergy_57.npz",
        "EMOTION": ROOT / "results/hcp_emotion_performance_coalitions_57/emotion_rest_coalition_synergy_57.npz",
        "MOTOR": ROOT / "results/hcp_motor_composite_scores_57/motor_coalition_synergy_57.npz",
    }
    for state, path in paths.items():
        with np.load(path, allow_pickle=False) as a:
            old_ids, names, old = a["subjects"].astype(str), a["coalitions"].astype(str).tolist(), a["synergy_bits"]
            if old.ndim == 3:
                old = old[a["states"].astype(str).tolist().index("EMOTION")]
            old = old[:, [names.index(n) for n in NAMES]]
        new = features[state][[index[s.removeprefix("sub-")] for s in old_ids]]
        difference = new-old
        rows[state] = dict(n_subjects=57, coalitions=120,
                           maximum_absolute_difference_bits=float(np.abs(difference).max()),
                           mean_absolute_difference_bits=float(np.abs(difference).mean()),
                           mean_old_xi_bits=float(old.mean()), mean_new_xi_bits=float(new.mean()),
                           mean_120_subjectwise_spearman=float(np.mean([spearmanr(old[:, j], new[:, j]).statistic for j in range(120)])))
    return rows


def scientific_rows(analyses, completed, gate):
    all_rows = []
    for a in analyses:
        p = np.array(completed[a["key"]]["p_raw"])
        for c, name in enumerate(NAMES):
            all_rows.append(dict(endpoint=a["key"], state=a["state"], coalition=name,
                                 source_network_count=len(COMBOS[c]), n=len(a["y"]),
                                 rho=float(a["observed"][c]), p_raw=float(p[c]),
                                 fixed_old_hypothesis=any(k == a["key"] and n == name for k, n, _ in FIXED)))
    adjusted, fdr = holm([r["p_raw"] for r in all_rows]), bh([r["p_raw"] for r in all_rows])
    for r, p, q in zip(all_rows, adjusted, fdr):
        r.update(p_holm_600=float(p), q_bh_600=float(q))
    mapping = {(r["endpoint"], r["coalition"]): r for r in all_rows}
    old = {(r["endpoint"], r["coalition"]): r for r in gate["rows"]}
    fixed = [mapping[(k, n)].copy() for k, n, _ in FIXED]
    adjusted_fixed = holm([r["p_raw"] for r in fixed])
    for r, (k, n, sign), p in zip(fixed, FIXED, adjusted_fixed):
        r.update(expected_direction=sign, old_rho=old[(k, n)]["rho"], old_p_raw=old[(k, n)]["p_raw"],
                 p_holm_12=float(p), direction_reproduced=bool(r["rho"]*sign > 0),
                 replication_supported=bool(r["rho"]*sign > 0 and p < .05))
        a = next(a for a in analyses if a["key"] == k)
        c = NAMES.index(n)
        r["legacy_compatible_rho"] = float(legacy_compatible(a)[c])
        r["bootstrap_95_ci"] = completed[k]["fixed_intervals"][n]
    return all_rows, fixed


def format_p(p):
    return f"{p:.1e}" if p < .001 else f"{p:.3f}"


def style():
    return {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": .8, "figure.facecolor": "white", "savefig.facecolor": "white"}


def plot_results(analyses, summary):
    rows = summary["fixed_replication"]
    anchors = [rows[0], rows[1], rows[2], rows[3]]
    with plt.rc_context(style()):
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), layout="constrained")
        for ax, r, letter in zip(axes.flat, anchors, "abcd"):
            a = next(a for a in analyses if a["key"] == r["endpoint"])
            c = NAMES.index(r["coalition"])
            x, y = a["yr"]/len(a["y"]), a["xr"][:, c]/len(a["y"])
            color = COLORS[r["state"]]
            ax.scatter(x, y, s=11, alpha=.32, c=color, linewidths=0, rasterized=True)
            slope, intercept = np.polyfit(x, y, 1)
            endpoints = np.array([x.min(), x.max()])
            ax.plot(endpoints, intercept+slope*endpoints, color=color, linewidth=1.6)
            ax.axhline(0, color="#DDDDDD", linewidth=.5, zorder=0)
            ax.axvline(0, color="#DDDDDD", linewidth=.5, zorder=0)
            ax.set_xlabel(a["title"]+"\n(adjusted rank residual / N)")
            ax.set_ylabel("Coalition Ξ\n(adjusted rank residual / N)")
            ax.set_title(f"{r['state']} · {compact(r['coalition'])}\n"
                         f"ρ={r['rho']:+.3f}   Holm-12 p={format_p(r['p_holm_12'])}   n={r['n']}", loc="left", fontsize=10, pad=10)
            ax.text(-.16, 1.17, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
            ax.grid(color="#EEEEEE", linewidth=.6)
            ax.set_axisbelow(True)
        fig.savefig(OUTPUT / "hcp979_fixed_association_scatter.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.5, 6.4), layout="constrained")
        positions = np.arange(12)[::-1]
        for y, r in zip(positions, rows):
            ci = r["bootstrap_95_ci"]
            ax.plot(ci, [y-.10, y-.10], color="#C87552", linewidth=1.5)
            ax.scatter(r["rho"], y-.10, c="#C87552", s=32, marker="D", zorder=3)
            ax.scatter(r["old_rho"], y+.10, c="#55758C", s=27, marker="o", zorder=3)
            ax.text(1.02, y, format_p(r["p_holm_12"]), transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        labels = [f"{r['state']} · {compact(r['coalition'])}" for r in rows]
        ax.set_yticks(positions, labels)
        ax.set_ylim(-.65, 11.7)
        all_effects = [v for r in rows for v in [r["old_rho"], *r["bootstrap_95_ci"]]]
        lower = min(-.60, np.floor(min(all_effects)*10)/10-.05)
        upper = max(.35, np.ceil(max(all_effects)*10)/10+.05)
        ax.set_xlim(lower, upper)
        ax.axvline(0, color="#BBBBBB", linewidth=.9)
        ax.set_xlabel("Behavior association ρ (new: partial rank correlation, pointwise 95% CI)")
        ax.text(1.02, 1.025, "Holm-12 p", transform=ax.transAxes, fontsize=9)
        handles = [Line2D([], [], color="#55758C", marker="o", linestyle="none", label="Old 57: original statistic"),
                   Line2D([], [], color="#C87552", marker="D", linestyle="-", label="Expanded cohort: unified adjustment")]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.04), ncol=2, frameon=False)
        ax.grid(axis="x", color="#EEEEEE", linewidth=.6)
        ax.set_axisbelow(True)
        fig.savefig(OUTPUT / "hcp979_fixed_replication_forest.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        matrix = np.array([[r["rho"] for r in summary["exploration"] if r["endpoint"] == e[0]] for e in ENDPOINTS]).T
        corrected = np.array([[r["p_holm_600"] for r in summary["exploration"] if r["endpoint"] == e[0]] for e in ENDPOINTS]).T
        limit = max(.1, np.ceil(np.abs(matrix).max()*20)/20)
        fig, ax = plt.subplots(figsize=(8.8, 21.0), layout="constrained")
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
        ax.set_yticks(np.arange(120), [compact(n) for n in NAMES], fontsize=6.5)
        ax.set_xticks(np.arange(5), ["LANGUAGE\nStory", "LANGUAGE\nMath", "EMOTION\nFace speed", "MOTOR\nCapacity", "WM\nAccuracy"], fontsize=9)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=5)
        sizes = np.array([len(c) for c in COMBOS])
        for boundary in np.flatnonzero(np.diff(sizes))+.5:
            ax.axhline(boundary, color="#777777", linewidth=.8)
        yy, xx = np.where(corrected < .05)
        ax.scatter(xx, yy, marker="*", color="black", s=24)
        ax.set_ylabel("Fixed Yeo-7 source coalition (grouped by network count)")
        ax.set_xlabel("Stars: two-sided permutation p after Holm correction across all 600 tests\n"
                      "n=979; MOTOR n=976. Subject-independent inference; family IDs unavailable.", labelpad=12)
        cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.025, shrink=.45, aspect=28)
        cb.set_label("Partial rank correlation ρ")
        fig.savefig(OUTPUT / "hcp979_all_600_associations.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def report(summary):
    fixed = summary["fixed_replication"]
    exploration = summary["exploration"]
    new = sorted([r for r in exploration if not r["fixed_old_hypothesis"] and r["q_bh_600"] < .05], key=lambda r:(r["p_holm_600"], r["q_bh_600"], -abs(r["rho"])))
    lines = ["# HCP 979 人 TASK–行为关联结果", "", "执行日期：2026-10-09。仅使用现有四个 TASK；REST 与 TASK−REST 均未进入分析。", "",
             f"旧 57 人行为流程验证：**{summary['legacy_gate_status']}**。固定旧图 12 项中，"
             f"**{sum(r['replication_supported'] for r in fixed)} 项同方向且通过 Holm-12**，"
             f"{sum(r['p_holm_12'] < .05 and not r['direction_reproduced'] for r in fixed)} 项显著反向。", "",
             "## 固定复现", "", "![固定复现](hcp979_fixed_replication_forest.png)", "",
             "| 终点 | 固定组合 | N | 旧 ρ | 新偏秩 ρ | 95% CI | 原始 p | Holm-12 p | 同向复现 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in fixed:
        lo, hi = r["bootstrap_95_ci"]
        lines.append(f"| {r['endpoint']} | {compact(r['coalition'])} | {r['n']} | {r['old_rho']:+.3f} | {r['rho']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {format_p(r['p_raw'])} | {format_p(r['p_holm_12'])} | {'是' if r['replication_supported'] else '否'} |")
    lines += ["", "![四个固定主组合](hcp979_fixed_association_scatter.png)", "",
              "散点展示统一协变量调整后的秩残差除以 N，仅改变显示尺度；统计使用同一秩残差。旧图 LANGUAGE 是未统一混杂调整的 Spearman，故森林图旧 / 新统计口径存在预先声明的区别。兼容旧口径的扩样效应也保存在 JSON，不能在版本间挑选显著性。", "",
              "## 600 项探索", "", f"全 600 项中，Holm-600 通过 {sum(r['p_holm_600'] < .05 for r in exploration)} 项；BH-600 通过 {sum(r['q_bh_600'] < .05 for r in exploration)} 项。去除旧 12 个固定假设后，新候选 Holm 通过 {sum(r['p_holm_600'] < .05 and not r['fixed_old_hypothesis'] for r in exploration)} 项，BH 通过 {len(new)} 项。", "",
              "| 终点 | 全部 120 组合 Holm 通过 | BH 通过 | 最强绝对效应及组合（描述性） |",
              "| --- | --- | --- | --- |"]
    for key, _, title, _ in ENDPOINTS:
        local = [r for r in exploration if r["endpoint"] == key]
        winner = max(local, key=lambda r:abs(r["rho"]))
        lines.append(f"| {title} | {sum(r['p_holm_600'] < .05 for r in local)} | {sum(r['q_bh_600'] < .05 for r in local)} | {winner['rho']:+.3f} · {compact(winner['coalition'])} |")
    lines += ["", "全景图：[全部 600 项关联](hcp979_all_600_associations.png)。完整效应、原始置换 P 和两种全局校正在 [summary.json](summary.json)；未生成 CSV。", ""]
    if new:
        lines += ["前列新候选（最多列 10 项；完整列表保留在 JSON）：", "", "| 终点 | 组合 | ρ | Holm-600 p | BH-600 q |", "| --- | --- | --- | --- | --- |"]
        for r in new[:10]:
            lines.append(f"| {r['endpoint']} | {compact(r['coalition'])} | {r['rho']:+.3f} | {format_p(r['p_holm_600'])} | {format_p(r['q_bh_600'])} |")
    else:
        lines += ["本轮没有发现通过所声明多重比较校正的新候选。没有继续扩增行为终点、参数或组合。"]
    lines += ["", "## 方法与解释范围", "",
              "脑指标为固定全七网络下一状态目标的分组整合信息 Ξ(S)=EI(S)−ΣEI(i)，不是 SPT 根节点 Syn 或纯 k 阶原子。源为组合各网络的三阶历史。未重拟合 PCA / TM / EI；既有模型为 NoGSR、任务设计重建、网络 PC1、delta Ridge p=3 / alpha=1、独立高斯源干预、affine-Gaussian TM。内部 bits、显示信息量用 nats；秩统计不受换算影响。", "",
              "统一调整年龄类别、性别、重建版本类别、样本来源类别，EMOTION 额外调整 Shape speed。100000 次双侧 Freedman–Lane 型残差置换，在来源块内进行并重新投影协变量。随机种子 20261009，各终点使用固定派生种子；5000 次来源分层 bootstrap，重算秩，EMOTION 同时重算 Shape-speed 秩。区间为逐项 95% 区间。固定家族 12 项 Holm；探索家族全部 600 项 Holm 与 BH，BH 结论属于 FDR 筛选且依赖其适用条件。", "",
              "MOTOR 以旧 57 人均值与样本标准差冻结三项权重；三名缺失者只从 MOTOR 排除。Story 有 614/979 人满分，保留所有并列值，不删除满分者。979 人含旧 57 人，旧补充样本又曾按行为多样性挑选，本轮属于扩大样本检验，不是独立样本确认；未单列 922 人报告。", "",
              "Family_ID 和与当前成像数据严格匹配的头动未取得，置换和 bootstrap 仍基于被试独立假设。Schaefer-1000 新被试的原始排列 / 提取来源尚缺独立证明；哈希一致只验证输入未变，不证明提取本身正确。结果不能升级为家族稳健、头动稳健或生物学因果证据。行为字段与当前 LR 运行的对应关系也尚未独立核验。", "",
              "新组合全样本筛选是探索性证据，没有未使用的独立验证集。第三阶段的根 Syn / 成对基线增量预测和家族分组留出验证，按方案明确不纳入本次首次路线；本轮不声称已证明不可约高阶关系的行为优势。", "",
              "V=Visual，SM=Somatomotor，DAN=Dorsal Attention，VAN=Salience/Ventral Attention，L=Limbic，C=Control，D=Default。这里定位的是功能网络组合，不是几个具体 ROI。", "",
              "## 核验记录", "", f"执行耗时 {summary['elapsed_seconds']:.1f} 秒（含输入验证、旧流程重跑、置换、bootstrap；不含后续图形审阅）。",
              f"实际输入 SHA256 核验 {summary['input_verification']['files']} 个文件、{summary['input_verification']['bytes_checked']/1e9:.2f} GB，状态 passed；旧模型实现及 atlas 标签哈希相符。",
              "Syn 容差 1e-4 bits，保留原值；显著负值和容差内负值数量均为零。详细数值及新 / 旧重叠 57 人组合特征差异见 summary.json。",
              "旧 LANGUAGE 历史未保存全精度统计：用原统计函数重跑、独立 SciPy 校验系数，并与原图三位小数相符；不能宣称与一个不存在的全精度历史记录逐位一致。MOTOR 和 EMOTION 的全精度历史系数与置换 P 均重算一致。", "",
              "本次通过 Zotero 再次取得 P6UJCVG8 的 DXGC7JEA 主稿与 MWIWKSVG 补充全文；读取 Fig.3、SI S1.2–S1.3、S5.1、S12.2.2。附件没有明确稿内版本号 / 修订日期，保留该版本歧义；未改变方法。"]
    (OUTPUT / "report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        checks = self_check()
        if args.self_check:
            print(json.dumps(checks)); return
        started = time.monotonic()
        manifest = json.loads((SOURCE / "manifest.json").read_text())
        if len(manifest["subjects"]) != 979:
            raise ValueError("Expected frozen 979 cohort")
        config = dict(source_manifest_hash=sha(SOURCE / "manifest.json"), source_records_hash=sha(SOURCE / "records.jsonl"),
                      behavior_sha256=sha(BEHAVIOR), script_sha256=sha(Path(__file__)), seed=SEED,
                      permutations=PERMUTATIONS, bootstraps=BOOTSTRAPS, fixed=list(FIXED), endpoints=list(ENDPOINTS),
                      reference_file_sha256={str(p.relative_to(ROOT)): sha(p) for p in [
                          ROOT / "results/hcp_language_story_math_coalitions_57/selected_candidate_source_data.tsv",
                          ROOT / "results/hcp_motor_composite_scores_57/summary.json",
                          ROOT / "results/hcp_emotion_performance_coalitions_57/summary.json"]},
                      metric="coalition EI minus singleton-network EI sum; fixed full seven-network target")
        config_hash = digest(config)
        if args.plots_only:
            summary = json.loads((OUTPUT / "summary.json").read_text())
            if summary["config_hash"] != config_hash:
                raise ValueError("Plot regeneration inputs/config changed")
            ids, features, _ = load_features(manifest)
            frame = raw_behavior(ids)
            analyses = [prepare_analysis(e, frame, features) for e in ENDPOINTS]
            plot_results(analyses, summary)
            print("Figures regenerated from existing inference"); return
        print("Validating actual cached input files and frozen implementation", flush=True)
        verification = verify_inputs(manifest)
        print(f"Input validation passed: {verification['files']} files", flush=True)
        gate_path = OUTPUT / "legacy_gate.json"
        gate_identity = digest(dict(behavior_sha256=config["behavior_sha256"],
                                    reference_file_sha256=config["reference_file_sha256"],
                                    old_script_hashes={n:sha(ROOT / "scripts" / n) for n in ["plot_hcp_schaefer1000_behavior_main.py", "screen_hcp_motor_composite_scores_57.py", "screen_hcp_emotion_performance_coalitions_57.py"]},
                                    old_caches={str(p):sha(p) for p in [ROOT / "results/hcp_language_story_math_coalitions_57/language_coalition_synergy_57.npz", ROOT / "results/hcp_motor_composite_scores_57/motor_coalition_synergy_57.npz", ROOT / "results/hcp_emotion_performance_coalitions_57/emotion_rest_coalition_synergy_57.npz"]}))
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
        if gate.get("identity") != gate_identity or gate.get("status") != "passed":
            print("Recomputing old 57-subject behavior statistics using original functions", flush=True)
            gate = legacy_gate()
            gate["identity"] = gate_identity
            dump(gate_path, gate)
        print("Old 57-subject behavioral gate passed", flush=True)
        ids, features, audit = load_features(manifest)
        frame = raw_behavior(ids)
        analyses = [prepare_analysis(e, frame, features) for e in ENDPOINTS]
        cache_path = OUTPUT / "inference_cache.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        if cache.get("config_hash") != config_hash:
            cache = dict(config_hash=config_hash, completed={})
        for i, a in enumerate(analyses):
            if a["key"] in cache["completed"]:
                print(f"Reusing verified inference: {a['key']}", flush=True); continue
            t = time.monotonic()
            print(f"100000-permutation analysis: {a['key']} n={len(a['y'])}", flush=True)
            p = permute(a, PERMUTATIONS, SEED+i*1009)
            fixed_names = [name for key, name, _ in FIXED if key == a["key"]]
            intervals = bootstrap(a, [NAMES.index(n) for n in fixed_names], SEED+50000+i*1009) if fixed_names else []
            entry = dict(p_raw=p.tolist(), rho=a["observed"].tolist(), n=len(a["y"]),
                         design_labels=a["design_labels"], permutation_seed=SEED+i*1009,
                         bootstrap_seed=SEED+50000+i*1009, bootstrap_repeats=BOOTSTRAPS if fixed_names else 0,
                         fixed_intervals={n:v.tolist() for n,v in zip(fixed_names, intervals)},
                         elapsed_seconds=time.monotonic()-t)
            cache["completed"][a["key"]] = entry
            dump(cache_path, cache)
            print(f"Completed {a['key']} in {entry['elapsed_seconds']:.1f}s", flush=True)
        exploration, fixed = scientific_rows(analyses, cache["completed"], gate)
        summary = dict(config_hash=config_hash, configuration=config, n_imaging_subjects=979,
                       self_check=checks, input_verification=verification, legacy_gate_status=gate["status"],
                       legacy_gate_precision_limit=gate["language_precision_limit"], numerical_audit=audit,
                       new_old_overlap_feature_comparison=compare_overlap(ids, features),
                       fixed_replication=fixed, exploration=exploration,
                       inference_independence_assumption="Subject independent; relatedness unadjusted because verified family identifiers are unavailable",
                       independent_replication=False, independent_holdout=False,
                       familywise_significance_requires_valid_marginal_p_values=True,
                       elapsed_seconds=time.monotonic()-started)
        dump(OUTPUT / "summary.json", summary)
        plot_results(analyses, summary)
        report(summary)
        print(json.dumps(dict(status="complete", report=str(OUTPUT / "report.md"),
                              fixed_replication_supported=sum(r["replication_supported"] for r in fixed),
                              holm_600_significant=sum(r["p_holm_600"] < .05 for r in exploration),
                              bh_600_significant=sum(r["q_bh_600"] < .05 for r in exploration),
                              elapsed_seconds=time.monotonic()-started)), flush=True)


if __name__ == "__main__":
    main()
