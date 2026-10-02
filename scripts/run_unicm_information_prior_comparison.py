#!/usr/bin/env python3
"""Compare information priors on the existing frozen UniCM calibrator.

All priors use the same independent-source affine degree-1 TM channel. Pair
attribution is Shapley for a sum-of-internal-edges game. Causal density uses
cross-source conditional information and outgoing attribution. Removed methods
are retained only in the legacy attribution cache, not in current comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_unicm_11mode_shapley import (
    fit_affine_readout,
    mode_feature_blocks,
    stable_logdet,
    standardize,
)
from scripts.run_unicm_synergy_guided_forecast import (
    cell_rmse,
    chronological_split,
    circular_block_indices,
    history_summaries,
    mean_cell_acc,
    mean_cell_nrmse,
    target_scaling,
)
from scripts.run_unicm_synergy_regularized_calibration import predict_generalized_ridge, prepare_designs, tune_prior
from scripts.run_unicm_target_xi_shapley_prior import (
    DEFAULT_CACHE,
    DEFAULT_CALIBRATION,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT as XI_REFERENCE,
    prediction_cache_path,
)
from scripts.unicm_peid_syn_analysis import MODE_NAMES, sample_full_history_mode_inputs

DEFAULT_OUTPUT = ROOT / "results/unicm_information_prior_comparison_n16384"
EXTENDED_OUTPUT = ROOT / "results/unicm_information_prior_comparison_extended_n16384"
CURRENT_OUTPUT = ROOT / "results/unicm_information_prior_comparison_literature_shared_xi_n16384"
LEGACY_CACHED_METHODS = ("xi_shapley", "mmi_pid", "phi_r", "ei_shapley", "conditional_mi", "xi_target_averaged")
ALL_CACHED_METHODS = (*LEGACY_CACHED_METHODS, "o_shapley", "phi_wms", "mim", "phi_si", "causal_density")
METHODS = ("xi_shapley", "phi_r", "ei_shapley", "phi_wms", "phi_si", "causal_density")
EXCLUDED_METHODS = ("mmi_pid", "xi_target_averaged", "o_shapley", "conditional_mi", "mim")
EXCLUSION_REASON = (
    "User-specified scope: MIM uses the shared bounded-uniform intervention channel "
    "and equals singleton EI, so it is excluded as an ordinary observational-MI comparator. "
    "Singleton EI attribution still differs from all-coalition EI-Shapley attribution."
)
COMPETITORS = METHODS[1:]
LABELS = {
    "frozen": "Frozen",
    "univariate": "Univariate",
    "uniform": "Uniform ridge",
    "xi_shapley": r"$\Xi$-Shapley (all orders)",
    "mmi_pid": "MMI-PID synergy",
    "phi_r": r"$\Phi^R$ pair prior",
    "ei_shapley": "EI-Shapley",
    "conditional_mi": "Conditional MI + self MI",
    "xi_target_averaged": r"$\Xi$-Shapley (target-averaged)",
    "o_shapley": "O-Shapley (all orders)",
    "phi_wms": r"$\Phi^{\mathrm{WMS}}$ pair prior",
    "mim": "Single-source MI (MIM)",
    "phi_si": r"$\Phi_{\mathrm{SI}}$ pair prior (forward)",
    "causal_density": "Causal density (outgoing)",
    "surd": "SURD synergy (all orders)",
}
COLORS = {
    "frozen": "#8A929D", "univariate": "#9BACCA", "uniform": "#58687D",
    "xi_shapley": "#007D79", "mmi_pid": "#D55E00", "phi_r": "#7B5EA7",
    "ei_shapley": "#0072B2", "conditional_mi": "#B67800",
    "xi_target_averaged": "#738E83",
    "o_shapley": "#CC79A7", "phi_wms": "#D55E00",
    "mim": "#B67800", "phi_si": "#CC79A7", "causal_density": "#405A45",
    "surd": "#B67800",
}
TOLERANCE_BITS = 1e-8
CLOSURE_BITS = 1e-10
ESTIMATOR_VERSION = "independent-affine-tm-information-priors-v3-mim-si-cd"


def nonnegative(values: np.ndarray, name: str, audit: dict) -> np.ndarray:
    """Record numerical zeros and fail on significant negative information."""
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: non-finite values.")
    minimum = float(array.min())
    bad = array < -TOLERANCE_BITS
    numerical = (array < 0) & ~bad
    entry = audit.setdefault(name, {"minimum_bit": minimum, "numerical_zero_count": 0,
                                    "significant_negative_count": 0, "tolerance_bit": TOLERANCE_BITS})
    entry["minimum_bit"] = min(entry["minimum_bit"], minimum)
    entry["numerical_zero_count"] += int(numerical.sum())
    entry["significant_negative_count"] += int(bad.sum())
    if bad.any():
        raise ValueError(f"{name}: minimum={minimum:.12g} bit, threshold=-{TOLERANCE_BITS:g} bit, "
                         f"affected_count={int(bad.sum())}.")
    result = array.copy()
    result[numerical] = 0.0
    return result


def coalition_operators(n: int) -> tuple[np.ndarray, np.ndarray]:
    masks = np.arange(1 << n)
    membership = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
    shapley = np.zeros((n, 1 << n))
    for player in range(n):
        for mask in masks:
            if int(mask) & (1 << player):
                continue
            size = int(mask).bit_count()
            weight = 1.0 / (n * math.comb(n - 1, size))
            shapley[player, mask] -= weight
            shapley[player, int(mask) | (1 << player)] += weight
    return membership, shapley


def target_oinfo_game(ei: np.ndarray, n: int, audit: dict) -> np.ndarray:
    """Negative target O-information increment, on the same coalition MI table.

    Independent source blocks make this conditional DTC. Every marginal is
    nonnegative, so exact Shapley can use the existing nonnegative prior map.
    This is not the history-conditioned dynamic O-information definition.
    """
    ei = np.asarray(ei, dtype=float)
    if ei.ndim != 2 or ei.shape[0] != 1 << n or not np.isfinite(ei).all():
        raise ValueError("Expected a finite coalition-by-target MI table.")
    if np.max(np.abs(ei[0])) > CLOSURE_BITS:
        raise ValueError("Empty-coalition MI must be numerical zero.")
    masks = np.arange(1 << n)
    sizes = np.asarray([int(mask).bit_count() for mask in masks])
    game = (sizes[:, None] - 1) * ei
    for player in range(n):
        included = (masks & (1 << player)) != 0
        game[included] -= ei[masks[included] ^ (1 << player)]
    game[0] = 0.0  # Declared empty-game convention, not a projection.
    game = nonnegative(game, "subset_o_information", audit)
    for player in range(n):
        absent = masks[(masks & (1 << player)) == 0]
        nonnegative(game[absent | (1 << player)] - game[absent], "o_marginal_increment", audit)
    return game


def channel_priors(coefficients: np.ndarray, residual: np.ndarray, blocks: tuple,
                   audit: dict | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return [target, source] priors and native pair scores for one lead."""
    if audit is None:
        audit = {}
    coefficients, residual = np.asarray(coefficients), np.asarray(residual)
    n = len(blocks)
    if coefficients.ndim != 2 or coefficients.shape[1] != n or residual.shape != (n, n):
        raise ValueError("Channel must have one output per mode and a square residual covariance.")
    covered = sorted(column for block in blocks for column in block)
    if covered != list(range(coefficients.shape[0])):
        raise ValueError("Mode blocks must cover the source columns exactly once.")
    if not np.isfinite(coefficients).all() or not np.isfinite(residual).all():
        raise ValueError("Non-finite affine channel.")
    stable_logdet(residual, "channel residual covariance")
    gram = np.stack([coefficients[np.asarray(block)].T @ coefficients[np.asarray(block)] for block in blocks])
    full = residual + gram.sum(axis=0)
    membership, shapley = coalition_operators(n)
    diagonal = np.diagonal(gram, axis1=1, axis2=2)
    conditional = np.diag(residual)[None, :] + (1.0 - membership) @ diagonal
    # Identical scalar-target logdet formula to coalition_ei_table, vectorized.
    ei = 0.5 * (np.log(np.diag(full))[None, :] - np.log(conditional)) / np.log(2.0)
    ei = nonnegative(ei, "coalition_ei", audit)
    singleton = ei[1 << np.arange(n)]  # [source, target]
    xi_game = ei - membership @ singleton
    xi_game = nonnegative(xi_game, "subset_xi", audit)
    xi = nonnegative((shapley @ xi_game).T, "xi_shapley", audit)
    ei_attribution = nonnegative((shapley @ ei).T, "ei_shapley", audit)
    o_game = target_oinfo_game(ei, n, audit)
    o_attribution = nonnegative((shapley @ o_game).T, "o_shapley", audit)
    closure = float(np.max(np.abs(xi.sum(axis=1) - xi_game[-1])))
    ei_closure = float(np.max(np.abs(ei_attribution.sum(axis=1) - ei[-1])))
    o_closure = float(np.max(np.abs(o_attribution.sum(axis=1) - o_game[-1])))
    linearity = float(np.max(np.abs(ei_attribution - xi - singleton.T)))
    audit["maximum_shapley_closure_error_bit"] = max(audit.get("maximum_shapley_closure_error_bit", 0), closure, ei_closure, o_closure)
    audit["maximum_ei_xi_linearity_error_bit"] = max(audit.get("maximum_ei_xi_linearity_error_bit", 0), linearity)
    if max(closure, ei_closure, o_closure, linearity) > CLOSURE_BITS:
        raise ValueError(f"Shapley closure/linearity exceeds {CLOSURE_BITS:g} bit: "
                         f"Xi={closure:g}, EI={ei_closure:g}, O={o_closure:g}, linearity={linearity:g}.")

    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    mmi_edges = np.empty((len(pairs), n))
    phi_edges = np.empty(len(pairs))
    wms_edges = np.empty(len(pairs))
    si_edges = np.empty(len(pairs))
    for index, (a, b) in enumerate(pairs):
        mmi_edges[index] = ei[(1 << a) | (1 << b)] - np.maximum(singleton[a], singleton[b])
        output = np.asarray([a, b])
        marginal_full = full[np.ix_(output, output)]
        # Marginalize all omitted modes; do not refit a different pair channel.
        other = [m for m in range(n) if m not in (a, b)]
        marginal_conditional = residual[np.ix_(output, output)].copy()
        if other:
            marginal_conditional += gram[other].sum(axis=0)[np.ix_(output, output)]
        joint_mi = 0.5 * (stable_logdet(marginal_full, "pair full covariance")
                          - stable_logdet(marginal_conditional, "pair conditional covariance")) / np.log(2.0)
        wms_edges[index] = joint_mi - singleton[a, a] - singleton[b, b]
        phi_edges[index] = wms_edges[index] + float(singleton[np.ix_([a, b], [a, b])].min())
        # Forward stochastic interaction, Kitazono et al. (2018), Eq. A8.
        # All omitted inputs are marginalized, as for the WMS pair channel.
        si_edges[index] = 0.5 * (np.log(conditional[1 << a, a])
                                + np.log(conditional[1 << b, b])
                                - stable_logdet(marginal_conditional, "SI pair conditional covariance")) / np.log(2.0)
        output_tc = 0.5 * (np.log(full[a, a]) + np.log(full[b, b])
                          - stable_logdet(marginal_full, "SI pair full covariance")) / np.log(2.0)
        identity_error = abs(si_edges[index] - wms_edges[index] - output_tc)
        audit["maximum_forward_si_wms_tc_error_bit"] = max(
            audit.get("maximum_forward_si_wms_tc_error_bit", 0), float(identity_error))
        if identity_error > CLOSURE_BITS:
            raise ValueError("Forward stochastic interaction = WMS + output TC failed.")
    mmi_edges = nonnegative(mmi_edges, "mmi_pid_edges", audit)
    phi_edges = nonnegative(phi_edges, "phi_r_edges", audit)
    wms_edges = nonnegative(wms_edges, "phi_wms_edges", audit)
    si_edges = nonnegative(si_edges, "phi_si_edges", audit)
    mmi = np.zeros((n, n))
    phi = np.zeros(n)
    wms = np.zeros(n)
    si = np.zeros(n)
    for index, (a, b) in enumerate(pairs):
        mmi[:, a] += 0.5 * mmi_edges[index]
        mmi[:, b] += 0.5 * mmi_edges[index]
        phi[[a, b]] += 0.5 * phi_edges[index]
        wms[[a, b]] += 0.5 * wms_edges[index]
        si[[a, b]] += 0.5 * si_edges[index]
    edge_closure = max(float(np.max(np.abs(mmi.sum(axis=1) - mmi_edges.sum(axis=0)))),
                       abs(float(phi.sum() - phi_edges.sum())), abs(float(wms.sum() - wms_edges.sum())),
                       abs(float(si.sum() - si_edges.sum())))
    audit["maximum_half_edge_closure_error_bit"] = max(audit.get("maximum_half_edge_closure_error_bit", 0), edge_closure)
    if edge_closure > CLOSURE_BITS:
        raise ValueError("Pair-attribution closure failed.")
    wms = nonnegative(wms, "phi_wms", audit)
    # Conditional TE conditions on every other source, including target history.
    # Its diagonal is excluded by the published causal-density definition.
    transfer = nonnegative(0.5 * np.log1p(diagonal / np.diag(residual)[None, :]) / np.log(2.0),
                           "all_conditioned_transfer", audit)  # [source, target]
    np.fill_diagonal(transfer, 0.0)
    outgoing_cd = transfer.sum(axis=1) / (n * (n - 1)) if n > 1 else np.zeros(n)
    cmi = np.empty((n, n))
    for target in range(n):
        for source in range(n):
            cmi[target, source] = (singleton[source, target] if source == target else
                                  ei[(1 << source) | (1 << target), target] - singleton[target, target])
    priors = {"xi_shapley": xi, "mmi_pid": mmi,
              "phi_r": np.broadcast_to(phi, (n, n)).copy(),
              "ei_shapley": ei_attribution,
              "conditional_mi": nonnegative(cmi, "conditional_mi", audit),
              "xi_target_averaged": np.broadcast_to(xi.mean(axis=0), (n, n)).copy(),
              "o_shapley": o_attribution, "phi_wms": np.broadcast_to(wms, (n, n)).copy(),
              "mim": singleton.T.copy(), "phi_si": np.broadcast_to(si, (n, n)).copy(),
              "causal_density": np.broadcast_to(outgoing_cd, (n, n)).copy()}
    return priors, {"mmi_pid_edges": mmi_edges, "phi_r_edges": phi_edges,
                    "phi_wms_edges": wms_edges, "singleton_ei": singleton,
                    "phi_si_edges": si_edges, "conditional_te_edges": transfer}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_centralities(args: argparse.Namespace) -> tuple[dict, dict]:
    paths = [prediction_cache_path(args.cache_dir, seed, args) for seed in args.checkpoint_seeds]
    config = {name: getattr(args, name) for name in ("n_samples", "sampling_seed", "intervention_bound", "start_month", "checkpoint_seeds", "covariance_ridge")}
    config.update(estimator_version=ESTIMATOR_VERSION, cache_sha256={str(p): sha256(p) for p in paths}, mode_names=list(MODE_NAMES))
    cache_path = args.centrality_cache
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            if json.loads(str(archive["config"])) != config:
                raise ValueError("Information-prior cache configuration/provenance mismatch.")
            priors = {key: archive[key].copy() for key in ALL_CACHED_METHODS}
            audit = json.loads(str(archive["audit"]))
        print("Reused information attribution cache.", flush=True)
        return priors, audit
    source = sample_full_history_mode_inputs(n_samples=args.n_samples, intervention_bound=args.intervention_bound,
                                            seed=args.sampling_seed).astype(np.float64)
    x = standardize(source.reshape(args.n_samples, -1), "intervention source")
    blocks = mode_feature_blocks()
    n = len(MODE_NAMES)
    priors = {key: np.empty((len(paths), n, 24, n)) for key in ALL_CACHED_METHODS}
    edges = {"mmi_pid_edges": np.empty((len(paths), 24, n * (n - 1) // 2, n)),
             "phi_r_edges": np.empty((len(paths), 24, n * (n - 1) // 2)),
             "phi_wms_edges": np.empty((len(paths), 24, n * (n - 1) // 2)),
             "phi_si_edges": np.empty((len(paths), 24, n * (n - 1) // 2)),
             "conditional_te_edges": np.empty((len(paths), 24, n, n)),
             "singleton_ei": np.empty((len(paths), 24, n, n))}
    audit: dict = {"tolerance_bit": TOLERANCE_BITS, "closure_tolerance_bit": CLOSURE_BITS}
    for seed_index, (seed, path) in enumerate(zip(args.checkpoint_seeds, paths)):
        with np.load(path, allow_pickle=False) as archive:
            expected = {"checkpoint": seed, "n_samples": args.n_samples, "sampling_seed": args.sampling_seed,
                        "intervention_bound": args.intervention_bound, "start_month": args.start_month}
            metadata = json.loads(str(archive["metadata"]))
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Intervention metadata mismatch: {path}.")
            prediction = archive["all_mode_targets"].astype(np.float64)
        if prediction.shape != (args.n_samples, 24, n) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid intervention predictions: {path}.")
        for lead in range(24):
            coefficients, residual = fit_affine_readout(x, prediction[:, lead], args.covariance_ridge)
            values, native = channel_priors(coefficients, residual, blocks, audit)
            for key in ALL_CACHED_METHODS:
                priors[key][seed_index, :, lead] = values[key]
            for key in edges:
                edges[key][seed_index, lead] = native[key]
            if (lead + 1) % 6 == 0:
                print(f"Checkpoint {seed}: information priors {lead + 1}/24 leads.", flush=True)
    with np.load(args.xi_reference / "target_xi_shapley_centrality.npz", allow_pickle=False) as archive:
        if json.loads(str(archive["config"])) != {k: config[k] for k in ("n_samples", "sampling_seed", "intervention_bound", "start_month", "checkpoint_seeds", "covariance_ridge")}:
            raise ValueError("Original Xi reference configuration mismatch.")
        reference = archive["centrality"]
    error = float(np.max(np.abs(priors["xi_shapley"].mean(axis=0) - reference)))
    audit["maximum_original_xi_cache_difference_bit"] = error
    if error > 1e-9:
        raise ValueError(f"Recomputed Xi differs from original cache by {error:g} bit.")
    legacy_path = DEFAULT_OUTPUT / "information_centralities.npz"
    if legacy_path.exists():
        with np.load(legacy_path, allow_pickle=False) as legacy:
            legacy_config = json.loads(str(legacy["config"]))
            comparable = {k: v for k, v in config.items() if k != "estimator_version"}
            if {k: v for k, v in legacy_config.items() if k != "estimator_version"} != comparable:
                raise ValueError("Legacy attribution provenance differs from the expanded run.")
            differences = {key: float(np.max(np.abs(priors[key] - legacy[key])))
                           for key in LEGACY_CACHED_METHODS}
        audit["legacy_attribution_maximum_difference_bit"] = differences
        if max(differences.values()) > 1e-9:
            raise ValueError(f"Legacy attributions changed: {differences}.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **priors, **edges, config=json.dumps(config, sort_keys=True), audit=json.dumps(audit, sort_keys=True))
    return priors, audit


def fit_prior(designs: dict, centrality: np.ndarray, validation_target: np.ndarray,
              target_scale: np.ndarray, *, alphas: list, gammas: list,
              floor_fraction: float, fixed_parameters: dict | None = None) -> tuple:
    """Use either independent validation selection or one declared shared setting."""
    if fixed_parameters is None:
        return tune_prior(designs, centrality, validation_target, target_scale,
                          alphas=alphas, gammas=gammas, floor_fraction=floor_fraction)
    alpha, gamma = fixed_parameters["alpha"], fixed_parameters["gamma"]
    validation, test = predict_generalized_ridge(designs, centrality, alpha=alpha,
                                                gamma=gamma, floor_fraction=floor_fraction)
    score = mean_cell_nrmse(validation, validation_target, target_scale)
    return alpha, gamma, {f"alpha={alpha:g},gamma={gamma:g}": score}, validation, test


def paired_bootstrap(predictions: dict[str, np.ndarray], target: np.ndarray, scale: np.ndarray,
                     replicates: int, block: int, seed: int) -> tuple[dict, dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = np.empty((replicates, len(target)))
    for replicate in range(replicates):
        indices = circular_block_indices(rng, len(target), block)
        counts[replicate] = np.bincount(indices, minlength=len(target)) / len(target)
    aggregate, leads = {}, {}
    for key, prediction in predictions.items():
        squared = ((prediction - target) / scale) ** 2
        cells = np.sqrt((counts @ squared.reshape(len(target), -1)).reshape(replicates, *target.shape[1:]))
        aggregate[key], leads[key] = cells.mean(axis=(1, 2)), cells.mean(axis=1)
    return aggregate, leads, counts


def plot_results(result: dict, predictions: dict, target: np.ndarray, scale: np.ndarray, output: Path,
                 *, panel_a_result: dict | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rc_context
    methods = tuple(result.get("included_methods", METHODS))
    order = ("frozen", "univariate", "uniform", *methods)
    fixed = result["parameter_mode"] == "fixed_xi"
    panel_a = result if panel_a_result is None else panel_a_result
    panel_a_fixed = panel_a["parameter_mode"] == "fixed_xi"
    labels = dict(LABELS)
    labels.update(result.get("method_labels", {}))
    if panel_a_fixed:
        labels["uniform"] = r"Uniform ridge ($\alpha=30000$)"
    else:
        for key in methods:
            if panel_a["methods"][key]["gamma"] == 0:
                labels[key] += r" ($\gamma=0$)"
    with rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "savefig.facecolor": "white"}):
        fig = plt.figure(figsize=(12.2, 8.0), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.05), width_ratios=(1.13, 1))
        ax_a, ax_b, ax_c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
        for row, key in enumerate(order):
            value = panel_a["methods"][key]["test_nrmse"]
            ax_a.scatter(value, row, s=28, color=COLORS[key], zorder=3)
            ax_a.annotate(f"{value:.6f}", (value, row), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7)
        ax_a.set_yticks(range(len(order)), [labels[key] for key in order])
        ax_a.invert_yaxis()
        scores = [panel_a["methods"][key]["test_nrmse"] for key in order]
        ax_a.set_xlim(min(scores) - 0.02, max(scores) + 0.055)
        ax_a.set_xlabel("Test normalized RMSE (lower is better)")
        ax_a.grid(axis="x", color="#E5E9EE", lw=0.5)
        compared = methods[1:]
        for row, key in enumerate(compared):
            delta = result["methods"][key]["gain_over_xi"]
            low, high = delta["ci95"]
            ax_b.plot([low, high], [row, row], color=COLORS[key], lw=1.5)
            ax_b.scatter(delta["point"], row, color=COLORS[key], s=28, zorder=3)
        ax_b.axvline(0, color="#707780", lw=0.8, ls="--")
        ax_b.set_yticks(range(len(compared)), [labels[key] for key in compared])
        ax_b.invert_yaxis()
        ax_b.set_ylim(len(compared) - 0.5, -0.5)
        ax_b.set_xlabel(r"nRMSE gain over $\Xi$-Shapley" + "\n(positive favors the comparator)")
        ax_b.grid(axis="x", color="#E5E9EE", lw=0.5)
        leads = np.arange(1, 25)
        baseline = cell_rmse(predictions["uniform"], target, scale)
        for key in methods:
            gain = (baseline - cell_rmse(predictions[key], target, scale)).mean(axis=0)
            ax_c.plot(leads, gain, color=COLORS[key], lw=1.7 if key == "xi_shapley" else 1.15,
                      ls="--" if key in ("o_shapley", "phi_wms") else "-", marker="o" if key == "xi_shapley" else None,
                      markersize=2.5, label=labels[key])
        ax_c.axhline(0, color="#707780", lw=0.7, ls="--")
        ax_c.set(xlabel="Prediction lead (months)",
                 ylabel=(r"nRMSE gain over uniform ridge ($\alpha=30000$)" if fixed else
                         "nRMSE gain over uniform ridge"),
                 xticks=(1, 4, 8, 12, 16, 20, 24), xlim=(0.5, 24.5))
        ax_c.grid(axis="y", color="#E5E9EE", lw=0.5)
        ax_c.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7.5)
        for letter, axis in zip("abc", (ax_a, ax_b, ax_c)):
            axis.set_title(letter, loc="left", fontweight="bold", fontsize=11, pad=10)
        if panel_a_result is not None:
            ax_a.set_title("a   Independent validation selection", loc="left", fontsize=9, pad=10)
            ax_b.set_title("b   Shared parameters", loc="left", fontsize=9, pad=10)
            ax_c.set_title("c   Shared parameters", loc="left", fontsize=9, pad=10)
            fig.suptitle(r"Panel a: independently tuned; panels b,c: $\alpha=30000$, $\gamma=3$", fontsize=10)
        else:
            fig.suptitle(r"Shared $\Xi$ settings: $\alpha=30000$, $\gamma=3$" if fixed else
                         "Independent validation selection", fontsize=10)
        fig.savefig(output, dpi=400, bbox_inches="tight")
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((args.calibration_dir / "summary.json").read_text())
    with np.load(args.input_dir / "model_inputs.npz", allow_pickle=False) as archive:
        history, target = archive["history"].astype(float), archive["targets"].astype(float)
        names = archive["mode_names"].astype(str).tolist()
        input_metadata = json.loads(str(archive["metadata"]))
    with np.load(args.input_dir / "modeformer_predictions.npz", allow_pickle=False) as archive:
        ensemble = archive["ensemble_prediction"].astype(float)
        target_dates = archive["target_dates"].astype(str)
        if archive["mode_names"].astype(str).tolist() != names:
            raise ValueError("Observational prediction mode order mismatch.")
    if names != ["ENSO" if m == "nino" else m for m in MODE_NAMES]:
        raise ValueError("Observational and intervention mode order mismatch.")
    if any(not np.isfinite(v).all() for v in (history, target, ensemble)):
        raise ValueError("Non-finite observational inputs.")
    if input_metadata.get("normalization_fit_period") != "1980-01/2003-12":
        raise ValueError("Expected fit-only upstream normalization.")
    split = chronological_split(target_dates, **calibration["chronological_split"])
    if [len(split.fit), len(split.validation), len(split.test)] != [253, 36, 96]:
        raise ValueError("Chronological split differs from the declared experiment.")
    _, scale = target_scaling(target, split.fit)
    additive, _ = history_summaries(history)
    designs = prepare_designs(ensemble, target, additive, split)
    centralities, audit = build_centralities(args)
    reference = json.loads((args.xi_reference / "summary.json").read_text())
    fixed_parameters = reference["selected_hyperparameters"] if args.fixed_xi_parameters else None
    if fixed_parameters is not None and args.floor_fraction != 0.05:
        raise ValueError("The fixed-Xi run must preserve the original floor_fraction=0.05.")
    alphas, gammas = [float(v) for v in args.alphas.split(",")], [float(v) for v in args.gammas.split(",")]
    if fixed_parameters is not None:
        if fixed_parameters["gamma"] <= 0:
            raise ValueError("Shared Xi parameters must keep information weighting active (gamma > 0).")
    elif any(gamma <= 0 for gamma in gammas):
        raise ValueError("Information-prior selection requires gamma > 0; uniform is a separate baseline.")
    predictions = {"frozen": ensemble[split.test]}
    with np.load(args.calibration_dir / "evaluation_arrays.npz", allow_pickle=False) as archive:
        if not np.allclose(archive["test_target"], target[split.test], rtol=0, atol=1e-6):
            raise ValueError("Cached test target mismatch.")
        predictions["univariate"] = archive["prediction_univariate"].astype(float)
        reference_uniform = archive["prediction_uniform"].astype(float)
    result = {"status": "completed", "parameter_mode": "fixed_xi" if fixed_parameters else "independently_tuned",
              "included_methods": list(METHODS), "excluded_methods": list(EXCLUDED_METHODS),
              "exclusion_reason": EXCLUSION_REASON,
              "shared_hyperparameters": fixed_parameters,
              "fixed_parameter_reference": str(args.xi_reference / "summary.json") if fixed_parameters else None,
              "centrality_cache": str(args.centrality_cache.resolve()),
              "new_prior_definitions": {
                  "phi_wms": "55 matched dual-output WMS edges, half-edge allocation, broadcast across targets; no system MIP search",
                  "phi_si": "forward stochastic interaction: H(Ya|ha)+H(Yb|hb)-H(Ya,Yb|ha,hb); matched 55 pairs, half-edge allocation; no MIP search",
                  "causal_density": "published CD average of cross-source conditional TE I(h_m;Y_j|h_-m); all other histories conditioned, diagonal excluded; outgoing source contributions broadcast to targets",
              },
              "estimator": ESTIMATOR_VERSION, "intervention_n_samples": args.n_samples,
              "checkpoint_seeds": args.checkpoint_seeds, "mode_names": names,
              "samples": {k: len(getattr(split, k)) for k in ("fit", "validation", "test")},
              "chronological_split": calibration["chronological_split"], "input_preprocessing": input_metadata,
              "hyperparameter_grid": {"alphas": alphas, "gammas": gammas, "floor_fraction": args.floor_fraction},
              "bootstrap": {"replicates": args.bootstrap, "block_length_months": args.bootstrap_block, "seed": args.seed,
                            "paired_shared_indices": True, "intervals": "pointwise percentile 95%; not multiplicity-adjusted"},
              "centrality_audit": audit, "methods": {}, "checkpoint_attribution_sensitivity": {}}
    uniform_alpha, uniform_gamma, tuning, _, predictions["uniform"] = fit_prior(
        designs, np.ones((len(names), 24, len(names))), target[split.validation], scale,
        alphas=alphas, gammas=[0.0], floor_fraction=args.floor_fraction, fixed_parameters=fixed_parameters)
    result["methods"]["uniform"] = {"alpha": uniform_alpha, "gamma": uniform_gamma,
                                      "validation_nrmse": min(tuning.values()), "validation_scores": tuning}
    if not fixed_parameters:
        result["maximum_cached_uniform_prediction_difference"] = float(np.max(np.abs(predictions["uniform"] - reference_uniform)))
    for key in METHODS:
        alpha, gamma, tuning, _, prediction = fit_prior(
            designs, centralities[key].mean(axis=0), target[split.validation], scale,
            alphas=alphas, gammas=gammas, floor_fraction=args.floor_fraction, fixed_parameters=fixed_parameters)
        predictions[key] = prediction
        result["methods"][key] = {"alpha": alpha, "gamma": gamma, "validation_nrmse": min(tuning.values()), "validation_scores": tuning}
        result["methods"][key].update(
            prior_disabled_by_gamma_zero=bool(gamma == 0),
            prediction_exactly_equals_uniform=bool(np.array_equal(prediction, predictions["uniform"])),
            maximum_prediction_difference_from_uniform=float(np.max(np.abs(prediction - predictions["uniform"]))),
        )
        print(f"{key}: alpha={alpha:g}, gamma={gamma:g}, validation={min(tuning.values()):.6f}, "
              f"test={mean_cell_nrmse(prediction, target[split.test], scale):.6f}", flush=True)
    reference_score = reference["test_nrmse"]["target_xi_shapley_prior"]
    if abs(mean_cell_nrmse(predictions["xi_shapley"], target[split.test], scale) - reference_score) > 1e-9:
        raise ValueError("Primary Xi result was not reproduced.")
    test_target = target[split.test]
    print("Shared paired block bootstrap.", flush=True)
    aggregate, lead_bootstrap, counts = paired_bootstrap(predictions, test_target, scale, args.bootstrap, args.bootstrap_block, args.seed)
    scores = {key: mean_cell_nrmse(v, test_target, scale) for key, v in predictions.items()}
    for key, prediction in predictions.items():
        entry = result["methods"].setdefault(key, {})
        entry.update(test_nrmse=scores[key], test_acc=mean_cell_acc(prediction, test_target),
                     rmse_reduction_vs_frozen_percent=100 * (1 - scores[key] / scores["frozen"]),
                     lead_nrmse=cell_rmse(prediction, test_target, scale).mean(axis=0).tolist(),
                     target_nrmse=cell_rmse(prediction, test_target, scale).mean(axis=1).tolist())
        for baseline in ("uniform", "xi_shapley"):
            gain = aggregate[baseline] - aggregate[key]
            entry["gain_over_" + ("xi" if baseline == "xi_shapley" else baseline)] = {
                "point": scores[baseline] - scores[key], "ci95": np.percentile(gain, [2.5, 97.5]).tolist(),
                "lead_ci95": np.percentile(lead_bootstrap[baseline] - lead_bootstrap[key], [2.5, 97.5], axis=0).tolist()}
    for index, seed in enumerate(args.checkpoint_seeds):
        row = {}
        for key in METHODS:
            alpha, gamma, tuning, _, prediction = fit_prior(
                designs, centralities[key][index], target[split.validation], scale,
                alphas=alphas, gammas=gammas, floor_fraction=args.floor_fraction, fixed_parameters=fixed_parameters)
            row[key] = {"alpha": alpha, "gamma": gamma, "validation_nrmse": min(tuning.values()),
                        "test_nrmse": mean_cell_nrmse(prediction, test_target, scale)}
        result["checkpoint_attribution_sensitivity"][str(seed)] = row
        print(f"Checkpoint {seed} attribution sensitivity complete.", flush=True)
    np.savez_compressed(args.output_dir / "evaluation_arrays.npz", **{f"prediction_{k}": v for k, v in predictions.items()},
                        **{f"bootstrap_nrmse_{k}": v for k, v in aggregate.items()},
                        **{f"bootstrap_lead_nrmse_{k}": v for k, v in lead_bootstrap.items()},
                        bootstrap_month_weights=counts, test_target=test_target, target_scale=scale,
                        test_issue_dates=split.issue_dates[split.test].astype(str))
    figure = args.output_dir / "information_prior_comparison.png"
    panel_a_result = None
    if args.panel_a_independent_summary is not None:
        if not fixed_parameters:
            raise ValueError("An independent panel-a summary is only used alongside fixed-Xi panels b,c.")
        panel_a_result = json.loads(args.panel_a_independent_summary.read_text())
        if panel_a_result["parameter_mode"] != "independently_tuned":
            raise ValueError("Panel a must use independent validation selection.")
        if any(panel_a_result["methods"][key]["gamma"] <= 0 for key in METHODS):
            raise ValueError("Panel a must keep information weighting active for every prior (gamma > 0).")
        for field in ("included_methods", "estimator", "samples", "chronological_split",
                      "checkpoint_seeds", "intervention_n_samples"):
            if panel_a_result[field] != result[field]:
                raise ValueError(f"Panel-a experiment mismatch: {field}.")
        result["panel_a_source_summary"] = str(args.panel_a_independent_summary.resolve())
    result["figure_panel_parameter_modes"] = {
        "a": (panel_a_result or result)["parameter_mode"],
        "b": result["parameter_mode"], "c": result["parameter_mode"],
    }
    plot_results(result, predictions, test_target, scale, figure, panel_a_result=panel_a_result)
    result["figure"] = str(figure)
    result["elapsed_seconds"] = time.monotonic() - started
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Completed in {result['elapsed_seconds']:.1f}s. Figure: {figure}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for option, default in (("input-dir", DEFAULT_INPUT), ("cache-dir", DEFAULT_CACHE),
                            ("calibration-dir", DEFAULT_CALIBRATION), ("xi-reference", XI_REFERENCE),
                            ("output-dir", CURRENT_OUTPUT)):
        parser.add_argument("--" + option, type=Path, default=default)
    parser.add_argument("--centrality-cache", type=Path, default=CURRENT_OUTPUT / "information_centralities.npz")
    parser.add_argument("--fixed-xi-parameters", action="store_true",
                        default=True, help="Fix alpha and gamma to the original all-order Xi validation optimum (default).")
    parser.add_argument("--independent-validation-selection", action="store_false", dest="fixed_xi_parameters",
                        help="Independently select each prior on the positive-gamma validation grid.")
    parser.add_argument("--panel-a-independent-summary", type=Path,
                        help="Display independently tuned cached results in panel a; retain fixed settings in b,c.")
    parser.add_argument("--checkpoint-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n-samples", type=int, default=16384)
    parser.add_argument("--sampling-seed", type=int, default=20260901)
    parser.add_argument("--intervention-bound", type=float, default=4.0)
    parser.add_argument("--start-month", type=int, default=0)
    parser.add_argument("--covariance-ridge", type=float, default=1e-6)
    parser.add_argument("--alphas", default="100,300,1000,3000,10000,30000")
    parser.add_argument("--gammas", default="0.5,1,2,3")
    parser.add_argument("--floor-fraction", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-block", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261001)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
