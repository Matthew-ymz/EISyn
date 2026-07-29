#!/usr/bin/env python3
"""Validate IOD synergy against ORAS5 forecast degradation under mode misalignment."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm


IOD_INDEX = 4
MODE_COUNT = 11
LEAD_COUNT = 24


def make_parameters(torch):
    return SimpleNamespace(
        d_size=256,
        device=torch.device("cpu"),
        input_channal=5,
        patch_size=[2, 2],
        emb_spatial_size=216,
        nheads=4,
        dim_feedforward=512,
        dropout=0.2,
        num_encoder_layers=4,
        num_decoder_layers=4,
        val_relative=[None] * 10,
        t20d_mode=1,
        mode_interaction="1",
        his_len=12,
        pred_len=24,
        autoregressive=0,
    )


def load_model(checkpoint: str, source: Path):
    import torch

    sys.path.insert(0, str(source))
    sys.modules.setdefault("pynvml", ModuleType("pynvml"))
    from models import UniCM

    parameters = make_parameters(torch)
    model = UniCM(parameters)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_iod(model, windows: np.ndarray, timestamps: np.ndarray, batch_size: int):
    import torch

    output = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            modes = torch.from_numpy(windows[start : start + batch_size]).float()
            time = torch.from_numpy(timestamps[start : start + batch_size]).long()
            internal = modes.permute(0, 2, 1).unsqueeze(-1).unsqueeze(2)
            pred, _, _ = model.forward_sep(
                internal,
                time,
                model.encoder_mode,
                model.decoder_mode,
                model.linear_output_mode,
                model.predictor_emb_mode,
                model.predictand_emb_mode,
                1,
                [1, 1],
                train=False,
            )
            pred = pred.squeeze(-1).squeeze(2).permute(0, 2, 1)
            output.append(pred[:, IOD_INDEX].cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def derangements(count: int, repeats: int, rng: np.random.Generator) -> np.ndarray:
    result = []
    base = np.arange(count)
    while len(result) < repeats:
        candidate = rng.permutation(count)
        if np.all(candidate != base):
            result.append(candidate)
    return np.stack(result)


def build_permuted_windows(
    windows: np.ndarray,
    donor_indices: np.ndarray,
) -> np.ndarray:
    repeats, samples = donor_indices.shape
    result = np.repeat(windows[None], repeats, axis=0)
    non_iod = [index for index in range(MODE_COUNT) if index != IOD_INDEX]
    for repeat in range(repeats):
        donors = windows[donor_indices[repeat]]
        result[repeat][:, non_iod, :12] = donors[:, non_iod, :12]
    return result.reshape(repeats * samples, MODE_COUNT, 36)


def correlation_by_lead(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.corrcoef(prediction[:, lead], target[:, lead])[0, 1]
            for lead in range(target.shape[1])
        ]
    )


def metrics(prediction: np.ndarray, target: np.ndarray):
    return {
        "acc": correlation_by_lead(prediction, target),
        "rmse": np.sqrt(np.mean((prediction - target) ** 2, axis=0)),
        "mae": np.mean(np.abs(prediction - target), axis=0),
    }


def bootstrap_deltas(
    baseline: np.ndarray,
    perturbed: np.ndarray,
    target: np.ndarray,
    *,
    repeats: int,
    rng: np.random.Generator,
):
    samples = target.shape[0]
    delta_rmse = np.empty((repeats, LEAD_COUNT))
    delta_acc = np.empty((repeats, LEAD_COUNT))
    for repeat in range(repeats):
        index = rng.integers(0, samples, samples)
        base = metrics(baseline[index], target[index])
        changed = [
            metrics(item[index], target[index])
            for item in perturbed
        ]
        changed_rmse = np.mean([item["rmse"] for item in changed], axis=0)
        changed_acc = np.mean([item["acc"] for item in changed], axis=0)
        delta_rmse[repeat] = changed_rmse - base["rmse"]
        delta_acc[repeat] = base["acc"] - changed_acc
    return delta_rmse, delta_acc


def circular_shift_p(x: np.ndarray, y: np.ndarray) -> float:
    observed = abs(spearmanr(x, y).statistic)
    null = [
        abs(spearmanr(x, np.roll(y, shift)).statistic)
        for shift in range(1, len(y))
    ]
    return float((1 + np.sum(np.asarray(null) >= observed)) / (1 + len(null)))


def relation_stats(
    xi: np.ndarray,
    delta_rmse: np.ndarray,
    delta_acc: np.ndarray,
    baseline_acc: np.ndarray,
):
    output = {}
    for label, index in {
        "all_leads": np.arange(24),
        "year1": np.arange(12),
        "year2": np.arange(12, 24),
    }.items():
        x = xi[index]
        y = delta_rmse[index]
        acc = baseline_acc[index]
        output[label] = {
            "n": int(len(index)),
            "spearman_xi_delta_rmse": float(spearmanr(x, y).statistic),
            "pearson_xi_delta_rmse": float(pearsonr(x, y).statistic),
            "spearman_xi_delta_acc": float(
                spearmanr(x, delta_acc[index]).statistic
            ),
            "pearson_xi_delta_acc": float(
                pearsonr(x, delta_acc[index]).statistic
            ),
            "spearman_xi_baseline_acc": float(spearmanr(x, acc).statistic),
            "circular_shift_p": circular_shift_p(x, y),
            "circular_shift_p_delta_acc": circular_shift_p(
                x, delta_acc[index]
            ),
        }
    return output


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_results(
    output_base: Path,
    xi_mean: np.ndarray,
    xi_std: np.ndarray,
    baseline_acc: np.ndarray,
    perm_acc: np.ndarray,
    perm_acc_low: np.ndarray,
    perm_acc_high: np.ndarray,
    zero_acc: np.ndarray,
    delta_rmse: np.ndarray,
    delta_rmse_low: np.ndarray,
    delta_rmse_high: np.ndarray,
    relation: dict,
) -> None:
    configure_plotting()
    leads = np.arange(1, 25)
    blue = "#3F6F9F"
    orange = "#D9822B"
    grey = "#7D8790"
    green = "#3D8B6D"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.8),
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    for axis in (ax_a, ax_b, ax_c):
        axis.axvspan(6.5, 8.5, color="#F2D7A0", alpha=0.35, lw=0)
    ax_a.text(
        7.5,
        0.985,
        "Jul–Aug targets",
        transform=ax_a.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6.5,
        color="#8A6428",
    )

    ax_a.plot(leads, xi_mean, color=blue, marker="o", ms=2.8, lw=1.3)
    ax_a.fill_between(
        leads, xi_mean - xi_std, xi_mean + xi_std, color=blue, alpha=0.16, lw=0
    )
    ax_a.set_ylabel(r"IOD-target synergy $\Xi$ (bit)")
    ax_a.set_xlabel("Prediction lead (months)")

    ax_b.plot(leads, baseline_acc, color=blue, lw=1.5, marker="o", ms=2.5, label="Full history")
    ax_b.plot(leads, perm_acc, color=orange, lw=1.4, label="IOD–rest misaligned")
    ax_b.fill_between(leads, perm_acc_low, perm_acc_high, color=orange, alpha=0.15, lw=0)
    ax_b.plot(leads, zero_acc, color=grey, lw=1.2, ls="--", label="IOD history only")
    ax_b.axhline(0, color="0.75", lw=0.7)
    ax_b.set_ylabel("IOD forecast ACC")
    ax_b.set_xlabel("Prediction lead (months)")

    ax_c.plot(leads, delta_rmse, color=green, lw=1.5, marker="o", ms=2.5)
    ax_c.fill_between(
        leads, delta_rmse_low, delta_rmse_high, color=green, alpha=0.17, lw=0
    )
    ax_c.axhline(0, color="0.55", lw=0.8)
    ax_c.set_ylabel(r"RMSE increase after misalignment ($^\circ$C)")
    ax_c.set_xlabel("Prediction lead (months)")

    scatter = ax_d.scatter(
        xi_mean,
        delta_rmse,
        c=leads,
        cmap="viridis",
        s=24,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    slope, intercept = np.polyfit(xi_mean, delta_rmse, 1)
    grid = np.linspace(xi_mean.min(), xi_mean.max(), 100)
    ax_d.plot(grid, slope * grid + intercept, color="0.35", lw=1.0, zorder=2)
    for lead in (7, 8, 19, 20):
        index = lead - 1
        ax_d.annotate(
            str(lead),
            (xi_mean[index], delta_rmse[index]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    rho = relation["all_leads"]["spearman_xi_delta_rmse"]
    p_value = relation["all_leads"]["circular_shift_p"]
    ax_d.text(
        0.03,
        0.97,
        rf"Spearman $\rho$={rho:.2f}" + "\n" + rf"shift $P$={p_value:.3f}",
        transform=ax_d.transAxes,
        va="top",
    )
    ax_d.set_xlabel(r"IOD-target synergy $\Xi$ (bit)")
    ax_d.set_ylabel(r"RMSE increase ($^\circ$C)")
    colorbar = fig.colorbar(scatter, ax=ax_d, pad=0.02, fraction=0.05)
    colorbar.set_label("Lead")

    for label, axis in zip("abcd", axes.ravel()):
        axis.text(
            -0.15,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=3,
        frameon=False,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ORAS5/modeformer_1980_2014/model_inputs.npz"),
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=Path(
            "data/ORAS5/modeformer_1980_2014/modeformer_predictions.npz"
        ),
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("data/UniCM-checkpoint")
    )
    parser.add_argument(
        "--xi-rows",
        type=Path,
        default=Path(
            "results/unicm_target_resolved_xi_tm_degree1_signed_n8192/"
            "target_resolved_xi_rows.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/unicm_iod_realdata_synergy_validation"),
    )
    parser.add_argument("--permutations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=20260728)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    with np.load(args.input, allow_pickle=False) as data:
        windows_all = data["mode_windows"].astype(np.float32)
        timestamps_all = data["timestamps"].astype(np.int64)
        dates = data["dates"]
        metadata = json.loads(str(data["metadata"]))
    with np.load(args.baseline_predictions, allow_pickle=False) as data:
        baseline_all = data["predictions_by_seed"].astype(np.float32)

    selection = np.flatnonzero(timestamps_all[:, 0] == 0)
    windows = windows_all[selection]
    timestamps = timestamps_all[selection]
    baseline = baseline_all[:, selection, IOD_INDEX]
    targets = windows[:, IOD_INDEX, 12:]
    target_dates = np.stack(
        [dates[index + 12 : index + 36] for index in selection]
    )
    if windows.shape[0] != 33:
        raise RuntimeError(f"Expected 33 January-start windows, found {windows.shape[0]}")

    donors = derangements(windows.shape[0], args.permutations, rng)
    permuted = build_permuted_windows(windows, donors)
    permuted_timestamps = np.repeat(
        timestamps[None], args.permutations, axis=0
    ).reshape(-1, 36)
    iod_only = windows.copy()
    iod_only[:, [index for index in range(MODE_COUNT) if index != IOD_INDEX], :12] = 0

    sst_std = float(metadata["sst_std"])
    targets_c = targets * sst_std
    baseline_c = baseline * sst_std
    cache_path = args.output_dir / "metrics.npz"
    if args.reuse_cache and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            perm_predictions_c = cache["permuted_predictions_c"]
            zero_predictions_c = cache["iod_only_predictions_c"]
        expected_shape = (3, args.permutations, windows.shape[0], LEAD_COUNT)
        if perm_predictions_c.shape != expected_shape:
            raise RuntimeError(
                f"Cached permutation shape {perm_predictions_c.shape} != {expected_shape}"
            )
    else:
        source = args.checkpoint_root / "src"
        checkpoints = sorted(
            glob.glob(str(source / "experiments" / "*" / "model_save" / "model_best.pkl"))
        )
        if len(checkpoints) != 3:
            raise RuntimeError(f"Expected three checkpoints, found {len(checkpoints)}")

        perm_predictions = []
        zero_predictions = []
        for checkpoint in tqdm(checkpoints, desc="Modeformer seeds", unit="seed"):
            model = load_model(checkpoint, source)
            prediction = predict_iod(
                model, permuted, permuted_timestamps, args.batch_size
            ).reshape(args.permutations, windows.shape[0], LEAD_COUNT)
            perm_predictions.append(prediction)
            zero_predictions.append(
                predict_iod(model, iod_only, timestamps, args.batch_size)
            )

        perm_predictions = np.stack(perm_predictions)
        zero_predictions = np.stack(zero_predictions)
        perm_predictions_c = perm_predictions * sst_std
        zero_predictions_c = zero_predictions * sst_std

    baseline_metrics = [metrics(item, targets_c) for item in baseline_c]
    zero_metrics = [metrics(item, targets_c) for item in zero_predictions_c]
    perm_metrics = [
        [metrics(prediction, targets_c) for prediction in seed_predictions]
        for seed_predictions in perm_predictions_c
    ]

    ensemble_baseline = baseline_c.mean(axis=0)
    ensemble_permuted = perm_predictions_c.mean(axis=0)
    ensemble_zero = zero_predictions_c.mean(axis=0)
    ensemble_base_metrics = metrics(ensemble_baseline, targets_c)
    ensemble_zero_metrics = metrics(ensemble_zero, targets_c)
    ensemble_perm_metrics = [
        metrics(prediction, targets_c) for prediction in ensemble_permuted
    ]
    mean_permuted_metrics = {
        name: np.mean([item[name] for item in ensemble_perm_metrics], axis=0)
        for name in ("acc", "rmse", "mae")
    }

    bootstrap_rmse, bootstrap_acc = bootstrap_deltas(
        ensemble_baseline,
        ensemble_permuted,
        targets_c,
        repeats=2000,
        rng=rng,
    )
    delta_rmse = (
        mean_permuted_metrics["rmse"] - ensemble_base_metrics["rmse"]
    )
    delta_acc = ensemble_base_metrics["acc"] - mean_permuted_metrics["acc"]
    delta_rmse_ci = np.quantile(bootstrap_rmse, [0.025, 0.975], axis=0)
    delta_acc_ci = np.quantile(bootstrap_acc, [0.025, 0.975], axis=0)

    xi_rows = pd.read_csv(args.xi_rows)
    xi_rows = xi_rows.loc[xi_rows["display_target"].eq("IOD")].copy()
    xi_seed = np.stack(
        [
            xi_rows.loc[xi_rows["seed"].eq(seed)]
            .sort_values("lead")["xi_target"]
            .to_numpy()
            for seed in (1, 2, 3)
        ]
    )
    xi_mean = xi_seed.mean(axis=0)
    xi_std = xi_seed.std(axis=0, ddof=1)
    relation = relation_stats(
        xi_mean,
        delta_rmse,
        delta_acc,
        ensemble_base_metrics["acc"],
    )

    seed_delta_rmse = np.stack(
        [
            np.mean(
                [
                    item["rmse"] - baseline_metrics[seed]["rmse"]
                    for item in perm_metrics[seed]
                ],
                axis=0,
            )
            for seed in range(3)
        ]
    )
    seed_delta_acc = np.stack(
        [
            np.mean(
                [
                    baseline_metrics[seed]["acc"] - item["acc"]
                    for item in perm_metrics[seed]
                ],
                axis=0,
            )
            for seed in range(3)
        ]
    )
    positive_seed_count = np.sum(seed_delta_rmse > 0, axis=0)

    perm_acc = np.stack([item["acc"] for item in ensemble_perm_metrics])
    july_august = np.asarray([6, 7])
    other_year1 = np.asarray([index for index in range(12) if index not in july_august])
    july_august_contrast = float(
        delta_rmse[july_august].mean() - delta_rmse[other_year1].mean()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "metrics.npz",
        selected_window_indices=selection,
        donor_indices=donors,
        target_dates=target_dates,
        targets_c=targets_c,
        baseline_predictions_c=baseline_c,
        permuted_predictions_c=perm_predictions_c,
        iod_only_predictions_c=zero_predictions_c,
        xi_by_seed=xi_seed,
        xi_mean=xi_mean,
        baseline_acc=ensemble_base_metrics["acc"],
        baseline_rmse=ensemble_base_metrics["rmse"],
        permuted_acc=mean_permuted_metrics["acc"],
        permuted_rmse=mean_permuted_metrics["rmse"],
        iod_only_acc=ensemble_zero_metrics["acc"],
        iod_only_rmse=ensemble_zero_metrics["rmse"],
        delta_rmse=delta_rmse,
        delta_rmse_ci=delta_rmse_ci,
        delta_acc=delta_acc,
        delta_acc_ci=delta_acc_ci,
        seed_delta_rmse=seed_delta_rmse,
        seed_delta_acc=seed_delta_acc,
        positive_seed_count=positive_seed_count,
    )

    summary = {
        "question": (
            "Does IOD-target synergy predict real-data skill loss when only "
            "IOD–rest historical alignment is destroyed?"
        ),
        "samples": int(windows.shape[0]),
        "permutations": args.permutations,
        "checkpoint_seeds": [1, 2, 3],
        "relation": relation,
        "mean_delta_rmse_c": float(delta_rmse.mean()),
        "mean_delta_acc": float(delta_acc.mean()),
        "july_august_year1_delta_rmse_c": float(delta_rmse[july_august].mean()),
        "other_months_year1_delta_rmse_c": float(delta_rmse[other_year1].mean()),
        "july_august_minus_other_delta_rmse_c": july_august_contrast,
        "lead7_delta_rmse_c": float(delta_rmse[6]),
        "lead8_delta_rmse_c": float(delta_rmse[7]),
        "lead7_delta_acc": float(delta_acc[6]),
        "lead8_delta_acc": float(delta_acc[7]),
        "positive_seed_count_by_lead": positive_seed_count.tolist(),
        "all_three_seeds_positive_leads": (
            np.flatnonzero(positive_seed_count == 3) + 1
        ).tolist(),
        "preprocessing": metadata,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plot_results(
        args.output_dir / "iod_synergy_skill_validation",
        xi_mean,
        xi_std,
        ensemble_base_metrics["acc"],
        perm_acc.mean(axis=0),
        np.quantile(perm_acc, 0.025, axis=0),
        np.quantile(perm_acc, 0.975, axis=0),
        ensemble_zero_metrics["acc"],
        delta_rmse,
        delta_rmse_ci[0],
        delta_rmse_ci[1],
        relation,
    )

    stable = relation["all_leads"]["spearman_xi_delta_rmse"]
    stable_acc = relation["all_leads"]["spearman_xi_delta_acc"]
    report = f"""# IOD real-data synergy validation

## Stable finding

The proposed positive relationship was not validated. Across the 24 matched leads, signed IOD-target synergy has Spearman correlation {stable:.3f} with the RMSE penalty and {stable_acc:.3f} with the ACC penalty after destroying IOD–rest historical alignment. The corresponding circular-shift references are P={relation["all_leads"]["circular_shift_p"]:.3f} and P={relation["all_leads"]["circular_shift_p_delta_acc"]:.3f}.

## Evidence

- ORAS5 January-start histories: n={windows.shape[0]}.
- Frozen Modeformer checkpoints: seeds 1–3.
- Season-matched block derangements: {args.permutations}, shared by all checkpoints.
- Mean RMSE change: {delta_rmse.mean():.4f} °C.
- Lead 7/8 RMSE changes: {delta_rmse[6]:.4f}/{delta_rmse[7]:.4f} °C.
- Lead 7/8 ACC changes: {delta_acc[6]:.4f}/{delta_acc[7]:.4f}.
- July–August first-year RMSE change minus the other first-year target months: {july_august_contrast:.4f} °C.
- Leads with positive RMSE penalties in all three seeds: {summary["all_three_seeds_positive_leads"]}.

Although cross-modal misalignment reduces ACC by {delta_acc.mean():.4f} on average across all leads, the loss is not larger where \(\Xi_\mathrm{{IOD}}\) is high. At lead 7, both RMSE and ACC improve rather than degrade after misalignment; at lead 8, the effects are near zero. Thus the frozen model uses non-IOD histories overall, but the maximum-entropy synergy curve does not predict where those histories improve observational forecast accuracy.

## Controls

IOD history, future targets, initialization season, checkpoints, timestamps, sample count, and non-IOD donor trajectories were fixed or paired. The treatment changes only the year-wise alignment between IOD history and the jointly preserved state of the remaining ten modes.

## Limits

The 33 January-start histories are a short real-data record, neighboring lead cells are autocorrelated, and the released checkpoints were trained jointly with Globalformer even though this readout uses only their Modeformer branch. The shift P value is therefore a conservative descriptive reference rather than a definitive causal test.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
