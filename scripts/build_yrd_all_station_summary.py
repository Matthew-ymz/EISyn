from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from yrd.config import YRDExperimentConfig
from yrd.coupling import (
    compute_subset_nis_summary,
    estimate_residual_covariance,
    jacobian_for_target_subset,
    select_evenly_spaced_indices,
)
from yrd.data import (
    build_windowed_samples,
    build_time_splits,
    flatten_input_group_indices,
    load_dataset,
    load_station_metadata,
    select_station_metadata,
)
from yrd.models import JointStationMLP
from yrd.train import _predict_numpy, rebuild_joint_model_from_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all-station Shanghai summary figures.")
    parser.add_argument("--sample-count", type=int, default=2, help="Number of evenly spaced test samples for all-station NIS.")
    parser.add_argument("--rep-stations", type=int, default=5, help="Number of representative stations in the time-series figure.")
    parser.add_argument("--rep-hours", type=int, default=168, help="Trailing observation window for representative trajectories.")
    parser.add_argument(
        "--cache-run-tag",
        default=None,
        help="Optional cache run tag under exp/cache/yrd_coupling/. Defaults to the newest retained full-run cache.",
    )
    return parser.parse_args()


def resolve_cache_dir(root: Path, run_tag: str | None) -> Path:
    cache_root = root / "exp" / "cache" / "yrd_coupling"
    if run_tag is not None:
        cache_dir = cache_root / run_tag
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Requested cache directory does not exist: {cache_dir}")
        return cache_dir

    for candidate in (
        "shanghai_full_v5_resmlp",
        "shanghai_full_v4_o3_station_graph",
        "shanghai_full_v2_global_graph",
        "shanghai_full_v1",
    ):
        cache_dir = cache_root / candidate
        if cache_dir.is_dir():
            return cache_dir
    raise FileNotFoundError("No retained Shanghai full-run cache was found under exp/cache/yrd_coupling/.")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)

    root = Path(__file__).resolve().parents[1]
    cfg = replace(YRDExperimentConfig(root_dir=root), hidden_dim=128, batch_size=128, epochs=12, seed=0)
    cache_dir = resolve_cache_dir(root, args.cache_run_tag)
    fig_dir = root / "fig" / "yrd_shanghai"
    fig_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    ds, metadata = load_dataset(cfg, smoke=False, city_en="shanghai")
    full_city_metadata = select_station_metadata(
        load_station_metadata(cfg),
        available_station_ids=ds["station"].values.tolist(),
        city_en="shanghai",
    )
    sample_bundle = build_windowed_samples(ds, metadata, cfg, smoke=False)
    splits = sample_bundle["splits"]
    station_ids = sample_bundle["station_ids"]
    station_names = metadata["station_name"].tolist()
    x_test = splits["test"]["X"]
    y_test_scaled = splits["test"]["targets"]

    checkpoint_payload = torch.load(cache_dir / "joint_model_checkpoint.pt", map_location="cpu")
    if "model_kwargs" in checkpoint_payload:
        model = rebuild_joint_model_from_checkpoint(checkpoint_payload)
    else:
        model = JointStationMLP(
            n_stations=sample_bundle["n_stations"],
            n_features=sample_bundle["n_features"],
            history_hours=cfg.history_hours,
            target_dim=y_test_scaled[cfg.horizons[0]].shape[1],
            hidden_dim=cfg.hidden_dim,
            horizons=cfg.horizons,
        )
        model.load_state_dict(checkpoint_payload["state_dict"])
    model.eval()

    joint_scaled_predictions = _predict_numpy(model, x_test, cfg.horizons)
    sigma_eps_by_horizon = {
        horizon: estimate_residual_covariance(y_test_scaled[horizon], joint_scaled_predictions[horizon])
        for horizon in cfg.horizons
    }

    selected_test_indices = select_evenly_spaced_indices(len(splits["test"]["times"]), args.sample_count)
    selected_test_times = [splits["test"]["times"][index] for index in selected_test_indices]

    records: list[dict[str, object]] = []
    for station_index, station_id in enumerate(station_ids):
        source_groups = flatten_input_group_indices(
            cfg,
            n_stations=sample_bundle["n_stations"],
            station_index=station_index,
        )
        target_indices = [
            station_index * len(cfg.target_variables) + offset for offset in range(len(cfg.target_variables))
        ]
        for sample_index, sample_time in zip(selected_test_indices, selected_test_times):
            sample_x = torch.from_numpy(x_test[sample_index : sample_index + 1]).to(dtype=torch.float32)
            flat_sample = sample_x.reshape(-1).detach().clone().requires_grad_(True)
            for horizon in cfg.horizons:

                def horizon_model(tensor: torch.Tensor, target_horizon: int = horizon) -> torch.Tensor:
                    shaped = tensor.reshape(
                        1,
                        cfg.history_hours,
                        sample_bundle["n_stations"],
                        sample_bundle["n_features"],
                    )
                    return model(shaped)[target_horizon].reshape(-1)

                jacobian = jacobian_for_target_subset(
                    horizon_model,
                    flat_sample,
                    target_indices=target_indices,
                ).detach().cpu().numpy()
                summary = compute_subset_nis_summary(
                    jacobian=jacobian,
                    sigma_eps=sigma_eps_by_horizon[horizon],
                    source_groups=source_groups,
                    target_indices=target_indices,
                    box_size=cfg.box_size,
                )
                record = {
                    "station_id": station_id,
                    "station_name": station_names[station_index],
                    "sample_index": int(sample_index),
                    "time": sample_time,
                    "horizon": f"{horizon}h",
                    "ei_nis": float(summary["ei_nis"]),
                    "syn_nis": float(summary["syn_nis"]),
                }
                for group_name, value in summary["group_ei_nis"].items():
                    record[group_name] = float(value)
                records.append(record)

    records_df = pd.DataFrame(records)
    station_summary = records_df.groupby(["station_id", "horizon"], as_index=False).agg(
        syn_mean=("syn_nis", "mean")
    )
    group_cols = [
        "local_o3_history",
        "local_pm25_history",
        "local_meteorology_history",
        "cross_station_pollutants",
    ]
    group_long = records_df.melt(
        id_vars=["station_id", "station_name", "horizon"],
        value_vars=group_cols,
        var_name="group",
        value_name="group_ei_nis",
    )
    agg_group_summary = group_long.groupby(["horizon", "group"], as_index=False).agg(
        abs_mean=("group_ei_nis", lambda series: float(np.mean(np.abs(series))))
    )
    (cache_dir / "all_station_coupling_summary.json").write_text(
        json.dumps(
            {
                "records": records,
                "station_summary": station_summary.to_dict(orient="records"),
                "aggregate_group_summary": agg_group_summary.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    fig, ax = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
    ax.scatter(full_city_metadata["lon"], full_city_metadata["lat"], color="#2A9D8F", s=68, alpha=0.9, zorder=3)
    for _, row in full_city_metadata.iterrows():
        ax.text(row["lon"] + 0.004, row["lat"] + 0.002, row["station_id"], fontsize=7.2, color="#233142")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.text(0.02, 1.02, "A  station layout", transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")
    fig.savefig(fig_dir / "shanghai_station_layout.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    rep_meta = full_city_metadata.sort_values("lon").reset_index(drop=True)
    rep_ids = rep_meta.iloc[np.linspace(0, len(rep_meta) - 1, num=args.rep_stations, dtype=int)]["station_id"].tolist()
    window = ds.isel(time=slice(max(0, len(ds.time) - args.rep_hours), len(ds.time))).to_dataframe().reset_index()
    window = window[window["station"].isin(rep_ids)]
    colors = ["#1F3B73", "#2A9D8F", "#C56B3C", "#A23E48", "#6C7A89", "#7C6A0A"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.3), constrained_layout=True, sharex=True)
    for axis, variable in zip(axes, cfg.target_variables):
        for color, station_id in zip(colors, rep_ids):
            subset = window[window["station"] == station_id]
            axis.plot(subset["time"], subset[variable], label=station_id, linewidth=1.8, alpha=0.95, color=color)
        axis.set_ylabel(variable)
        axis.grid(True, alpha=0.25, linewidth=0.6)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    axes[-1].set_xlabel("Time")
    axes[0].text(
        0.01,
        1.02,
        "A  representative trajectories",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(fig_dir / "shanghai_multi_station_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    station_syn_pivot = station_summary.pivot(index="station_id", columns="horizon", values="syn_mean")[
        ["1h", "24h"]
    ].sort_values("24h", ascending=False)
    group_abs_pivot = agg_group_summary.pivot(index="group", columns="horizon", values="abs_mean")[["1h", "24h"]]
    group_rank_24h = agg_group_summary[agg_group_summary["horizon"].eq("24h")].sort_values(
        "abs_mean", ascending=True
    )

    fig = plt.figure(figsize=(15.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.1], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    box_data = [station_summary.loc[station_summary["horizon"].eq(horizon), "syn_mean"].values for horizon in ["1h", "24h"]]
    box = ax_a.boxplot(box_data, tick_labels=["1h", "24h"], patch_artist=True, widths=0.55)
    for patch, horizon in zip(box["boxes"], ["1h", "24h"]):
        patch.set_facecolor({"1h": "#2A9D8F", "24h": "#C56B3C"}[horizon])
        patch.set_alpha(0.45)
    for index, horizon in enumerate(["1h", "24h"], start=1):
        vals = station_summary.loc[station_summary["horizon"].eq(horizon), "syn_mean"].values
        ax_a.scatter(
            np.linspace(index - 0.10, index + 0.10, len(vals)),
            vals,
            s=28,
            alpha=0.75,
            color={"1h": "#2A9D8F", "24h": "#C56B3C"}[horizon],
        )
    ax_a.set_xlabel("Horizon")
    ax_a.set_ylabel(r"Station-level mean $Syn_p^{\mathrm{nis}}$")
    ax_a.grid(True, axis="y", alpha=0.2, linewidth=0.6)
    ax_a.text(0.01, 1.02, "A  station-level synergy", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")

    image_b = ax_b.imshow(station_syn_pivot.values, cmap="viridis", aspect="auto")
    ax_b.set_xticks(range(len(station_syn_pivot.columns)))
    ax_b.set_xticklabels(station_syn_pivot.columns)
    ax_b.set_yticks(range(len(station_syn_pivot.index)))
    ax_b.set_yticklabels(station_syn_pivot.index, fontsize=8)
    for row_index, station_id in enumerate(station_syn_pivot.index):
        for col_index, horizon in enumerate(station_syn_pivot.columns):
            ax_b.text(
                col_index,
                row_index,
                f"{station_syn_pivot.loc[station_id, horizon]:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
            )
    fig.colorbar(image_b, ax=ax_b, fraction=0.046, pad=0.04)
    ax_b.text(0.01, 1.02, "B  all-station horizon heatmap", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")

    image_c = ax_c.imshow(group_abs_pivot.values, cmap="Blues", aspect="auto")
    ax_c.set_xticks(range(len(group_abs_pivot.columns)))
    ax_c.set_xticklabels(group_abs_pivot.columns)
    ax_c.set_yticks(range(len(group_abs_pivot.index)))
    ax_c.set_yticklabels(group_abs_pivot.index)
    for row_index, group in enumerate(group_abs_pivot.index):
        for col_index, horizon in enumerate(group_abs_pivot.columns):
            ax_c.text(
                col_index,
                row_index,
                f"{group_abs_pivot.loc[group, horizon]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image_c, ax=ax_c, fraction=0.046, pad=0.04)
    ax_c.text(0.01, 1.02, "C  group-level absolute EI", transform=ax_c.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")

    ax_d.barh(group_rank_24h["group"], group_rank_24h["abs_mean"], color="#1F3B73", alpha=0.9)
    ax_d.set_xlabel(r"24h mean $|EI^{\mathrm{nis}}|$")
    ax_d.grid(True, axis="x", alpha=0.2, linewidth=0.6)
    ax_d.text(0.01, 1.02, "D  24h group ranking", transform=ax_d.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")
    fig.savefig(fig_dir / "shanghai_station_coupling_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
