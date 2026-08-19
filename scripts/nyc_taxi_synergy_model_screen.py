#!/usr/bin/env python3
"""Lightweight controlled screen of NYC taxi demand forecasting models.

The experiment separates predictive accuracy from a model's usefulness for
studying cross-region joint information.  The latter is assessed with two
explicitly labelled proxies rather than interpreted as formal information-
theoretic synergy:

1. joint-input gain over a local-history Ridge reference;
2. cross-region alignment penalty after circularly shifting non-target zones.

Raw monthly Parquet files are streamed one at a time.  Only the compact
half-hourly Manhattan count matrix is cached.
"""

from __future__ import annotations

import argparse
import http.client
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = ROOT / "data" / "nyc_taxi_yellow_2023_30min_manhattan.npz"
RESULTS_JSON = ROOT / "results" / "nyc_taxi_synergy_model_screen_metrics.json"
FIGURE_STEM = ROOT / "fig" / "nyc_taxi_synergy_model_screen"

YEAR = 2023
LAGS = (1, 2, 3, 6, 12, 48, 336)
TRAIN_END = np.datetime64("2023-10-01T00:00")
VALID_END = np.datetime64("2023-11-01T00:00")
SEEDS = (0, 1, 2)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


@dataclass
class Design:
    x: np.ndarray
    y: np.ndarray
    times: np.ndarray
    counts: np.ndarray
    zone_ids: np.ndarray
    train_mask: np.ndarray
    valid_mask: np.ndarray
    test_mask: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray


def fetch_bytes(url: str, timeout: int = 180) -> bytes:
    last_error = None
    for attempt in range(2):
        req = urllib.request.Request(url, headers={"User-Agent": "EISyn/NYC-taxi-screen"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (http.client.IncompleteRead, TimeoutError, urllib.error.URLError) as error:
            last_error = error
            print(f"download attempt={attempt + 1} failed for {url}: {error}", flush=True)
    raise RuntimeError(f"Unable to download {url} after two attempts") from last_error


def aggregate_counts(months: tuple[int, ...], cache_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    if cache_path.exists() and months == tuple(range(1, 13)):
        cached = np.load(cache_path, allow_pickle=False)
        metadata = json.loads(str(cached["metadata"]))
        return cached["counts"], cached["zone_ids"], metadata

    lookup_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    lookup = pd.read_csv(io.BytesIO(fetch_bytes(lookup_url)))
    zone_ids = np.sort(
        lookup.loc[lookup["Borough"].eq("Manhattan"), "LocationID"].astype(int).unique()
    )
    zone_to_col = {int(zone): i for i, zone in enumerate(zone_ids)}

    start = pd.Timestamp(f"{YEAR}-01-01")
    stop = pd.Timestamp(f"{YEAR + 1}-01-01")
    full_times = pd.date_range(start, stop, freq="30min", inclusive="left")
    counts = np.zeros((len(full_times), len(zone_ids)), dtype=np.int32)
    raw_rows = 0
    retained_rows = 0
    monthly_cache_dir = ROOT / "data" / "nyc_taxi_yellow_2023_monthly_30min_manhattan"
    if months == tuple(range(1, 13)):
        monthly_cache_dir.mkdir(parents=True, exist_ok=True)

    for month in months:
        monthly_cache = monthly_cache_dir / f"month_{month:02d}.npz"
        if monthly_cache.exists() and months == tuple(range(1, 13)):
            cached_month = np.load(monthly_cache, allow_pickle=False)
            counts += cached_month["counts"]
            raw_rows += int(cached_month["raw_rows"])
            retained_rows += int(cached_month["retained_rows"])
            print(
                f"month={month:02d} loaded monthly aggregate cache "
                f"raw={int(cached_month['raw_rows']):,} retained={int(cached_month['retained_rows']):,}",
                flush=True,
            )
            continue
        url = (
            "https://d37ci6vzurychx.cloudfront.net/trip-data/"
            f"yellow_tripdata_{YEAR}-{month:02d}.parquet"
        )
        tic = time.time()
        payload = fetch_bytes(url)
        table = pq.read_table(
            io.BytesIO(payload), columns=["tpep_pickup_datetime", "PULocationID"]
        )
        frame = table.to_pandas()
        month_raw_rows = len(frame)
        raw_rows += month_raw_rows
        pickup = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
        zones = pd.to_numeric(frame["PULocationID"], errors="coerce")
        valid = pickup.ge(start) & pickup.lt(stop) & zones.isin(zone_ids)
        pickup = pickup.loc[valid]
        zones = zones.loc[valid].astype(int)
        month_retained_rows = len(pickup)
        retained_rows += month_retained_rows

        time_bin = ((pickup - start).dt.total_seconds() // 1800).astype(np.int64).to_numpy()
        zone_col = zones.map(zone_to_col).to_numpy(dtype=np.int64)
        month_counts = np.zeros_like(counts)
        np.add.at(month_counts, (time_bin, zone_col), 1)
        counts += month_counts
        if months == tuple(range(1, 13)):
            np.savez_compressed(
                monthly_cache,
                counts=month_counts,
                raw_rows=np.array(month_raw_rows),
                retained_rows=np.array(month_retained_rows),
            )
        print(
            f"month={month:02d} raw={len(frame):,} retained={len(pickup):,} "
            f"seconds={time.time() - tic:.1f}",
            flush=True,
        )

    metadata = {
        "year": YEAR,
        "months": list(months),
        "interval_minutes": 30,
        "raw_rows": int(raw_rows),
        "retained_manhattan_rows": int(retained_rows),
        "time_steps": int(len(full_times)),
        "zones": int(len(zone_ids)),
    }
    if months == tuple(range(1, 13)):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            counts=counts,
            zone_ids=zone_ids,
            metadata=np.array(json.dumps(metadata)),
        )
    return counts, zone_ids, metadata


def calendar_features(times: pd.DatetimeIndex) -> np.ndarray:
    half_hour = times.hour.to_numpy() * 2 + times.minute.to_numpy() // 30
    weekday = times.dayofweek.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * half_hour / 48),
            np.cos(2 * np.pi * half_hour / 48),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
        ]
    ).astype(np.float32)


def make_design(counts: np.ndarray, zone_ids: np.ndarray, smoke: bool) -> Design:
    times = pd.date_range(f"{YEAR}-01-01", f"{YEAR + 1}-01-01", freq="30min", inclusive="left")
    if smoke:
        stop = 48 * 90
        counts = counts[:stop]
        times = times[:stop]

    log_counts = np.log1p(counts).astype(np.float32)
    preliminary_train = np.asarray(times < pd.Timestamp("2023-10-01"))
    if smoke:
        preliminary_train = np.arange(len(times)) < int(0.70 * len(times))
    y_mean = log_counts[preliminary_train].mean(axis=0)
    y_scale = log_counts[preliminary_train].std(axis=0)
    y_scale[y_scale < 1e-6] = 1.0
    z = (log_counts - y_mean) / y_scale

    max_lag = max(LAGS)
    target_positions = np.arange(max_lag, len(times))
    lag_blocks = [z[target_positions - lag] for lag in LAGS]
    x = np.concatenate(lag_blocks + [calendar_features(times[target_positions])], axis=1)
    y = z[target_positions]
    target_times = times[target_positions].to_numpy(dtype="datetime64[ns]")

    if smoke:
        n = len(target_times)
        train_mask = np.arange(n) < int(0.70 * n)
        valid_mask = (np.arange(n) >= int(0.70 * n)) & (np.arange(n) < int(0.85 * n))
        test_mask = np.arange(n) >= int(0.85 * n)
    else:
        train_mask = target_times < TRAIN_END
        valid_mask = (target_times >= TRAIN_END) & (target_times < VALID_END)
        test_mask = target_times >= VALID_END

    return Design(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        times=target_times,
        counts=counts[target_positions],
        zone_ids=zone_ids,
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask,
        y_mean=y_mean,
        y_scale=y_scale,
    )


def to_counts(pred_z: np.ndarray, design: Design) -> np.ndarray:
    pred_log = pred_z * design.y_scale + design.y_mean
    return np.maximum(np.expm1(pred_log), 0.0)


def metrics(pred_z: np.ndarray, design: Design, mask: np.ndarray) -> dict:
    true_z = design.y[mask]
    pred_z = np.asarray(pred_z)[mask]
    true_counts = design.counts[mask]
    pred_counts = to_counts(pred_z, design)
    per_zone_rmse = np.sqrt(np.mean((pred_z - true_z) ** 2, axis=0))
    return {
        "log_scaled_rmse": float(np.sqrt(np.mean((pred_z - true_z) ** 2))),
        "mean_zone_rmse": float(per_zone_rmse.mean()),
        "median_zone_rmse": float(np.median(per_zone_rmse)),
        "per_zone_log_rmse": per_zone_rmse.astype(float).tolist(),
        "raw_mae_rides": float(np.mean(np.abs(pred_counts - true_counts))),
        "wape": float(np.sum(np.abs(pred_counts - true_counts)) / np.sum(true_counts)),
    }


def fit_best_ridge(x_train, y_train, x_valid, y_valid, alphas=RIDGE_ALPHAS):
    best = None
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        score = float(np.sqrt(np.mean((model.predict(x_valid) - y_valid) ** 2)))
        if best is None or score < best[0]:
            best = (score, alpha, model)
    return best[2], {"alpha": float(best[1]), "valid_log_scaled_rmse": best[0]}


def fit_local_ridge(design: Design):
    n_zones = len(design.zone_ids)
    calendar_cols = np.arange(len(LAGS) * n_zones, design.x.shape[1])
    pred = np.zeros_like(design.y)
    selected_alphas = []
    for zone in range(n_zones):
        own_cols = np.array([lag_i * n_zones + zone for lag_i in range(len(LAGS))])
        cols = np.concatenate([own_cols, calendar_cols])
        model, info = fit_best_ridge(
            design.x[design.train_mask][:, cols],
            design.y[design.train_mask, zone],
            design.x[design.valid_mask][:, cols],
            design.y[design.valid_mask, zone],
        )
        pred[:, zone] = model.predict(design.x[:, cols])
        selected_alphas.append(info["alpha"])
    return pred, {"median_alpha": float(np.median(selected_alphas))}


def interaction_features(x: np.ndarray, n_zones: int, selected_zones: np.ndarray) -> np.ndarray:
    latest = x[:, selected_zones]
    pieces = [x]
    for left in range(len(selected_zones)):
        for right in range(left + 1, len(selected_zones)):
            pieces.append((latest[:, left] * latest[:, right])[:, None])
    return np.concatenate(pieces, axis=1).astype(np.float32)


def shifted_cross_region_x(x_test: np.ndarray, target: int, n_zones: int, shift: int = 37):
    shifted = x_test.copy()
    for lag_i in range(len(LAGS)):
        block = slice(lag_i * n_zones, (lag_i + 1) * n_zones)
        shifted[:, block] = np.roll(x_test[:, block], shift=shift, axis=0)
        shifted[:, lag_i * n_zones + target] = x_test[:, lag_i * n_zones + target]
    return shifted


def cross_region_penalty(
    model,
    design: Design,
    base_pred_z: np.ndarray,
    target_zones: np.ndarray,
    feature_builder=None,
) -> float:
    x_test = design.x[design.test_mask]
    y_test = design.y[design.test_mask]
    penalties = []
    for target in target_zones:
        shifted = shifted_cross_region_x(x_test, int(target), len(design.zone_ids))
        if feature_builder is not None:
            shifted = feature_builder(shifted)
        shifted_pred = model.predict(shifted)[:, target]
        base_rmse = np.sqrt(np.mean((base_pred_z[design.test_mask, target] - y_test[:, target]) ** 2))
        shifted_rmse = np.sqrt(np.mean((shifted_pred - y_test[:, target]) ** 2))
        penalties.append((shifted_rmse - base_rmse) / max(base_rmse, 1e-8))
    return float(np.mean(penalties))


def summarize_seed_runs(seed_runs: list[dict]) -> dict:
    keys = seed_runs[0]["test"].keys()
    test_summary = {}
    for key in keys:
        values = np.array([run["test"][key] for run in seed_runs], dtype=float)
        ddof = 1 if len(values) > 1 else 0
        if values.ndim == 2:
            test_summary[key] = {
                "mean_by_zone": values.mean(axis=0).astype(float).tolist(),
                "sd_by_zone": values.std(axis=0, ddof=ddof).astype(float).tolist(),
            }
        else:
            test_summary[key] = {"mean": float(values.mean()), "sd": float(values.std(ddof=ddof))}
    penalties = np.array([run["cross_region_penalty"] for run in seed_runs], dtype=float)
    penalty_ddof = 1 if len(penalties) > 1 else 0
    return {
        "test": test_summary,
        "cross_region_penalty": {
            "mean": float(penalties.mean()),
            "sd": float(penalties.std(ddof=penalty_ddof)),
        },
        "seeds": seed_runs,
    }


def run_models(design: Design, smoke: bool) -> dict:
    n_zones = len(design.zone_ids)
    train_volume = design.counts[design.train_mask].sum(axis=0)
    top_interaction_zones = np.argsort(train_volume)[-20:]
    diagnostic_zones = np.argsort(train_volume)[-12:]
    results = {
        "contract": {
            "treatment": "model family",
            "primary_metric": "test log_scaled_rmse (lower is better)",
            "secondary_metrics": ["WAPE", "raw MAE", "cross-region alignment penalty"],
            "split": "Jan-Sep train, Oct validation, Nov-Dec test",
            "lags_half_hours": list(LAGS),
            "seeds": list(SEEDS if not smoke else (0,)),
            "synergy_caution": "Joint-input gain and shuffle penalty are predictive proxies, not formal synergy estimators.",
        },
        "models": {},
    }

    # Baselines use exactly the same target timestamps and split.
    for name, lag in [("Persistence", 1), ("Daily seasonal", 48), ("Weekly seasonal", 336)]:
        lag_i = LAGS.index(lag)
        pred_z = design.x[:, lag_i * n_zones : (lag_i + 1) * n_zones]
        results["models"][name] = {
            "test": metrics(pred_z, design, design.test_mask),
            "cross_region_penalty": 0.0,
            "deterministic": True,
        }

    local_pred, local_info = fit_local_ridge(design)
    results["models"]["Local Ridge"] = {
        "test": metrics(local_pred, design, design.test_mask),
        "cross_region_penalty": 0.0,
        "fit": local_info,
        "deterministic": True,
    }

    global_ridge, global_info = fit_best_ridge(
        design.x[design.train_mask],
        design.y[design.train_mask],
        design.x[design.valid_mask],
        design.y[design.valid_mask],
    )
    global_pred = global_ridge.predict(design.x)
    results["models"]["Global Ridge"] = {
        "test": metrics(global_pred, design, design.test_mask),
        "cross_region_penalty": cross_region_penalty(
            global_ridge, design, global_pred, diagnostic_zones
        ),
        "fit": global_info,
        "deterministic": True,
    }

    builder = lambda x: interaction_features(x, n_zones, top_interaction_zones)
    x_interaction = builder(design.x)
    interaction_ridge, interaction_info = fit_best_ridge(
        x_interaction[design.train_mask],
        design.y[design.train_mask],
        x_interaction[design.valid_mask],
        design.y[design.valid_mask],
    )
    interaction_pred = interaction_ridge.predict(x_interaction)
    interaction_pairs = []
    for left in range(len(top_interaction_zones)):
        for right in range(left + 1, len(top_interaction_zones)):
            interaction_pairs.append((top_interaction_zones[left], top_interaction_zones[right]))
    interaction_coef = interaction_ridge.coef_[:, design.x.shape[1] :]
    interaction_strength = np.mean(np.abs(interaction_coef), axis=0)
    top_pair_indices = np.argsort(interaction_strength)[-15:][::-1]
    top_pairs = []
    for pair_i in top_pair_indices:
        left, right = interaction_pairs[int(pair_i)]
        target = int(np.argmax(np.abs(interaction_coef[:, pair_i])))
        top_pairs.append(
            {
                "source_zone_pair": [int(design.zone_ids[left]), int(design.zone_ids[right])],
                "mean_abs_coefficient": float(interaction_strength[pair_i]),
                "largest_target_zone": int(design.zone_ids[target]),
                "largest_target_coefficient": float(interaction_coef[target, pair_i]),
            }
        )
    results["models"]["Interaction Ridge"] = {
        "test": metrics(interaction_pred, design, design.test_mask),
        "cross_region_penalty": cross_region_penalty(
            interaction_ridge,
            design,
            interaction_pred,
            diagnostic_zones,
            feature_builder=builder,
        ),
        "fit": {
            **interaction_info,
            "interaction_zones": design.zone_ids[top_interaction_zones].tolist(),
            "top_interaction_pairs": top_pairs,
        },
        "deterministic": True,
    }

    seeds = (0,) if smoke else SEEDS
    extra_runs = []
    mlp_runs = []
    for seed in seeds:
        tic = time.time()
        extra = ExtraTreesRegressor(
            n_estimators=40 if smoke else 120,
            min_samples_leaf=2,
            max_features=0.5,
            n_jobs=-1,
            random_state=seed,
        )
        extra.fit(design.x[design.train_mask], design.y[design.train_mask])
        extra_pred = extra.predict(design.x)
        extra_runs.append(
            {
                "seed": seed,
                "test": metrics(extra_pred, design, design.test_mask),
                "cross_region_penalty": cross_region_penalty(
                    extra, design, extra_pred, diagnostic_zones
                ),
                "fit_seconds": time.time() - tic,
            }
        )
        print(f"Extra Trees seed={seed} seconds={time.time() - tic:.1f}", flush=True)

        tic = time.time()
        mlp = MLPRegressor(
            hidden_layer_sizes=(64,) if smoke else (128, 64),
            activation="relu",
            solver="adam",
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=30 if smoke else 80,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=seed,
        )
        mlp.fit(design.x[design.train_mask], design.y[design.train_mask])
        mlp_pred = mlp.predict(design.x)
        mlp_runs.append(
            {
                "seed": seed,
                "test": metrics(mlp_pred, design, design.test_mask),
                "cross_region_penalty": cross_region_penalty(mlp, design, mlp_pred, diagnostic_zones),
                "iterations": int(mlp.n_iter_),
                "fit_seconds": time.time() - tic,
            }
        )
        print(f"MLP seed={seed} seconds={time.time() - tic:.1f}", flush=True)

    results["models"]["Extra Trees"] = summarize_seed_runs(extra_runs)
    results["models"]["MLP"] = summarize_seed_runs(mlp_runs)
    return results


def scalar_metric(model_record: dict, key: str) -> tuple[float, float]:
    value = model_record["test"][key]
    if isinstance(value, dict):
        return value["mean"], value["sd"]
    return float(value), 0.0


def scalar_penalty(model_record: dict) -> tuple[float, float]:
    value = model_record["cross_region_penalty"]
    if isinstance(value, dict):
        return value["mean"], value["sd"]
    return float(value), 0.0


def plot_results(results: dict, stem: Path):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    names = list(results["models"])
    palette = {
        "Persistence": "#B7BDC8",
        "Daily seasonal": "#8E99A8",
        "Weekly seasonal": "#5C677D",
        "Local Ridge": "#1F3B73",
        "Global Ridge": "#3567A8",
        "Interaction Ridge": "#2A9D8F",
        "Extra Trees": "#C56B3C",
        "MLP": "#A23E48",
    }
    colors = [palette[name] for name in names]
    rmse = np.array([scalar_metric(results["models"][name], "log_scaled_rmse")[0] for name in names])
    rmse_sd = np.array([scalar_metric(results["models"][name], "log_scaled_rmse")[1] for name in names])
    wape = np.array([scalar_metric(results["models"][name], "wape")[0] for name in names])
    wape_sd = np.array([scalar_metric(results["models"][name], "wape")[1] for name in names])
    penalty = np.array([scalar_penalty(results["models"][name])[0] for name in names]) * 100

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    y = np.arange(len(names))
    axes[0].errorbar(rmse, y, xerr=rmse_sd, fmt="none", ecolor="#6B7280", lw=0.8, capsize=2)
    axes[0].scatter(rmse, y, c=colors, s=25, zorder=3)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Test log-scaled RMSE ↓")

    axes[1].errorbar(wape, y, xerr=wape_sd, fmt="none", ecolor="#6B7280", lw=0.8, capsize=2)
    axes[1].scatter(wape, y, c=colors, s=25, zorder=3)
    axes[1].set_yticks(y, [""] * len(names))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Test WAPE ↓")

    diagnostic_names = ["Local Ridge", "Global Ridge", "Interaction Ridge", "Extra Trees", "MLP"]
    label_offsets = {
        "Local Ridge": (4, 4),
        "Global Ridge": (4, 9),
        "Interaction Ridge": (4, -11),
        "Extra Trees": (4, 4),
        "MLP": (4, 4),
    }
    for name in diagnostic_names:
        i = names.index(name)
        axes[2].scatter(penalty[i], rmse[i], color=colors[i], s=28)
        axes[2].annotate(
            name,
            (penalty[i], rmse[i]),
            xytext=label_offsets[name],
            textcoords="offset points",
            fontsize=5.8,
        )
    axes[2].set_xscale("symlog", linthresh=1)
    axes[2].set_xlabel("Cross-region shuffle penalty (%) ↑")
    axes[2].set_ylabel("Test log-scaled RMSE ↓")
    axes[2].axvline(0, color="#B7BDC8", lw=0.7, ls="--")
    axes[2].margins(x=0.08, y=0.18)

    for label, ax in zip("abc", axes):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=8, fontweight="bold")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a 90-day, one-seed smoke test.")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    cache = DATA_CACHE
    if args.smoke:
        cache = ROOT / "data" / "nyc_taxi_yellow_2023_q1_30min_manhattan_smoke.npz"
    if args.rebuild_cache and cache.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing cache automatically: {cache}. Move it aside explicitly first."
        )

    months = (1, 2, 3) if args.smoke else tuple(range(1, 13))
    counts, zone_ids, metadata = aggregate_counts(months, cache)
    design = make_design(counts, zone_ids, args.smoke)
    results = run_models(design, args.smoke)
    results["data"] = metadata
    results["data"]["train_samples"] = int(design.train_mask.sum())
    results["data"]["valid_samples"] = int(design.valid_mask.sum())
    results["data"]["test_samples"] = int(design.test_mask.sum())

    output = RESULTS_JSON if not args.smoke else RESULTS_JSON.with_name("nyc_taxi_synergy_model_screen_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    stem = FIGURE_STEM if not args.smoke else FIGURE_STEM.with_name(FIGURE_STEM.name + "_smoke")
    plot_results(results, stem)
    print(f"wrote {output}")
    print(f"wrote {stem}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
