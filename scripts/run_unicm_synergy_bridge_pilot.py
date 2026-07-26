#!/usr/bin/env python3
"""Pilot a five-mode SynergyBridge on frozen UniCM mode forecasts.

The pilot trains only an interaction-only residual adapter on ORAS5 1958--1979,
selects the epoch on a chronology-held validation tail, and evaluates once on
ORAS5 1980--2014. It does not fine-tune UniCM and does not claim physical-field
improvement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNICM_SRC = ROOT / "data" / "UniCM-checkpoint" / "src"
DEFAULT_CHECKPOINT_ROOT = UNICM_SRC / "experiments"
DEFAULT_OUTPUT = ROOT / "results" / "unicm_synergy_bridge_pilot"

MODE_NAMES = (
    "ENSO",
    "NPMM",
    "SPMM",
    "IOB",
    "IOD",
    "SIOD",
    "TNA",
    "nino12",
    "nino3",
    "nino4",
    "WWV",
)
MODULE_INDICES = (0, 7, 8, 9, 4)
MODULE_NAMES = tuple(MODE_NAMES[index] for index in MODULE_INDICES)


def install_import_shims() -> None:
    if "pynvml" not in sys.modules:
        module = types.ModuleType("pynvml")
        module.nvmlInit = lambda: None
        module.nvmlShutdown = lambda: None
        module.nvmlDeviceGetCount = lambda: 0
        module.nvmlDeviceGetHandleByIndex = lambda index: None
        module.nvmlDeviceGetMemoryInfo = lambda handle: SimpleNamespace(
            total=0, used=0, free=0
        )
        sys.modules["pynvml"] = module


def import_unicm():
    if str(UNICM_SRC) not in sys.path:
        sys.path.insert(0, str(UNICM_SRC))
    install_import_shims()
    import torch
    from LoadData import make_test_data_ORAS5, make_train_data_ORAS5
    from models import UniCM
    from unicm_synergy_bridge import SynergyBridge

    return torch, make_train_data_ORAS5, make_test_data_ORAS5, UniCM, SynergyBridge


def choose_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_args(data_root: Path, device) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=str(data_root),
        his_len=12,
        pred_len=24,
        input_channal=5,
        resolution=5,
        norm_std=1,
        t20d_mode=1,
        mode_interaction="1",
        patch_size=[2, 2],
        emb_spatial_size=216,
        d_size=256,
        nheads=4,
        dim_feedforward=512,
        dropout=0.1,
        num_encoder_layers=4,
        num_decoder_layers=4,
        autoregressive=0,
        device=device,
    )


def resolve_checkpoint(checkpoint: Path | None, seed: int) -> Path:
    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved
    matches = sorted(
        DEFAULT_CHECKPOINT_ROOT.glob(
            f"*Seed{int(seed)}/model_save/model_best.pkl"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one seed-{seed} checkpoint under {DEFAULT_CHECKPOINT_ROOT}; "
            f"found {len(matches)}."
        )
    return matches[0]


def validate_data_root(data_root: Path) -> Path:
    expected = data_root / "ORAS5" / "ORAS5_1958_2014.nc"
    if not expected.is_file():
        raise FileNotFoundError(
            "SynergyBridge pilot needs the UniCM ORAS5 file:\n"
            f"  {expected}\n"
            "Pass its parent dataset directory with --data-root."
        )
    return expected


def cache_tag(checkpoint: Path, data_file: Path) -> str:
    payload = "|".join(
        (
            str(checkpoint.resolve()),
            str(checkpoint.stat().st_size),
            str(checkpoint.stat().st_mtime_ns),
            str(data_file.resolve()),
            str(data_file.stat().st_size),
            str(data_file.stat().st_mtime_ns),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mode_geometry(args) -> None:
    """Set the ORAS5 mode boxes without opening the data a second time."""

    reso = args.resolution
    lat0 = 45 // reso
    args.val_nino_relative = (70 // reso - lat0, 80 // reso - lat0, 190 // reso, 240 // reso)
    args.val_NPMM_relative = (85 // reso - lat0, 100 // reso - lat0, 200 // reso, 240 // reso)
    args.val_SPMM_relative = (50 // reso - lat0, 60 // reso - lat0, 250 // reso, 270 // reso)
    args.val_IOB_relative = (55 // reso - lat0, 75 // reso - lat0, 40 // reso, 100 // reso)
    args.val_IOD_relative = (
        (65 // reso - lat0, 85 // reso - lat0, 50 // reso, 70 // reso),
        (65 // reso - lat0, 75 // reso - lat0, 90 // reso, 110 // reso),
    )
    args.val_SIOD_relative = (
        (50 // reso - lat0, 65 // reso - lat0, 65 // reso, 80 // reso),
        (45 // reso - lat0, 65 // reso - lat0, 90 // reso, 120 // reso),
    )
    args.val_TNA_relative = (80 // reso - lat0, 100 // reso - lat0, 305 // reso, 345 // reso)
    args.val_nino12 = (65 // reso - lat0, 75 // reso - lat0, 270 // reso, 280 // reso)
    args.val_nino3 = (70 // reso - lat0, 80 // reso - lat0, 210 // reso, 270 // reso)
    args.val_nino4 = (70 // reso - lat0, 80 // reso - lat0, 200 // reso, 210 // reso)
    args.WWV = (70 // reso - lat0, 80 // reso - lat0, 120 // reso, 280 // reso)
    args.val_relative = [
        args.val_nino_relative,
        args.val_NPMM_relative,
        args.val_SPMM_relative,
        args.val_IOB_relative,
        args.val_IOD_relative,
        args.val_SIOD_relative,
        args.val_TNA_relative,
        args.val_nino12,
        args.val_nino3,
        args.val_nino4,
    ]


def region_mean(field, region):
    return field[..., region[0] : region[1], region[2] : region[3]].mean(dim=(-2, -1))


def extract_modes(torch, values, args):
    """Extract 10 SST modes plus WWV from [B,T,C,H,W] normalized fields."""

    sst = values[:, :, 0]
    outputs = []
    for index, region in enumerate(args.val_relative):
        if index in (4, 5):
            outputs.append(region_mean(sst, region[0]) - region_mean(sst, region[1]))
        else:
            outputs.append(region_mean(sst, region))
    outputs.append(region_mean(values[:, :, -1], args.WWV))
    return torch.stack(outputs, dim=1)


def predict_mode_head(torch, model, values, timestamps, modes):
    predictor_mode = modes.permute(0, 2, 1).unsqueeze(-1).unsqueeze(2)
    prediction, _, _ = model.forward_sep(
        predictor_mode,
        timestamps,
        model.encoder_mode,
        model.decoder_mode,
        model.linear_output_mode,
        model.predictor_emb_mode,
        model.predictand_emb_mode,
        1,
        [1, 1],
        train=False,
    )
    return prediction.squeeze(-1).squeeze(2).permute(0, 2, 1)


def cache_arrays(
    *,
    torch,
    split_name: str,
    arrays,
    model,
    args,
    device,
    batch_size: int,
    cache_path: Path,
    checkpoint: Path,
):
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files if key != "metadata"}

    values, timestamps, std = arrays[:3]
    dataset = torch.utils.data.TensorDataset(values, timestamps, std)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    histories, targets, base_forecasts, scalers = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch_values, batch_timestamps, batch_std in loader:
            batch_values = batch_values.to(device=device, dtype=torch.float32)
            batch_timestamps = batch_timestamps.to(device=device, dtype=torch.int64)
            modes = extract_modes(torch, batch_values, args)
            base = predict_mode_head(
                torch, model, batch_values, batch_timestamps, modes
            )
            histories.append(modes[:, :, : args.his_len].cpu().numpy())
            targets.append(modes[:, :, -args.pred_len :].cpu().numpy())
            base_forecasts.append(base.cpu().numpy())
            scalers.append(batch_std.reshape(batch_std.shape[0], -1)[:, 0].numpy())

    result = {
        "history": np.concatenate(histories).astype(np.float32),
        "target": np.concatenate(targets).astype(np.float32),
        "base_forecast": np.concatenate(base_forecasts).astype(np.float32),
        "sst_std": np.concatenate(scalers).astype(np.float32),
    }
    metadata = json.dumps(
        {
            "split": split_name,
            "checkpoint": str(checkpoint),
            "history_len": args.his_len,
            "prediction_len": args.pred_len,
            "mode_names": MODE_NAMES,
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result, metadata=np.array(metadata))
    return result


def chronological_inner_split(n_samples: int, sequence_len: int):
    validation_count = max(24, int(round(0.2 * n_samples)))
    validation_start = n_samples - validation_count
    fit_stop = validation_start - (sequence_len - 1)
    if fit_stop < 48:
        raise ValueError(
            f"Only {n_samples} early-period samples; not enough for a gapped split."
        )
    return np.arange(fit_stop), np.arange(validation_start, n_samples)


def standardize_history(train_history, *others):
    mean = train_history.mean(axis=0, keepdims=True)
    scale = train_history.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (mean, scale), tuple((array - mean) / scale for array in (train_history, *others))


def weighted_loss(torch, prediction, target):
    target_index = torch.as_tensor(MODULE_INDICES, device=prediction.device)
    error = prediction.index_select(1, target_index) - target.index_select(1, target_index)
    weights = torch.ones((1, 1, 24), device=prediction.device)
    weights[:, :, 6:10] = 2.0
    return (error.square() * weights).mean()


def train_bridge(
    *,
    torch,
    bridge,
    train_cache,
    device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    seed: int,
):
    fit_index, validation_index = chronological_inner_split(
        len(train_cache["history"]), 36
    )
    (history_mean, history_scale), standardized = standardize_history(
        train_cache["history"][fit_index],
        train_cache["history"][validation_index],
    )
    fit_history, validation_history = standardized

    def tensor(array):
        return torch.as_tensor(array, device=device, dtype=torch.float32)

    fit_dataset = torch.utils.data.TensorDataset(
        tensor(fit_history),
        tensor(train_cache["base_forecast"][fit_index]),
        tensor(train_cache["target"][fit_index]),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    fit_loader = torch.utils.data.DataLoader(
        fit_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_history = tensor(validation_history)
    val_base = tensor(train_cache["base_forecast"][validation_index])
    val_target = tensor(train_cache["target"][validation_index])

    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_state, best_loss, best_epoch = None, math.inf, -1
    history_log = []
    stale = 0
    for epoch in range(epochs):
        bridge.train()
        losses = []
        for batch_history, batch_base, batch_target in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(
                torch,
                bridge(batch_base, batch_history),
                batch_target,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        bridge.eval()
        with torch.no_grad():
            val_loss = float(
                weighted_loss(
                    torch,
                    bridge(val_base, val_history),
                    val_target,
                ).cpu()
            )
        history_log.append(
            {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        )
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(bridge.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("Bridge training did not produce a finite validation loss.")
    bridge.load_state_dict(best_state)
    return {
        "history_mean": history_mean.astype(np.float32),
        "history_scale": history_scale.astype(np.float32),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_log": history_log,
        "fit_samples": int(len(fit_index)),
        "validation_samples": int(len(validation_index)),
        "gap_samples": int(validation_index[0] - fit_index[-1] - 1),
    }


def predict_bridge(torch, bridge, cache, history_mean, history_scale, device, batch_size):
    history = ((cache["history"] - history_mean) / history_scale).astype(np.float32)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(history),
        torch.from_numpy(cache["base_forecast"]),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = []
    bridge.eval()
    with torch.no_grad():
        for batch_history, batch_base in loader:
            outputs.append(
                bridge(
                    batch_base.to(device),
                    batch_history.to(device),
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs).astype(np.float32)


def rmse_by_target_lead(forecast, target, scaler):
    error = (forecast - target) * scaler[:, None, None]
    return np.sqrt(np.mean(np.square(error), axis=0))


def correlation_by_target_lead(forecast, target):
    forecast_centered = forecast - forecast.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    numerator = np.sum(forecast_centered * target_centered, axis=0)
    denominator = np.sqrt(
        np.sum(np.square(forecast_centered), axis=0)
        * np.sum(np.square(target_centered), axis=0)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


def circular_block_indices(rng, n_samples: int, block_length: int):
    n_blocks = int(math.ceil(n_samples / block_length))
    starts = rng.integers(0, n_samples, size=n_blocks)
    blocks = [
        (start + np.arange(block_length, dtype=int)) % n_samples for start in starts
    ]
    return np.concatenate(blocks)[:n_samples]


def primary_rmse_gain(base, bridge, target, scaler):
    selected = np.asarray(MODULE_INDICES)
    lead_slice = slice(6, 10)
    scale = scaler[:, None, None]
    base_rmse = np.sqrt(
        np.mean(
            ((base[:, selected, lead_slice] - target[:, selected, lead_slice]) * scale)
            ** 2
        )
    )
    bridge_rmse = np.sqrt(
        np.mean(
            ((bridge[:, selected, lead_slice] - target[:, selected, lead_slice]) * scale)
            ** 2
        )
    )
    return float(base_rmse - bridge_rmse), float(base_rmse), float(bridge_rmse)


def block_bootstrap_gain(
    base,
    bridge,
    target,
    scaler,
    *,
    n_bootstrap: int,
    block_length: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    gains = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = circular_block_indices(rng, len(target), block_length)
        gains[index] = primary_rmse_gain(
            base[sample],
            bridge[sample],
            target[sample],
            scaler[sample],
        )[0]
    return gains


def make_metrics(test_cache, bridge_forecast, bootstrap_gains):
    base = test_cache["base_forecast"]
    target = test_cache["target"]
    scaler = test_cache["sst_std"]
    base_rmse = rmse_by_target_lead(base, target, scaler)
    bridge_rmse = rmse_by_target_lead(bridge_forecast, target, scaler)
    base_correlation = correlation_by_target_lead(base, target)
    bridge_correlation = correlation_by_target_lead(bridge_forecast, target)
    gain, primary_base, primary_bridge = primary_rmse_gain(
        base, bridge_forecast, target, scaler
    )
    selected = np.asarray(MODULE_INDICES)
    return {
        "primary": {
            "metric": "pooled RMSE across five target modes and leads 7-10",
            "unit": "degC",
            "baseline": primary_base,
            "synergy_bridge": primary_bridge,
            "absolute_gain": gain,
            "relative_gain_percent": 100.0 * gain / primary_base,
            "bootstrap_ci95": np.percentile(bootstrap_gains, [2.5, 97.5]).tolist(),
            "bootstrap_one_sided_p": float(
                (1 + np.count_nonzero(bootstrap_gains <= 0))
                / (len(bootstrap_gains) + 1)
            ),
        },
        "secondary": {
            "all_lead_module_rmse_baseline": float(base_rmse[selected].mean()),
            "all_lead_module_rmse_synergy_bridge": float(
                bridge_rmse[selected].mean()
            ),
            "lead7_10_mean_correlation_baseline": float(
                np.nanmean(base_correlation[selected, 6:10])
            ),
            "lead7_10_mean_correlation_synergy_bridge": float(
                np.nanmean(bridge_correlation[selected, 6:10])
            ),
        },
        "arrays": {
            "base_rmse": base_rmse,
            "bridge_rmse": bridge_rmse,
            "rmse_gain": base_rmse - bridge_rmse,
            "base_correlation": base_correlation,
            "bridge_correlation": bridge_correlation,
        },
    }


def plot_results(metrics, bootstrap_gains, output_dir: Path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    arrays = metrics["arrays"]
    selected = np.asarray(MODULE_INDICES)
    leads = np.arange(1, 25)
    base_curve = arrays["base_rmse"][selected].mean(axis=0)
    bridge_curve = arrays["bridge_rmse"][selected].mean(axis=0)
    heatmap = arrays["rmse_gain"][selected]
    limit = max(float(np.nanmax(np.abs(heatmap))), 1e-4)

    fig = plt.figure(figsize=(7.2, 3.25), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.35, 0.9))
    ax_curve = fig.add_subplot(grid[0, 0])
    ax_heatmap = fig.add_subplot(grid[0, 1])
    ax_bootstrap = fig.add_subplot(grid[0, 2])

    ax_curve.plot(leads, base_curve, color="#6F7782", lw=1.5, label="Frozen UniCM")
    ax_curve.plot(
        leads,
        bridge_curve,
        color="#D55E00",
        lw=1.7,
        label="SynergyBridge",
    )
    ax_curve.axvspan(6.5, 10.5, color="#E69F00", alpha=0.10, lw=0)
    ax_curve.set(xlabel="Prediction lead (months)", ylabel="Mean RMSE (°C)", xlim=(1, 24))
    ax_curve.set_xticks([1, 6, 10, 14, 18, 24])
    ax_curve.spines[["top", "right"]].set_visible(False)
    ax_curve.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=1.8,
    )

    image = ax_heatmap.imshow(
        heatmap,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax_heatmap.set(
        xlabel="Prediction lead (months)",
        ylabel="Target mode",
        yticks=np.arange(len(MODULE_NAMES)),
        yticklabels=MODULE_NAMES,
    )
    ax_heatmap.set_xticks([0, 5, 9, 13, 17, 23], [1, 6, 10, 14, 18, 24])
    ax_heatmap.axvline(5.5, color="0.25", lw=0.6, ls=":")
    ax_heatmap.axvline(9.5, color="0.25", lw=0.6, ls=":")
    colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.05, pad=0.03)
    colorbar.set_label("RMSE gain (°C)")

    ax_bootstrap.hist(
        bootstrap_gains,
        bins=28,
        color="#D55E00",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.3,
    )
    observed = metrics["primary"]["absolute_gain"]
    ax_bootstrap.axvline(0, color="0.25", lw=0.8, ls="--")
    ax_bootstrap.axvline(observed, color="#0072B2", lw=1.4)
    ax_bootstrap.set(xlabel="Lead 7–10 RMSE gain (°C)", ylabel="Bootstrap count")
    ax_bootstrap.spines[["top", "right"]].set_visible(False)

    for label, axis in zip("abc", (ax_curve, ax_heatmap, ax_bootstrap)):
        axis.text(
            -0.16,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
    ):
        fig.savefig(
            output_dir / f"unicm_synergy_bridge_pilot.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def smoke_test(torch, SynergyBridge, device) -> None:
    bridge = SynergyBridge().to(device)
    base = torch.randn(7, 11, 24, device=device)
    history = torch.randn(7, 11, 12, device=device)
    initial = bridge(base, history)
    if not torch.equal(initial, base):
        raise AssertionError("Zero-initialized bridge changed the frozen forecast.")
    target = base.clone()
    target[:, MODULE_INDICES, :] += 0.25
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-2)
    optimizer.zero_grad()
    loss = weighted_loss(torch, bridge(base, history), target)
    loss.backward()
    optimizer.step()
    updated = bridge(base, history)
    untouched = [index for index in range(11) if index not in MODULE_INDICES]
    if not torch.equal(updated[:, untouched], base[:, untouched]):
        raise AssertionError("Bridge changed a target outside the fixed module.")
    if torch.equal(updated[:, MODULE_INDICES], base[:, MODULE_INDICES]):
        raise AssertionError("Bridge failed to learn a nonzero correction.")
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "zero_initialization": "exact",
                "target_module": MODULE_NAMES,
                "trainable_parameters": bridge.trainable_parameter_count,
                "device": str(device),
            },
            indent=2,
        )
    )


def run(args) -> int:
    torch, MakeTrain, MakeTest, UniCM, SynergyBridge = import_unicm()
    device = choose_device(torch, args.device)
    set_seed(torch, args.seed)
    if args.smoke_test:
        smoke_test(torch, SynergyBridge, device)
        return 0

    data_root = args.data_root.expanduser().resolve()
    data_file = validate_data_root(data_root)
    checkpoint = resolve_checkpoint(args.checkpoint, args.checkpoint_seed)
    artifact_tag = cache_tag(checkpoint, data_file)
    model_args = make_args(data_root, device)
    mode_geometry(model_args)

    print(f"device={device}")
    print(f"checkpoint={checkpoint}")
    print("loading ORAS5 early/late chronological splits")
    train_arrays = MakeTrain(model_args).dataloader_seq()
    test_arrays = MakeTest(model_args).dataloader_seq()

    model = UniCM(model_args).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = output_dir / "cache"
    train_cache = cache_arrays(
        torch=torch,
        split_name="ORAS5_1958_1979",
        arrays=train_arrays,
        model=model,
        args=model_args,
        device=device,
        batch_size=args.inference_batch_size,
        cache_path=cache_dir
        / f"mode_forecasts_seed{args.checkpoint_seed}_train_{artifact_tag}.npz",
        checkpoint=checkpoint,
    )
    test_cache = cache_arrays(
        torch=torch,
        split_name="ORAS5_1980_2014",
        arrays=test_arrays,
        model=model,
        args=model_args,
        device=device,
        batch_size=args.inference_batch_size,
        cache_path=cache_dir
        / f"mode_forecasts_seed{args.checkpoint_seed}_test_{artifact_tag}.npz",
        checkpoint=checkpoint,
    )
    del model, train_arrays, test_arrays

    bridge = SynergyBridge(
        rank=args.rank,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    training = train_bridge(
        torch=torch,
        bridge=bridge,
        train_cache=train_cache,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.bridge_batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    bridge_forecast = predict_bridge(
        torch,
        bridge,
        test_cache,
        training["history_mean"],
        training["history_scale"],
        device,
        args.bridge_batch_size,
    )
    bootstrap = block_bootstrap_gain(
        test_cache["base_forecast"],
        bridge_forecast,
        test_cache["target"],
        test_cache["sst_std"],
        n_bootstrap=args.bootstrap,
        block_length=args.bootstrap_block,
        seed=args.seed + 1000,
    )
    metrics = make_metrics(test_cache, bridge_forecast, bootstrap)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": bridge.state_dict(),
            "history_mean": training["history_mean"],
            "history_scale": training["history_scale"],
            "module_indices": MODULE_INDICES,
            "module_names": MODULE_NAMES,
            "checkpoint": str(checkpoint),
        },
        output_dir / f"synergy_bridge_seed{args.checkpoint_seed}.pt",
    )
    np.savez_compressed(
        output_dir / "evaluation_arrays.npz",
        bridge_forecast=bridge_forecast,
        bootstrap_rmse_gain=bootstrap.astype(np.float32),
        **metrics["arrays"],
    )
    report = {
        "status": "completed",
        "scope": "mode-head forecast correction; frozen UniCM; no physical-field claim",
        "checkpoint": str(checkpoint),
        "checkpoint_seed": args.checkpoint_seed,
        "target_module": MODULE_NAMES,
        "source_module": MODULE_NAMES,
        "train_period": "ORAS5 1958-1979",
        "test_period": "ORAS5 1980-2014",
        "train_samples": len(train_cache["history"]),
        "test_samples": len(test_cache["history"]),
        "bridge_trainable_parameters": bridge.trainable_parameter_count,
        "training": training,
        "metrics": {key: value for key, value in metrics.items() if key != "arrays"},
        "bootstrap": {
            "replicates": args.bootstrap,
            "circular_block_length_months": args.bootstrap_block,
        },
        "deferred_controls": [
            "linear calibration of the frozen forecast",
            "five-mode additive adapter",
            "random five-mode adapter with matched parameter count",
            "all eleven modes with matched parameter count",
            "shuffled lead gate",
            "additional checkpoint seeds",
            "physical-field bias injection",
        ],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_results(metrics, bootstrap, output_dir)
    print(json.dumps(json_ready(report["metrics"]), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "UniCM-checkpoint" / "dataset",
        help="Directory containing ORAS5/ORAS5_1958_2014.nc.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--bridge-batch-size", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-block", type=int, default=36)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (FileNotFoundError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(2)
