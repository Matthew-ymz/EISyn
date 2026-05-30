from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_o3_h1_tm_peid import (
    build_intervention_samples,
    estimate_continuous_source_set_information,
    summarize_seed_runs,
)


def test_continuous_source_set_information_detects_joint_signal_gain() -> None:
    rng = np.random.default_rng(7)
    sources = rng.normal(size=(3500, 2))
    target = sources[:, 0] + sources[:, 1] + rng.normal(scale=0.18, size=sources.shape[0])

    summary = estimate_continuous_source_set_information(
        sources,
        target,
        source_names=["NOx", "VOC"],
    )

    assert summary["joint_mi_nats"] > max(summary["individual_mi_nats"].values())
    assert summary["gain_over_best_individual_nats"] > 0.25


def test_build_intervention_samples_uses_manifest_source_labels() -> None:
    frame = pd.DataFrame(
        {
            "meic_NOx": np.geomspace(0.5, 8.0, 24),
            "meic_VOC": np.geomspace(0.3, 18.0, 24),
            "temp_c": np.linspace(20.0, 36.0, 24),
            "RH": np.linspace(35.0, 90.0, 24),
            "msdwswrf": np.linspace(80.0, 260.0, 24),
        }
    )

    samples = build_intervention_samples(
        frame,
        source_set=["NOx", "VOC", "Temp"],
        sample_count=128,
        rng=np.random.default_rng(11),
    )

    assert list(samples.columns) == ["NOx", "VOC", "Temp"]
    assert samples.shape == (128, 3)
    assert samples["NOx"].between(frame["meic_NOx"].quantile(0.05), frame["meic_NOx"].quantile(0.95)).all()
    assert samples["VOC"].between(frame["meic_VOC"].quantile(0.05), frame["meic_VOC"].quantile(0.95)).all()
    assert samples["Temp"].between(frame["temp_c"].quantile(0.05), frame["temp_c"].quantile(0.95)).all()


def test_summarize_seed_runs_ranks_primary_source_set_first() -> None:
    runs = pd.DataFrame(
        {
            "sources": ["NOx+VOC", "NOx+VOC", "Temp+RH", "Temp+RH"],
            "joint_mi_nats": [1.0, 1.2, 0.2, 0.3],
            "gain_over_best_individual_nats": [0.6, 0.8, 0.01, 0.03],
            "net_synergy_nats": [0.3, 0.5, -0.02, 0.0],
        }
    )

    summary = summarize_seed_runs(runs)

    assert summary.iloc[0]["sources"] == "NOx+VOC"
    assert float(summary.iloc[0]["gain_over_best_individual_mean_nats"]) == 0.7
