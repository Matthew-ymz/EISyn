from __future__ import annotations

import sys
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sis_next_state_peid_alignment import (
    SisAlignmentConfig,
    build_training_pairs,
    fit_probabilistic_mlp,
    run_alignment,
    run_experiment,
    simulate_sis_transition,
)


def test_stochastic_transition_uses_tau_and_is_seed_reproducible() -> None:
    config = SisAlignmentConfig(tau=1.0, process_noise=0.05)
    states = np.array([[0.2, 0.4, 0.6], [0.7, 0.3, 0.5]])

    first = simulate_sis_transition(states, config=config, seed=17)
    second = simulate_sis_transition(states, config=config, seed=17)
    changed_seed = simulate_sis_transition(states, config=config, seed=18)

    assert config.integration_steps == 50
    assert first.shape == states.shape
    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed_seed)
    assert np.min(first) >= 0.0
    assert np.max(first) <= 1.0


def test_training_pairs_mix_natural_and_intervention_states_equally() -> None:
    config = SisAlignmentConfig(
        tau=1.0,
        process_noise=0.05,
        training_samples_per_source=24,
        transition_replicates=3,
        warmup_steps=20,
    )

    dataset = build_training_pairs(config=config, seed=9)

    assert dataset.inputs.shape == dataset.targets.shape == (144, 3)
    assert dataset.source_labels.shape == (144,)
    assert np.sum(dataset.source_labels == "natural") == 72
    assert np.sum(dataset.source_labels == "intervention") == 72
    assert dataset.target_names == ("w_tau", "x_tau", "y_tau")
    assert np.isfinite(dataset.inputs).all()
    assert np.isfinite(dataset.targets).all()
    assert np.min(dataset.targets) >= 0.0
    assert np.max(dataset.targets) <= 1.0


def test_probabilistic_mlp_returns_reproducible_conditional_samples() -> None:
    config = SisAlignmentConfig(
        training_samples_per_source=48,
        transition_replicates=2,
        warmup_steps=20,
    )
    dataset = build_training_pairs(config=config, seed=12)

    model = fit_probabilistic_mlp(dataset.inputs, dataset.targets, seed=31, epochs=40)
    query = dataset.inputs[:7]
    mean, std = model.predict_distribution(query)
    first = model.sample(query, seed=44)
    second = model.sample(query, seed=44)

    assert mean.shape == std.shape == first.shape == (7, 3)
    assert np.all(std > 0.0)
    assert np.array_equal(first, second)
    assert np.min(first) >= 0.0
    assert np.max(first) <= 1.0
    assert np.isfinite(model.test_nll)


def test_alignment_summary_contains_protocol_and_relation_errors() -> None:
    config = SisAlignmentConfig(
        tau=1.0,
        process_noise=0.05,
        training_samples_per_source=64,
        transition_replicates=2,
        warmup_steps=20,
    )

    summary = run_alignment(
        config=config,
        seed=5,
        estimator="histogram",
        peid_samples=500,
        epochs=50,
    )

    assert summary["protocol"]["tau"] == 1.0
    assert summary["protocol"]["integration_steps"] == 50
    assert summary["protocol"]["process_noise"] == 0.05
    assert summary["protocol"]["noise_location"] == "sis_dynamics"
    assert summary["protocol"]["target_names"] == ["w_tau", "x_tau", "y_tau"]
    assert set(summary["relations"]) == {"w+x->x_tau", "w+y->y_tau"}
    for relation in summary["relations"].values():
        assert np.isfinite(relation["oracle_synergy"])
        assert np.isfinite(relation["mlp_synergy"])
        assert relation["relative_error"] >= 0.0
    assert isinstance(summary["passed"], bool)


def test_experiment_writes_json_png_and_markdown(tmp_path: Path) -> None:
    result = run_experiment(
        config=SisAlignmentConfig(
            training_samples_per_source=48,
            transition_replicates=2,
            warmup_steps=20,
        ),
        seed=7,
        estimator="histogram",
        peid_samples=400,
        epochs=40,
        result_path=tmp_path / "result.json",
        figure_path=tmp_path / "comparison.png",
        report_path=tmp_path / "report.md",
    )

    assert Path(result["result_path"]).exists()
    assert Path(result["figure_path"]).exists()
    assert Path(result["report_path"]).exists()
    payload = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert payload["protocol"]["noise_location"] == "sis_dynamics"
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "tau=1.0" in report
    assert "SIS 动力学过程噪声" in report
    assert "MLP+PEID" in report


def test_cli_can_be_invoked_by_script_path() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sis_next_state_peid_alignment.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--mode" in completed.stdout
