from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_greedy_hierarchy_known_dynamics import (
    PLANTED_MODULES,
    SOURCES,
    decompose,
    generate_data,
    oracle_ei_table,
    summarize_run,
)
from scripts.phi_hierarchy import subset_phi_raw


def test_oracle_hierarchy_recovers_both_dynamical_terms_exactly() -> None:
    table = oracle_ei_table(noise=0.05)
    atoms = decompose(table)

    assert {atom.sources for atom in atoms} == set(PLANTED_MODULES)
    assert np.isclose(sum(atom.value for atom in atoms), subset_phi_raw(SOURCES, table))


def test_generated_targets_follow_the_planted_xor_equations_without_noise() -> None:
    source, target = generate_data(sample_count=256, noise=0.0, seed=7)

    assert np.array_equal(target[:, 0], source[:, 0] ^ source[:, 1] ^ source[:, 2])
    assert np.array_equal(target[:, 1], source[:, 3] ^ source[:, 4])


def test_empirical_run_recovers_planted_terms_at_main_sample_size() -> None:
    result = summarize_run(sample_count=32768, noise=0.05, seed=0)

    assert result["top2_exact_recovery"]
    assert result["planted_atom_mass_fraction"] > 0.98
    assert result["subset_ei_rmse_bits"] < 0.02
    assert abs(result["closure_error_bits"]) < 1.0e-7
