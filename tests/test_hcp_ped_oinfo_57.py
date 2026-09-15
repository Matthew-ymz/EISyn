from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hcp_ped_oinfo_57 import (
    PED_RED_INDEX,
    PED_SYN_INDICES,
    binarize_like_authors,
    copnorm_demean,
    ped_atoms_from_binary,
    triplet_oinfo,
)


TRIPLET = (("Vis", "SomMot", "DorsAttn"),)


def seven_column_signal(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(19)
    return np.column_stack([first, second, third, rng.normal(size=(len(first), 4))])


def test_ped_atoms_sum_to_joint_entropy_and_are_nonnegative() -> None:
    binary = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=np.int16)
    atoms = ped_atoms_from_binary(binary)
    assert np.isclose(atoms.sum(), 3.0)
    assert atoms.min() >= -1.0e-12
    assert atoms[PED_RED_INDEX] > 0.0
    assert atoms[np.asarray(PED_SYN_INDICES)].sum() > 0.0


def test_ped_red_and_syn_are_permutation_invariant() -> None:
    rng = np.random.default_rng(11)
    binary = rng.integers(0, 2, size=(513, 3), dtype=np.int16)
    reference = ped_atoms_from_binary(binary)
    reference_pair = (reference[PED_RED_INDEX], reference[np.asarray(PED_SYN_INDICES)].sum())
    for order in itertools.permutations(range(3)):
        atoms = ped_atoms_from_binary(binary[:, order])
        assert np.allclose((atoms[PED_RED_INDEX], atoms[np.asarray(PED_SYN_INDICES)].sum()), reference_pair, atol=1.0e-12)


def test_binarization_matches_zscore_sign_threshold() -> None:
    values = np.asarray([[1.0, 8.0], [2.0, 2.0], [6.0, 5.0]])
    expected = ((values - values.mean(axis=0)) / values.std(axis=0) > 0.0).astype(np.int16)
    assert np.array_equal(binarize_like_authors(values), expected)


def test_copnorm_is_rank_based_and_demeaned() -> None:
    values = np.asarray([[4.0, 1.0, 3.0, 2.0], [20.0, 40.0, 10.0, 30.0]])
    transformed = copnorm_demean(values)
    assert np.allclose(transformed.mean(axis=1), 0.0, atol=1.0e-14)
    assert np.array_equal(np.argsort(transformed, axis=1), np.argsort(values, axis=1))


def test_triplet_oinfo_detects_redundancy_and_synergy_directions() -> None:
    rng = np.random.default_rng(13)
    n_samples = 4_000
    shared = rng.normal(size=n_samples)
    redundant = seven_column_signal(shared + 0.25 * rng.normal(size=n_samples), shared + 0.25 * rng.normal(size=n_samples), shared + 0.25 * rng.normal(size=n_samples))
    x = rng.normal(size=n_samples)
    y = rng.normal(size=n_samples)
    synergistic = seven_column_signal(x, y, x + y + 0.2 * rng.normal(size=n_samples))
    assert triplet_oinfo(redundant, TRIPLET)[0] > 0.0
    assert triplet_oinfo(synergistic, TRIPLET)[0] < 0.0
