"""Regression coverage for exhaustive degree-3 TM Runge map inputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np


def canonical_candidates(n_components: int = 60) -> np.ndarray:
    return np.asarray(
        [
            (source_a, source_b, target)
            for source_a in range(n_components)
            for source_b in range(source_a + 1, n_components)
            for target in range(n_components)
            if target not in (source_a, source_b)
        ],
        dtype=np.int16,
    )


def fingerprint_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def write_fixture(
    root: Path,
    *,
    mutate_arrays: Callable[[dict[str, np.ndarray]], object] | None = None,
    metadata_updates: dict[str, object] | None = None,
) -> Path:
    horizon_dir = root / "H010"
    horizon_dir.mkdir(parents=True)
    candidates = canonical_candidates()
    assert len(candidates) == 102660
    count = len(candidates)
    arrays = {
        "source_a": candidates[:, 0],
        "source_b": candidates[:, 1],
        "target": candidates[:, 2],
        "raw_ei_a": np.full(count, 0.25),
        "raw_ei_b": np.full(count, 0.5),
        "raw_joint_ei": np.full(count, 1.0),
        "ei_a": np.full(count, 0.25),
        "ei_b": np.full(count, 0.5),
        "joint_ei": np.full(count, 1.0),
        "delta2_tm": np.linspace(10.0, 0.0, count, dtype=float),
        "tm_rank": np.arange(1, count + 1, dtype=np.int32),
    }
    if mutate_arrays is not None:
        mutate_arrays(arrays)
    triples = np.column_stack([arrays["source_a"], arrays["source_b"], arrays["target"]])
    metadata = {
        "schema_version": 1,
        "input_fingerprint": "fixture-input-fingerprint",
        "estimator_fingerprint": "fixture-estimator-fingerprint",
        "horizon": 10,
        "candidate_count": count,
        "candidate_order_hash": fingerprint_array(triples),
    }
    metadata.update(metadata_updates or {})
    np.savez_compressed(
        horizon_dir / "full_ranking.npz",
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )
    (horizon_dir / "summary.json").write_text(
        json.dumps(
            {
                "horizon": 10,
                "input_fingerprint": metadata["input_fingerprint"],
                "estimator_fingerprint": metadata["estimator_fingerprint"],
                "candidate_count": count,
                "finite": True,
                "ranking_metadata": metadata,
            }
        ),
        encoding="utf-8",
    )
    return root


class ExhaustiveTMMapInputTests(unittest.TestCase):
    def test_direct_cli_exposes_the_map_options(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "plot_runge_exhaustive_tm_maps.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[1],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--result-dir", result.stdout)
        self.assertIn("--component-maps", result.stdout)

    def test_loads_verified_global_top10_with_paper_component_labels(self) -> None:
        from scripts.plot_runge_exhaustive_tm_maps import load_exhaustive_top10

        with tempfile.TemporaryDirectory() as tmpdir:
            frame = load_exhaustive_top10(write_fixture(Path(tmpdir)), horizon=10)

        self.assertEqual(len(frame), 10)
        self.assertEqual(frame["tm_rank"].tolist(), list(range(1, 11)))
        self.assertEqual(frame["source_a_local"].tolist(), [0] * 10)
        self.assertEqual(frame["source_b_local"].tolist(), [1] * 10)
        self.assertEqual(frame["target_local"].tolist(), list(range(2, 12)))
        self.assertEqual(frame["source_a_paper"].tolist(), [0] * 10)
        self.assertEqual(frame["source_b_paper"].tolist(), [1] * 10)
        self.assertEqual(frame["target_paper"].tolist(), [2, 3, 4, 5, 6, 18, 26, 9, 10, 11])
        self.assertTrue(np.isfinite(frame[["delta2_tm", "joint_ei", "ei_a", "ei_b"]].to_numpy()).all())
        self.assertEqual(set(frame["input_fingerprint"]), {"fixture-input-fingerprint"})

    def test_rejects_ranking_with_noncanonical_candidate_tuple(self) -> None:
        from scripts.plot_runge_exhaustive_tm_maps import load_exhaustive_top10

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "source_a < source_b|candidate universe"):
                load_exhaustive_top10(
                    write_fixture(
                        Path(tmpdir),
                        mutate_arrays=lambda arrays: arrays["source_b"].__setitem__(0, 0),
                    ),
                    horizon=10,
                )

    def test_rejects_out_of_range_indices_before_integer_narrowing(self) -> None:
        from scripts.plot_runge_exhaustive_tm_maps import load_exhaustive_top10

        def wrap_target_index(arrays: dict[str, np.ndarray]) -> None:
            target = arrays["target"].astype(np.int64)
            target[0] = 65538  # Wraps to the valid local index 2 under int16.
            arrays["target"] = target

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, r"\[0, 59\]"):
                load_exhaustive_top10(
                    write_fixture(
                        Path(tmpdir),
                        mutate_arrays=wrap_target_index,
                        metadata_updates={"candidate_order_hash": fingerprint_array(canonical_candidates())},
                    ),
                    horizon=10,
                )

    def test_rejects_metadata_with_an_unsupported_schema_version(self) -> None:
        from scripts.plot_runge_exhaustive_tm_maps import load_exhaustive_top10

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_exhaustive_top10(
                    write_fixture(Path(tmpdir), metadata_updates={"schema_version": 2}),
                    horizon=10,
                )

    def test_rejects_each_required_ranking_invariant(self) -> None:
        from scripts.plot_runge_exhaustive_tm_maps import load_exhaustive_top10

        mutations = {
            "rank": lambda arrays: arrays["tm_rank"].__setitem__(1, 1),
            "delta": lambda arrays: arrays["delta2_tm"].__setitem__(1, 11.0),
            "finite": lambda arrays: arrays["joint_ei"].__setitem__(2, np.nan),
            "target": lambda arrays: arrays["target"].__setitem__(0, 0),
            "bounds": lambda arrays: arrays["target"].__setitem__(0, 60),
            "duplicate": lambda arrays: [
                arrays[column].__setitem__(-1, arrays[column][0])
                for column in ("source_a", "source_b", "target")
            ],
        }
        for name, mutate_arrays in mutations.items():
            with self.subTest(invariant=name), tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaises(ValueError):
                    load_exhaustive_top10(write_fixture(Path(tmpdir), mutate_arrays=mutate_arrays), horizon=10)

    def test_rejects_a_candidate_hash_that_does_not_describe_the_ranking_order(self) -> None:
        from scripts.plot_runge_exhaustive_tm_maps import load_exhaustive_top10

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "order hash"):
                load_exhaustive_top10(
                    write_fixture(Path(tmpdir), metadata_updates={"candidate_order_hash": "not-the-array-hash"}),
                    horizon=10,
                )

    def test_plot_horizons_writes_the_map_and_top10_table_in_all_requested_formats(self) -> None:
        from scripts.plot_runge_exhaustive_tm_maps import plot_horizons

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            component_maps = root / "component_maps.npz"
            np.savez_compressed(component_maps, component_maps=np.arange(4 * 60, dtype=float).reshape(2, 2, 60))
            outputs = plot_horizons(write_fixture(root / "ranking"), [10], root / "figures", component_maps)

            self.assertEqual({path.suffix for path in outputs[10]}, {".png", ".svg", ".pdf", ".csv"})
            for path in outputs[10]:
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
