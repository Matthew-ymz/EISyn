import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from density_benchmark import (
    DensityBenchmarkConfig,
    DensityPlotConfig,
    DENSITY_METHOD_ORDER,
    load_or_run_density_benchmark,
    plot_density_benchmark_composite,
)


class DensityBenchmarkTests(unittest.TestCase):
    def tiny_config(self, root: Path) -> DensityBenchmarkConfig:
        return DensityBenchmarkConfig(
            cache_dir=root / "cache",
            fig_dir=root / "fig",
            families=("gaussian",),
            methods=("transport_map", "kde", "knn"),
            accuracy_repeats=2,
            scan_repeats=2,
            accuracy_dim=2,
            scan_dim=2,
            n_train=80,
            n_test=60,
            sample_sizes=(40, 80),
            total_dims=(2, 3),
            seed=11,
        )

    def test_tiny_benchmark_writes_expected_repeats_and_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.tiny_config(Path(tmpdir))

            results = load_or_run_density_benchmark(config=config, force=True)

            accuracy_raw = results["accuracy_raw"]
            expected_methods = set(DENSITY_METHOD_ORDER)
            self.assertEqual(set(accuracy_raw["method"]), expected_methods)
            self.assertEqual(int(accuracy_raw["repeat"].nunique()), 2)
            self.assertEqual(len(accuracy_raw), 2 * len(expected_methods))

            accuracy_summary = results["accuracy_summary"]
            required_columns = {
                "rmse_log_density_mean",
                "rmse_log_density_sem",
                "kl_p_phat_mean",
                "kl_p_phat_sem",
                "heldout_nll_mean",
                "total_time_mean",
            }
            self.assertTrue(required_columns.issubset(accuracy_summary.columns))
            self.assertFalse(accuracy_summary[list(required_columns)].isna().any().any())

    def test_cache_reuse_loads_existing_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self.tiny_config(Path(tmpdir))
            first = load_or_run_density_benchmark(config=config, force=True)

            marker = config.cache_dir / "accuracy_summary.csv"
            cached_summary = pd.read_csv(marker)
            cached_summary.loc[:, "rmse_log_density_mean"] = 123.456
            cached_summary.to_csv(marker, index=False)

            second = load_or_run_density_benchmark(config=config, force=False)

            self.assertNotEqual(
                float(first["accuracy_summary"]["rmse_log_density_mean"].iloc[0]),
                float(second["accuracy_summary"]["rmse_log_density_mean"].iloc[0]),
            )
            self.assertEqual(float(second["accuracy_summary"]["rmse_log_density_mean"].iloc[0]), 123.456)

    def test_composite_plot_exports_png_and_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self.tiny_config(root)
            results = load_or_run_density_benchmark(config=config, force=True)
            plot_config = DensityPlotConfig(
                fig_dir=config.fig_dir,
                basename="density_benchmark_composite_test",
            )

            paths = plot_density_benchmark_composite(results, plot_config=plot_config)

            self.assertEqual({path.suffix for path in paths}, {".png", ".pdf"})
            for path in paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
