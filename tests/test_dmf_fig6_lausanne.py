import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "exp" / "brain" / "dmf_fig6.py"

spec = importlib.util.spec_from_file_location("dmf_fig6_exp_brain", MODULE_PATH)
dmf_fig6 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dmf_fig6
spec.loader.exec_module(dmf_fig6)


class LausanneZipLoadingTests(unittest.TestCase):
    def test_load_lausanne_zip_labels_and_drops_unknown_region(self) -> None:
        zip_path = ROOT / "data" / "Lausanne2008-33.zip"

        labels = dmf_fig6.load_lausanne_region_labels(zip_path)
        entries = dmf_fig6.load_lausanne_atlas_entries(zip_path)
        connectivity, kept_labels = dmf_fig6.prepare_lausanne_count_connectivity(
            entries[0]["count"],
            region_labels=labels,
            expected_regions=83,
            return_labels=True,
        )

        self.assertEqual(connectivity.shape, (83, 83))
        self.assertEqual(len(kept_labels), 83)
        self.assertNotIn("Unknown", kept_labels)
        self.assertTrue(np.allclose(np.diag(connectivity), 0.0))
        self.assertTrue(np.isfinite(connectivity).all())
        raw_scaled = dmf_fig6.prepare_lausanne_count_connectivity(
            entries[0]["count"],
            region_labels=labels,
            expected_regions=83,
            connectivity_scale=1.0,
        )
        self.assertTrue(np.allclose(connectivity, 0.2 * raw_scaled))

    def test_connectivity_scale_can_restore_raw_max_normalization(self) -> None:
        zip_path = ROOT / "data" / "Lausanne2008-33.zip"

        labels = dmf_fig6.load_lausanne_region_labels(zip_path)
        entries = dmf_fig6.load_lausanne_atlas_entries(zip_path)
        connectivity = dmf_fig6.prepare_lausanne_count_connectivity(
            entries[0]["count"],
            region_labels=labels,
            expected_regions=83,
            connectivity_scale=1.0,
        )

        self.assertAlmostEqual(float(connectivity.max()), 0.5314685314685315)

    def test_pairwise_phi_metrics_match_manual_bivariate_formula(self) -> None:
        x0 = np.arange(1.0, 9.0)
        x1 = np.array([2.0, 0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 3.0])
        rates = np.column_stack([x0, x1])

        metrics = dmf_fig6.compute_pairwise_phi_metrics(rates)

        lagged = np.column_stack([rates[:-1, 0], rates[:-1, 1], rates[1:, 0], rates[1:, 1]])
        covariance = np.cov(lagged, rowvar=False, bias=False)
        tdmi = dmf_fig6.gaussian_mutual_information(covariance, sources=[0, 1], targets=[2, 3])
        mi_self_0 = dmf_fig6.gaussian_mutual_information(covariance, sources=[0], targets=[2])
        mi_self_1 = dmf_fig6.gaussian_mutual_information(covariance, sources=[1], targets=[3])
        phi_wms = tdmi - mi_self_0 - mi_self_1
        redundancy = min(
            dmf_fig6.gaussian_mutual_information(covariance, sources=[source], targets=[target])
            for source in (0, 1)
            for target in (2, 3)
        )

        self.assertEqual(metrics["pair_count"], 1)
        self.assertAlmostEqual(metrics["phi_wms_mean"], phi_wms)
        self.assertAlmostEqual(metrics["phi_r_mean"], phi_wms + redundancy)


if __name__ == "__main__":
    unittest.main()
