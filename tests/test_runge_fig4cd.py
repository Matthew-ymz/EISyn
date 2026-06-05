import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reproduce_runge_fig4cd import build_fig4cd_frame, plot_fig4cd


class RungeFig4CDTests(unittest.TestCase):
    def test_build_fig4cd_frame_computes_reach_fraction_from_total_effects(self) -> None:
        gateway = pd.DataFrame(
            {
                "component": [0, 1, 2],
                "paper_component": [0, 18, 2],
                "ace": [0.3, 0.2, 0.1],
                "acs": [0.05, 0.02, 0.01],
            }
        )
        mediator = pd.DataFrame(
            {
                "component": [0, 1, 2],
                "paper_component": [0, 18, 2],
                "amce": [0.001, 0.002, 0.0],
                "mediated_fraction": [0.5, 0.75, 0.0],
            }
        )
        total = pd.DataFrame(
            {
                "source": [0, 0, 0, 1, 1, 2],
                "target": [1, 1, 2, 0, 2, 0],
                "lag": [1, 2, 1, 1, 1, 1],
                "total_effect": [0.04, 0.08, -0.06, 0.10, 0.02, -0.01],
            }
        )

        frame = build_fig4cd_frame(gateway, mediator, total, effect_threshold=0.05)

        row0 = frame.set_index("paper_component").loc[0]
        self.assertAlmostEqual(float(row0["nout_fraction"]), 1.0)
        self.assertAlmostEqual(float(row0["amce"]), 0.001)
        self.assertAlmostEqual(float(row0["mediated_fraction"]), 0.5)
        self.assertEqual(int(row0["component"]), 0)

    def test_plot_fig4cd_writes_nonempty_png(self) -> None:
        frame = pd.DataFrame(
            {
                "component": np.arange(6),
                "paper_component": np.arange(6),
                "ace": [0.06, 0.05, 0.048, 0.02, 0.015, 0.01],
                "acs": [0.05, 0.045, 0.04, 0.015, 0.01, 0.005],
                "amce": [0.0020, 0.0017, 0.0015, 0.0004, 0.0002, 0.0001],
                "nout_fraction": [0.7, 0.6, 0.55, 0.2, 0.1, 0.05],
                "mediated_fraction": [0.9, 0.85, 0.8, 0.3, 0.15, 0.05],
                "is_highlight": [True, True, True, False, False, False],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "fig4cd.png"

            plot_fig4cd(frame, output)

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
