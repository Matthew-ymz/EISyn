import unittest

import pandas as pd

from yrd.plotting import render_station_causal_graph_svg


class YRDStationGraphRenderingTests(unittest.TestCase):
    def test_render_station_causal_graph_places_binary_hyperedge_junction(self) -> None:
        station_positions = pd.DataFrame(
            [
                {"station_id": "A", "lon": 121.3, "lat": 31.1},
                {"station_id": "B", "lon": 121.4, "lat": 31.2},
                {"station_id": "C", "lon": 121.5, "lat": 31.15},
            ]
        )
        pairwise_edges = pd.DataFrame(
            [
                {"source_station_id": "A", "target_station_id": "C", "abs_mean": 1.2},
            ]
        )
        binary_hyperedges = pd.DataFrame(
            [
                {"source_station_ids": ("A", "B"), "target_station_id": "C", "abs_mean": 0.9},
            ]
        )

        svg = render_station_causal_graph_svg(
            station_positions=station_positions,
            pairwise_edges=pairwise_edges,
            binary_hyperedges=binary_hyperedges,
            horizon_label="24h",
        )

        self.assertIn("stroke-dasharray='5,3'", svg)
        self.assertIn("<circle", svg)
        self.assertIn("Binary hyperedge", svg)
        self.assertIn("Pairwise edge", svg)


if __name__ == "__main__":
    unittest.main()
