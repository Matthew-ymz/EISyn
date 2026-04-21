import unittest

import pandas as pd

from yrd.groups import build_city_groups, build_nearest_neighbor_groups


class YRDGroupTests(unittest.TestCase):
    def test_build_city_groups_collects_station_ids(self) -> None:
        frame = pd.DataFrame(
            {"station_id": ["a", "b", "c"], "city_en": ["shanghai", "shanghai", "nanjing"]}
        )
        groups = build_city_groups(frame)
        self.assertEqual(groups["shanghai"], ["a", "b"])
        self.assertEqual(groups["nanjing"], ["c"])

    def test_build_nearest_neighbor_groups_excludes_self(self) -> None:
        frame = pd.DataFrame(
            {
                "station_id": ["a", "b", "c"],
                "lon": [121.0, 121.1, 130.0],
                "lat": [31.0, 31.1, 40.0],
            }
        )
        groups = build_nearest_neighbor_groups(frame, k=1)
        self.assertEqual(groups["a"], ["b"])


if __name__ == "__main__":
    unittest.main()
