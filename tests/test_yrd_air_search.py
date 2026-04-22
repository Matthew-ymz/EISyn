import unittest
from pathlib import Path
import tempfile

from yrd.air_search import (
    build_air_search_artifact_paths,
    build_air_search_config,
    prepare_air_search_bundle,
    resolve_city_scope,
    run_or_load_air_search_predictions,
    summarize_air_search_station_pollutant_effects,
)


class AirSearchRoutingTests(unittest.TestCase):
    def test_resolve_city_scope_routes_beijing_to_bthsa_files(self) -> None:
        scope = resolve_city_scope("beijing")

        self.assertEqual(scope.dataset_path, Path("data/dataset_bthsa.nc"))
        self.assertEqual(scope.station_path, Path("data/stations_bthsa.csv"))

    def test_build_air_search_config_keeps_one_step_anchor_shape(self) -> None:
        cfg = build_air_search_config(Path("."), city_en="shanghai", horizon=3, test_mode=True)

        self.assertEqual(cfg.sample_mode, "one_step")
        self.assertEqual(cfg.history_hours, 1)
        self.assertEqual(cfg.horizons, (3,))
        self.assertEqual(cfg.dataset_path, Path("data/dataset_yrd.nc"))
        self.assertEqual(cfg.station_path, Path("data/stations_yrd.csv"))


class AirSearchPathTests(unittest.TestCase):
    def test_build_air_search_artifact_paths_use_city_and_horizon_namespaces(self) -> None:
        paths = build_air_search_artifact_paths(
            root_dir=Path("/tmp/eisyn"),
            city_en="hangzhou",
            horizon=6,
            run_tag="trial",
            use_smoke=False,
        )

        self.assertEqual(
            paths["cache_dir"],
            Path("/tmp/eisyn/exp/cache/yrd_coupling/air_search/hangzhou/6h/trial"),
        )
        self.assertEqual(
            paths["results_dir"],
            Path("/tmp/eisyn/fig/yrd_air_search/hangzhou/6h/trial"),
        )


class AirSearchBundleTests(unittest.TestCase):
    def test_prepare_air_search_bundle_smoke_supports_non_shanghai_city(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cfg = build_air_search_config(project_root, city_en="nanjing", horizon=3, test_mode=True)

        bundle = prepare_air_search_bundle(
            cfg=cfg,
            city_en="nanjing",
            run_tag="bundle-smoke",
            use_smoke=True,
        )

        self.assertEqual(bundle["cfg"].sample_mode, "one_step")
        self.assertEqual(set(bundle["y_train_scaled"]), {3})
        self.assertTrue(bundle["station_ids"])
        self.assertTrue(bundle["artifact_paths"]["cache_dir"].exists())
        self.assertTrue(bundle["artifact_paths"]["results_dir"].exists())

    def test_run_or_load_air_search_predictions_writes_smoke_cache(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cfg = build_air_search_config(project_root, city_en="shanghai", horizon=3, test_mode=True)
        with tempfile.TemporaryDirectory(prefix="air-search-") as tmpdir:
            bundle = prepare_air_search_bundle(
                cfg=cfg,
                city_en="shanghai",
                run_tag=Path(tmpdir).name,
                use_smoke=True,
            )

            predictions = run_or_load_air_search_predictions(bundle, force_retrain=True)

        self.assertTrue(bundle["artifact_paths"]["checkpoint"].exists())
        self.assertTrue(bundle["artifact_paths"]["loss_history"].exists())
        self.assertTrue(bundle["artifact_paths"]["predictions"].exists())
        self.assertTrue(bundle["artifact_paths"]["run_manifest"].exists())
        self.assertEqual(set(predictions["joint_original_predictions"]), {3})
        self.assertEqual(
            predictions["joint_original_predictions"][3].shape,
            predictions["y_test_original"][3].shape,
        )


class AirSearchSummaryTests(unittest.TestCase):
    def test_summarize_air_search_station_pollutant_effects_returns_synergy_and_pairwise_views(self) -> None:
        summary = summarize_air_search_station_pollutant_effects(
            sample_summaries=[
                {
                    "target_station_id": "A",
                    "station_pair_synergy_nis": {"A": 0.6, "B": 0.2},
                    "joint_station_pair_ei_nis": {"A": 1.1, "B": 0.8},
                    "single_pollutant_ei_nis": {
                        "A": {"O3": 0.3, "PM2.5": 0.5},
                        "B": {"O3": 0.2, "PM2.5": 0.4},
                    },
                }
            ],
            station_ids=["A", "B"],
        )

        self.assertIn("conditional_synergy", summary)
        self.assertIn("single_pollutant_pairwise", summary)
        self.assertTrue(summary["conditional_synergy"]["conditional_synergy_edges"])
        self.assertTrue(summary["single_pollutant_pairwise"]["pairwise_edges"])


if __name__ == "__main__":
    unittest.main()
