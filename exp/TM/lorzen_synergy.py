from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exp.TM.lorzen_tm_ei import load_lorzen_transition_csvs
from yrd import (
    clip_nonnegative_ei,
    estimate_mutual_information_transport_map,
    lift_transport_source_features,
    summarize_two_source_synergy_transport_map,
)


def run_lorzen_synergy_check(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    sample_mode: str = "observed",
    sample_count: int = 3000,
    box_width: float = 8.0,
    forcing: float = 8.0,
    seed: int = 4,
) -> dict[str, Any]:
    source_df, _ = load_lorzen_transition_csvs(input_path)
    labels = list(source_df.columns)
    source = _resolve_source_samples(
        source_df.to_numpy(dtype=float),
        sample_mode=sample_mode,
        sample_count=sample_count,
        box_width=box_width,
        seed=seed,
    )

    product_rows: list[dict[str, Any]] = []
    rhs_rows: list[dict[str, Any]] = []
    for target_index, target_label in enumerate(labels):
        self_index = target_index
        prev1_index = (target_index - 1) % len(labels)
        prev2_index = (target_index - 2) % len(labels)
        next1_index = (target_index + 1) % len(labels)

        self_value = source[:, [self_index]]
        prev1 = source[:, [prev1_index]]
        prev2 = source[:, [prev2_index]]
        next1 = source[:, [next1_index]]
        positive_product = next1 * prev1
        negative_product = prev2 * prev1
        interaction = positive_product - negative_product
        rhs = interaction - self_value + float(forcing)

        for term_name, left_label, left, right_label, right, target in (
            ("positive_product", labels[next1_index], next1, labels[prev1_index], prev1, positive_product),
            ("negative_product", labels[prev2_index], prev2, labels[prev1_index], prev1, negative_product),
            ("full_interaction_pos_pair", labels[next1_index], next1, labels[prev1_index], prev1, interaction),
            ("full_interaction_neg_pair", labels[prev2_index], prev2, labels[prev1_index], prev1, interaction),
        ):
            product_rows.append(
                {
                    "target": target_label,
                    "term": term_name,
                    "left_source": left_label,
                    "right_source": right_label,
                    **summarize_two_source_synergy_transport_map(left, right, target),
                }
            )

        self_ei = _single_source_tm_ei(self_value, rhs)
        pos_pair = summarize_two_source_synergy_transport_map(next1, prev1, rhs)
        neg_pair = summarize_two_source_synergy_transport_map(prev2, prev1, rhs)
        rhs_rows.extend(
            [
                {
                    "target": target_label,
                    "rhs_component": "linear_self",
                    "sources": labels[self_index],
                    "ei": self_ei,
                    "joint_ei": self_ei,
                    "synergy": 0.0,
                },
                {
                    "target": target_label,
                    "rhs_component": "positive_product_pair",
                    "sources": f"{labels[next1_index]}+{labels[prev1_index]}",
                    "ei": pos_pair["joint_ei"],
                    "joint_ei": pos_pair["joint_ei"],
                    "synergy": pos_pair["syn"],
                },
                {
                    "target": target_label,
                    "rhs_component": "negative_product_pair",
                    "sources": f"{labels[prev2_index]}+{labels[prev1_index]}",
                    "ei": neg_pair["joint_ei"],
                    "joint_ei": neg_pair["joint_ei"],
                    "synergy": neg_pair["syn"],
                },
            ]
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    product_frame = pd.DataFrame(product_rows)
    rhs_frame = pd.DataFrame(rhs_rows)
    product_path = output_path / "lorzen_product_synergy.csv"
    rhs_path = output_path / "lorzen_rhs_synergy.csv"
    summary_path = output_path / "lorzen_synergy_summary.json"
    product_frame.to_csv(product_path, index=False)
    rhs_frame.to_csv(rhs_path, index=False)

    summary = {
        "sample_mode": sample_mode,
        "sample_count": int(source.shape[0]),
        "box_width": float(box_width),
        "forcing": float(forcing),
        "seed": int(seed),
        "product_terms": _summarize_frame(product_frame, value_columns=("joint_ei", "left_ei", "right_ei", "syn")),
        "rhs_terms": _summarize_frame(rhs_frame, value_columns=("joint_ei", "synergy")),
        "artifacts": {
            "product_synergy": str(product_path),
            "rhs_synergy": str(rhs_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "product_synergy": product_frame,
        "rhs_synergy": rhs_frame,
        "summary": summary,
        "summary_path": summary_path,
    }


def _resolve_source_samples(
    observed_source: np.ndarray,
    *,
    sample_mode: str,
    sample_count: int,
    box_width: float,
    seed: int,
) -> np.ndarray:
    if sample_mode == "observed":
        return np.asarray(observed_source, dtype=float)
    if sample_mode == "uniform_box":
        if sample_count < 4:
            raise ValueError("sample_count must be at least 4.")
        if box_width <= 0.0:
            raise ValueError("box_width must be positive.")
        rng = np.random.default_rng(seed)
        center = np.asarray(observed_source, dtype=float).mean(axis=0)
        half_width = float(box_width) / 2.0
        return rng.uniform(center - half_width, center + half_width, size=(int(sample_count), observed_source.shape[1]))
    raise ValueError("sample_mode must be 'observed' or 'uniform_box'.")


def _single_source_tm_ei(source: np.ndarray, target: np.ndarray) -> float:
    summary = estimate_mutual_information_transport_map(lift_transport_source_features(source), target)
    return clip_nonnegative_ei(float(summary["mi_hat"]))


def _summarize_frame(frame: pd.DataFrame, *, value_columns: tuple[str, ...]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    group_column = "term" if "term" in frame.columns else "rhs_component"
    for group_name, group in frame.groupby(group_column):
        summary[str(group_name)] = {
            f"{column}_mean": float(group[column].mean())
            for column in value_columns
            if column in group.columns
        }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check Lorenz-96 product-term TM-EI synergy.")
    parser.add_argument("input_path", type=Path, help="Directory or zip containing yt.csv and yt+1.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("exp/TM/lorzen/results_synergy"))
    parser.add_argument("--sample-mode", choices=("observed", "uniform_box"), default="uniform_box")
    parser.add_argument("--sample-count", type=int, default=3000)
    parser.add_argument("--box-width", type=float, default=8.0)
    parser.add_argument("--forcing", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args(argv)
    result = run_lorzen_synergy_check(
        args.input_path,
        output_dir=args.output_dir,
        sample_mode=args.sample_mode,
        sample_count=args.sample_count,
        box_width=args.box_width,
        forcing=args.forcing,
        seed=args.seed,
    )
    print(json.dumps({"summary_path": str(result["summary_path"]), "summary": result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
