from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tempfile
from itertools import product
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = Path(tempfile.gettempdir()) / "eisyn-matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "DejaVu Serif"

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from utils import (
    coarse_grain_tpm_by_state_labels,
    effective_information_from_tpm,
    enumerate_binary_states,
    enumerate_partitions_fixed_blocks,
    enumerate_surjective_binary_mappings,
)

MICRO_LABELS = ("A", "B", "C", "D")
HOEL_SYSTEM_ID = "hoel_fig2_toy_example"
HOEL_GROUPS = ((0, 1), (2, 3))
HOEL_MAPPINGS = ((0, 0, 0, 1), (0, 0, 0, 1))
HOEL_Q_OFF = 0.3
HOEL_SHARED_NOISE_GAMMA = 0.0
FAMILY_Q_OFF_VALUES = tuple(index / 20.0 for index in range(21))

# v2: Syn macro uses EI(Z_i->X+) marginalized from the same Z->X+ TPM
# intervention regime as EI(Z->X+), so older cached payloads must be invalidated.
DEFAULT_CACHE_VERSION = "rq3_hoel_micro_mechanism_v2"
DEFAULT_CACHE_DIR = REPO_ROOT / "exp" / "cache" / "rq3_boolean_causal_emergence"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "rq3_boolean_causal_emergence"
RANK_TIE_ABS_TOL = 1e-12
RANK_TIE_REL_TOL = 1e-12

ALL_MICRO_STATES = enumerate_binary_states(4)
MICRO_STATE_TO_INDEX = {
    tuple(int(bit) for bit in state.tolist()): index
    for index, state in enumerate(ALL_MICRO_STATES)
}
STATE_BITS_BY_SIZE = {
    size: [tuple(int(bit) for bit in row.tolist()) for row in enumerate_binary_states(size)]
    for size in (2,)
}

KNOWN_MAPPING_NAMES = {
    (0, 0, 0, 1): "AND/off-on",
    (0, 0, 1, 0): "bit-10 detector",
    (0, 0, 1, 1): "copy right",
    (0, 1, 0, 0): "bit-01 detector",
    (0, 1, 0, 1): "copy left",
    (0, 1, 1, 0): "XOR",
    (0, 1, 1, 1): "OR/on-if-any",
}

REPRESENTATIVE_MAPPINGS_BY_SIZE = {
    2: [
        mapping
        for mapping in enumerate_surjective_binary_mappings(2)
        if mapping[0] == 0
    ]
}

ALL_PARTITIONS = sorted(
    enumerate_partitions_fixed_blocks(range(4), 2),
    key=lambda groups: (tuple(sorted(len(block) for block in groups)), groups),
)
PAIR_PARTITIONS = [
    groups
    for groups in ALL_PARTITIONS
    if tuple(sorted(len(block) for block in groups)) == (2, 2)
]

BLOCK_STATE_LABELS: dict[tuple[int, ...], np.ndarray] = {}
for groups in PAIR_PARTITIONS:
    for block in groups:
        block_tuple = tuple(block)
        if block_tuple in BLOCK_STATE_LABELS:
            continue
        labels = np.zeros(len(ALL_MICRO_STATES), dtype=int)
        for row_index, state in enumerate(ALL_MICRO_STATES):
            block_bits = tuple(int(state[index]) for index in block_tuple)
            labels[row_index] = int(
                sum(bit << offset for offset, bit in enumerate(reversed(block_bits)))
            )
        BLOCK_STATE_LABELS[block_tuple] = labels

CORRELATION_TERM_ORDER = ["neg_syn_micro", "neg_loss_sum", "syn_macro"]
CORRELATION_TARGET_TERM = "macro_to_full_ei"
TERM_LABELS = {
    "neg_syn_micro": "neg. syn",
    "neg_loss_sum": "neg. loss",
    "syn_macro": "Syn macro",
    "macro_to_full_ei": "EI(Z->X+)",
}

FAMILY_TERM_ORDER = CORRELATION_TERM_ORDER
FAMILY_TERM_COLORS = {
    "neg_syn_micro": "#1d4ed8",
    "neg_loss_sum": "#9a3412",
    "syn_macro": "#0f766e",
}


def partition_signature(groups: tuple[tuple[int, ...], tuple[int, ...]]) -> str:
    return " | ".join("{" + ",".join(MICRO_LABELS[index] for index in block) + "}" for block in groups)


def mapping_name(mapping: tuple[int, ...]) -> str:
    mapping = tuple(int(bit) for bit in mapping)
    if mapping in KNOWN_MAPPING_NAMES:
        return f"{KNOWN_MAPPING_NAMES[mapping]} [{''.join(str(bit) for bit in mapping)}]"
    return f"2b [{''.join(str(bit) for bit in mapping)}]"


def average_rows_by_label(system_tpm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    n_labels = int(labels.max()) + 1
    averaged = np.zeros((n_labels, system_tpm.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_labels).astype(float)
    np.add.at(averaged, labels, system_tpm)
    averaged /= counts[:, None]
    return averaged


def labels_for_candidate(
    groups: tuple[tuple[int, ...], tuple[int, ...]],
    mappings: tuple[tuple[int, ...], tuple[int, ...]],
) -> np.ndarray:
    block_bits = []
    for block, mapping in zip(groups, mappings):
        state_ids = BLOCK_STATE_LABELS[tuple(block)]
        block_bits.append(np.asarray(mapping, dtype=int)[state_ids])
    return 2 * block_bits[0] + block_bits[1]


def build_candidate_catalog() -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    for groups in PAIR_PARTITIONS:
        block_state_labels = [BLOCK_STATE_LABELS[tuple(block)] for block in groups]
        for mappings in product(REPRESENTATIVE_MAPPINGS_BY_SIZE[2], repeat=2):
            mapped_block_labels = [
                np.asarray(mapping, dtype=int)[raw_labels]
                for mapping, raw_labels in zip(mappings, block_state_labels)
            ]
            macro_labels = 2 * mapped_block_labels[0] + mapped_block_labels[1]
            catalog.append(
                {
                    "groups": groups,
                    "partition_name": partition_signature(groups),
                    "mappings": tuple(tuple(int(bit) for bit in mapping) for mapping in mappings),
                    "mapping_names": [mapping_name(mapping) for mapping in mappings],
                    "macro_labels": macro_labels,
                    "block_state_labels": block_state_labels,
                    "mapped_block_labels": mapped_block_labels,
                }
            )
    return catalog


CANDIDATE_CATALOG = build_candidate_catalog()
HOEL_LABELS = labels_for_candidate(HOEL_GROUPS, HOEL_MAPPINGS)


def pair_state_marginal(target_bit: int, q_off: float) -> np.ndarray:
    if int(target_bit) == 1:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return np.array(
        [
            (1.0 - q_off) ** 2,
            q_off * (1.0 - q_off),
            q_off * (1.0 - q_off),
            q_off**2,
        ],
        dtype=float,
    )


def comonotone_coupling(left_probs: np.ndarray, right_probs: np.ndarray) -> np.ndarray:
    left_remaining = np.asarray(left_probs, dtype=float).copy()
    right_remaining = np.asarray(right_probs, dtype=float).copy()
    joint = np.zeros((len(left_remaining), len(right_remaining)), dtype=float)
    left_index = 0
    right_index = 0
    while left_index < len(left_remaining) and right_index < len(right_remaining):
        mass = min(float(left_remaining[left_index]), float(right_remaining[right_index]))
        joint[left_index, right_index] += mass
        left_remaining[left_index] -= mass
        right_remaining[right_index] -= mass
        if left_remaining[left_index] <= 1e-15:
            left_index += 1
        if right_remaining[right_index] <= 1e-15:
            right_index += 1
    return joint


def build_shared_noise_system_tpm(q_off: float, shared_noise_gamma: float) -> np.ndarray:
    system_tpm = np.zeros((len(ALL_MICRO_STATES), len(ALL_MICRO_STATES)), dtype=float)
    for row_index, state in enumerate(ALL_MICRO_STATES):
        a_t, b_t, c_t, d_t = (int(bit) for bit in state.tolist())
        left_target = int(c_t == 1 and d_t == 1)
        right_target = int(a_t == 1 and b_t == 1)
        left_pair_marginal = pair_state_marginal(left_target, q_off)
        right_pair_marginal = pair_state_marginal(right_target, q_off)
        joint_pair_distribution = (
            (1.0 - shared_noise_gamma) * np.outer(left_pair_marginal, right_pair_marginal)
            + shared_noise_gamma * comonotone_coupling(left_pair_marginal, right_pair_marginal)
        )

        row = np.zeros(len(ALL_MICRO_STATES), dtype=float)
        for left_state_id in range(4):
            for right_state_id in range(4):
                next_bits = STATE_BITS_BY_SIZE[2][left_state_id] + STATE_BITS_BY_SIZE[2][right_state_id]
                row[MICRO_STATE_TO_INDEX[next_bits]] += float(
                    joint_pair_distribution[left_state_id, right_state_id]
                )
        row /= row.sum()
        system_tpm[row_index] = row
    return system_tpm


def build_hoel_fig2_micro_tpm() -> np.ndarray:
    return build_shared_noise_system_tpm(HOEL_Q_OFF, HOEL_SHARED_NOISE_GAMMA)


def build_hoel_micro_mechanism_system_tpm(q_off: float) -> np.ndarray:
    return build_shared_noise_system_tpm(q_off, HOEL_SHARED_NOISE_GAMMA)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and math.isclose(
            float(sorted_values[end + 1]),
            float(sorted_values[start]),
            rel_tol=RANK_TIE_REL_TOL,
            abs_tol=RANK_TIE_ABS_TOL,
        ):
            end += 1
        ranks[order[start : end + 1]] = 0.5 * (start + end) + 1.0
        start = end + 1
    return ranks


def pearson_corr(values_x: np.ndarray, values_y: np.ndarray) -> float:
    values_x = np.asarray(values_x, dtype=float)
    values_y = np.asarray(values_y, dtype=float)
    x_centered = values_x - values_x.mean()
    y_centered = values_y - values_y.mean()
    denominator = math.sqrt(float(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator <= 1e-15:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denominator)


def spearman_corr(values_x: np.ndarray, values_y: np.ndarray) -> float:
    return pearson_corr(rankdata_average(values_x), rankdata_average(values_y))


def fit_line(values_x: np.ndarray, values_y: np.ndarray) -> tuple[float, float, float]:
    values_x = np.asarray(values_x, dtype=float)
    values_y = np.asarray(values_y, dtype=float)
    if len(values_x) < 2 or float(np.std(values_x)) <= 1e-15:
        return 0.0, float(values_y.mean()), 0.0
    slope, intercept = np.polyfit(values_x, values_y, deg=1)
    predicted = slope * values_x + intercept
    ss_res = float(np.sum((values_y - predicted) ** 2))
    ss_tot = float(np.sum((values_y - values_y.mean()) ** 2))
    r_squared = 0.0 if ss_tot <= 1e-15 else max(0.0, 1.0 - ss_res / ss_tot)
    return float(slope), float(intercept), float(r_squared)


def term_value(candidate_row: dict[str, object], term: str) -> float:
    if term == "neg_syn_micro":
        return -float(candidate_row["syn_micro"])
    if term == "neg_loss_sum":
        return -float(candidate_row["loss_sum"])
    if term == "syn_macro":
        return float(candidate_row["syn_macro"])
    if term == "macro_to_full_ei":
        return float(candidate_row["macro_to_full_ei"])
    raise KeyError(term)


def compute_fit_summary(
    candidate_rows: list[dict[str, object]],
    system_name: str,
    family: str,
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    target_values = np.array(
        [float(row[CORRELATION_TARGET_TERM]) for row in candidate_rows],
        dtype=float,
    )
    fit_summary: dict[str, dict[str, float]] = {}
    fit_rows: list[dict[str, object]] = []
    for term in CORRELATION_TERM_ORDER:
        x_values = np.array([term_value(row, term) for row in candidate_rows], dtype=float)
        rho = float(spearman_corr(x_values, target_values))
        slope, intercept, r_squared = fit_line(x_values, target_values)
        fit_summary[term] = {
            "spearman_rho": rho,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "target": CORRELATION_TARGET_TERM,
        }
        fit_rows.append(
            {
                "system": system_name,
                "family": family,
                "term": term,
                "target": CORRELATION_TARGET_TERM,
                "spearman_rho": rho,
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
            }
        )
    return fit_summary, fit_rows


def candidate_sort_key(row: dict[str, object]) -> tuple[float, float, int, str, str]:
    return (
        -float(row["macro_ei"]),
        -float(row["macro_to_full_ei"]),
        -int(bool(row["planted"])),
        str(row["partition_name"]),
        str(row["mapping_strings"]),
    )


def evaluate_candidate_space_system(
    *,
    name: str,
    display_label: str,
    family: str,
    family_label: str,
    system_tpm: np.ndarray,
    planted_groups: tuple[tuple[int, ...], tuple[int, ...]],
    planted_mappings: tuple[tuple[int, ...], tuple[int, ...]],
    q_off: float,
    shared_noise_gamma: float,
    include_candidate_rows: bool = True,
    include_candidate_export_rows: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    planted_labels = labels_for_candidate(planted_groups, planted_mappings)
    planted_signature = {"groups": planted_groups, "mappings": planted_mappings}
    micro_ei = float(effective_information_from_tpm(system_tpm))

    candidate_rows: list[dict[str, object]] = []
    for candidate in CANDIDATE_CATALOG:
        macro_tpm = coarse_grain_tpm_by_state_labels(
            system_tpm,
            input_state_labels=candidate["macro_labels"],
            output_state_labels=candidate["macro_labels"],
        )
        macro_ei = float(effective_information_from_tpm(macro_tpm))
        macro_source_tpm = average_rows_by_label(system_tpm, candidate["macro_labels"])
        macro_to_full_ei = float(effective_information_from_tpm(macro_source_tpm))

        block_eis = [
            float(effective_information_from_tpm(average_rows_by_label(system_tpm, labels)))
            for labels in candidate["block_state_labels"]
        ]
        # Compute EI(Z_i -> X+) under the same uniform intervention used for EI(Z -> X+):
        # start from the Z->X+ TPM, then marginalize over the other Z variable.
        macro_component_labels = (
            np.array([0, 0, 1, 1], dtype=int),
            np.array([0, 1, 0, 1], dtype=int),
        )
        mapped_block_eis = [
            float(effective_information_from_tpm(average_rows_by_label(macro_source_tpm, labels)))
            for labels in macro_component_labels
        ]
        syn_micro = float(micro_ei - sum(block_eis))
        loss_sum = float(
            sum(
                block_ei - mapped_block_ei
                for block_ei, mapped_block_ei in zip(block_eis, mapped_block_eis)
            )
        )
        syn_macro = float(macro_to_full_ei - sum(mapped_block_eis))
        residual = float(macro_to_full_ei - (micro_ei - syn_micro - loss_sum + syn_macro))
        candidate_rows.append(
            {
                "groups": candidate["groups"],
                "partition_name": candidate["partition_name"],
                "mappings": candidate["mappings"],
                "mapping_strings": tuple(candidate["mapping_names"]),
                "macro_ei": macro_ei,
                "macro_to_full_ei": macro_to_full_ei,
                "micro_ei": micro_ei,
                "syn_micro": syn_micro,
                "loss_sum": loss_sum,
                "syn_macro": syn_macro,
                "decomposition_residual": residual,
                "planted": bool(
                    candidate["groups"] == planted_signature["groups"]
                    and candidate["mappings"] == planted_signature["mappings"]
                ),
            }
        )

    candidate_rows.sort(key=candidate_sort_key)
    best_candidate = dict(candidate_rows[0])
    fit_summary, fit_rows = compute_fit_summary(candidate_rows, name, family)
    mean_abs_rho = float(
        np.mean([abs(float(fit_summary[term]["spearman_rho"])) for term in CORRELATION_TERM_ORDER])
    )

    system_row = {
        "name": name,
        "display_label": display_label,
        "family": family,
        "family_label": family_label,
        "q_off": q_off,
        "shared_noise_gamma": shared_noise_gamma,
        "planted_groups": planted_groups,
        "planted_mappings": planted_mappings,
        "micro_ei": micro_ei,
        "candidate_count": len(candidate_rows),
        "best_candidate": best_candidate,
        "fit_summary": fit_summary,
        "mean_abs_rho": mean_abs_rho,
    }
    if include_candidate_rows:
        system_row["candidate_rows"] = candidate_rows

    candidate_export_rows: list[dict[str, object]] = []
    if include_candidate_export_rows:
        candidate_export_rows = [
            {
                "system": name,
                "q_off": q_off,
                "shared_noise_gamma": shared_noise_gamma,
                "partition": row["partition_name"],
                "mapping_1": row["mapping_strings"][0],
                "mapping_2": row["mapping_strings"][1],
                "macro_ei": float(row["macro_ei"]),
                "macro_to_full_ei": float(row["macro_to_full_ei"]),
                "syn_micro": float(row["syn_micro"]),
                "loss_sum": float(row["loss_sum"]),
                "syn_macro": float(row["syn_macro"]),
                "decomposition_residual": float(row["decomposition_residual"]),
                "planted": int(bool(row["planted"])),
            }
            for row in candidate_rows
        ]
    return system_row, fit_rows, candidate_export_rows


def summarize_family_system_row(system_row: dict[str, object]) -> dict[str, object]:
    best_candidate = dict(system_row["best_candidate"])
    fit_summary = dict(system_row["fit_summary"])
    return {
        "system_id": str(system_row["name"]),
        "q_off": float(system_row["q_off"]),
        "micro_ei": float(system_row["micro_ei"]),
        "candidate_count": int(system_row["candidate_count"]),
        "best_partition": str(best_candidate["partition_name"]),
        "best_mapping_1": str(best_candidate["mapping_strings"][0]),
        "best_mapping_2": str(best_candidate["mapping_strings"][1]),
        "best_macro_ei": float(best_candidate["macro_ei"]),
        "best_macro_to_full_ei": float(best_candidate["macro_to_full_ei"]),
        "best_planted": int(bool(best_candidate["planted"])),
        "neg_syn_micro": float(fit_summary["neg_syn_micro"]["spearman_rho"]),
        "neg_loss_sum": float(fit_summary["neg_loss_sum"]["spearman_rho"]),
        "syn_macro": float(fit_summary["syn_macro"]["spearman_rho"]),
        "mean_abs_rho": float(system_row["mean_abs_rho"]),
    }


def build_hoel_fig2_toy_system_row() -> dict[str, object]:
    system_row, _fit_rows, _candidate_export_rows = evaluate_candidate_space_system(
        name=HOEL_SYSTEM_ID,
        display_label="Hoel Figure 2",
        family="hoel_fig2",
        family_label="Hoel Figure 2 toy example",
        system_tpm=build_hoel_fig2_micro_tpm(),
        planted_groups=HOEL_GROUPS,
        planted_mappings=HOEL_MAPPINGS,
        q_off=HOEL_Q_OFF,
        shared_noise_gamma=HOEL_SHARED_NOISE_GAMMA,
    )
    return system_row


def evaluate_fixed_hoel_decomposition(q_off: float, shared_noise_gamma: float) -> dict[str, object]:
    system_tpm = build_shared_noise_system_tpm(q_off, shared_noise_gamma)
    micro_ei = float(effective_information_from_tpm(system_tpm))
    macro_tpm = coarse_grain_tpm_by_state_labels(
        system_tpm,
        input_state_labels=HOEL_LABELS,
        output_state_labels=HOEL_LABELS,
    )
    macro_ei = float(effective_information_from_tpm(macro_tpm))
    macro_source_tpm = average_rows_by_label(system_tpm, HOEL_LABELS)
    macro_to_full_ei = float(effective_information_from_tpm(macro_source_tpm))
    block_eis = [
        float(effective_information_from_tpm(average_rows_by_label(system_tpm, labels)))
        for labels in [BLOCK_STATE_LABELS[(0, 1)], BLOCK_STATE_LABELS[(2, 3)]]
    ]
    mapped_block_eis = [
        float(effective_information_from_tpm(average_rows_by_label(macro_source_tpm, labels)))
        for labels in (
            np.array([0, 0, 1, 1], dtype=int),
            np.array([0, 1, 0, 1], dtype=int),
        )
    ]
    syn_micro = float(micro_ei - sum(block_eis))
    loss_sum = float(
        sum(block_ei - mapped_block_ei for block_ei, mapped_block_ei in zip(block_eis, mapped_block_eis))
    )
    syn_macro = float(macro_to_full_ei - sum(mapped_block_eis))
    residual = float(macro_to_full_ei - (micro_ei - syn_micro - loss_sum + syn_macro))
    return {
        "system_id": f"shared_noise_q{int(round(q_off * 100)):02d}_g{int(round(shared_noise_gamma * 100)):03d}",
        "q_off": float(q_off),
        "gamma": float(shared_noise_gamma),
        "micro_ei": micro_ei,
        "macro_ei": macro_ei,
        "macro_to_full_ei": macro_to_full_ei,
        "syn_micro": syn_micro,
        "loss_sum": loss_sum,
        "syn_macro": syn_macro,
        "decomposition_residual": residual,
    }


def build_family_rows(
    q_off_values: tuple[float, ...] = FAMILY_Q_OFF_VALUES,
) -> list[dict[str, object]]:
    family_rows: list[dict[str, object]] = []
    for q_off in q_off_values:
        system_row, _fit_rows, _candidate_export_rows = evaluate_candidate_space_system(
            name=f"hoel_micro_q{int(round(q_off * 100)):02d}",
            display_label="Hoel Figure 2 micro-mechanism family system",
            family="hoel_micro_mechanism_family",
            family_label="Hoel Figure 2 micro-mechanism family",
            system_tpm=build_hoel_micro_mechanism_system_tpm(q_off),
            planted_groups=HOEL_GROUPS,
            planted_mappings=HOEL_MAPPINGS,
            q_off=q_off,
            shared_noise_gamma=HOEL_SHARED_NOISE_GAMMA,
            include_candidate_rows=False,
            include_candidate_export_rows=False,
        )
        family_rows.append(summarize_family_system_row(system_row))
    return family_rows


def compute_family_rho_rows(
    family_rows: list[dict[str, object]],
) -> list[dict[str, float]]:
    return [
        {
            "system_id": str(row["system_id"]),
            "q_off": float(row["q_off"]),
            "candidate_count": int(row["candidate_count"]),
            "neg_syn_micro": float(row["neg_syn_micro"]),
            "neg_loss_sum": float(row["neg_loss_sum"]),
            "syn_macro": float(row["syn_macro"]),
            "mean_abs_rho": float(row["mean_abs_rho"]),
        }
        for row in family_rows
    ]


def compute_family_term_stats(family_rho_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    term_stats: dict[str, dict[str, float]] = {}
    for term in CORRELATION_TERM_ORDER:
        rho_values = np.array([float(row[term]) for row in family_rho_rows], dtype=float)
        term_stats[term] = {
            "mean_rho": float(np.mean(rho_values)),
            "std": float(np.std(rho_values, ddof=0)),
            "min": float(rho_values.min()),
            "max": float(rho_values.max()),
        }
    return term_stats


def render_html_table(title: str, columns: list[str], rows: list[list[object]]) -> str:
    pieces = [
        f"<h4 style='margin:12px 0 8px 0;'>{title}</h4>",
        "<table style='border-collapse:collapse;font-size:13px;'>",
        "<thead><tr>",
    ]
    for column in columns:
        pieces.append(
            f"<th style='border-bottom:1px solid #bbb;padding:6px 10px;text-align:left;'>{column}</th>"
        )
    pieces.append("</tr></thead><tbody>")
    for row in rows:
        pieces.append("<tr>")
        for value in row:
            pieces.append(
                f"<td style='border-bottom:1px solid #e5e5e5;padding:6px 10px;'>{value}</td>"
            )
        pieces.append("</tr>")
    pieces.append("</tbody></table>")
    return "".join(pieces)


def build_summary_html(summary: dict[str, object]) -> str:
    rows = [
        ["Hoel baseline q_off", f"{float(summary['hoel_q_off']):.3f}"],
        ["Family systems", f"{int(summary['n_family_systems'])}"],
        ["Per-term rho points", f"{int(summary['n_family_rho_points'])}"],
        ["q_off slices", f"{int(summary['n_family_q_off_slices'])}"],
        ["q_off range", f"{float(summary['family_q_off_values'][0]):.2f} to {float(summary['family_q_off_values'][-1]):.2f}"],
        ["2+2 candidate count", f"{int(summary['candidate_count_per_system'])}"],
    ]
    return render_html_table("Hoel Figure 2 micro-mechanism family", ["Metric", "Value"], rows)


def build_hoel_example_html(hoel_example_row: dict[str, object]) -> str:
    rows = [
        ["micro EI", f"{float(hoel_example_row['micro_ei']):.3f}"],
        ["best macro EI", f"{float(hoel_example_row['best_candidate']['macro_ei']):.3f}"],
        ["candidate count", f"{int(hoel_example_row['candidate_count'])}"],
        ["q_off", f"{float(hoel_example_row['q_off']):.3f}"],
    ]
    return render_html_table("Hoel Figure 2 toy example", ["Metric", "Value"], rows)


def build_family_stats_html(family_term_stats: dict[str, dict[str, float]]) -> str:
    rows = [
        [
            TERM_LABELS[term],
            f"{float(stats['mean_rho']):.3f}",
            f"{float(stats['std']):.6f}",
            f"{float(stats['min']):.6f}",
            f"{float(stats['max']):.6f}",
        ]
        for term, stats in family_term_stats.items()
    ]
    return render_html_table(
        "Family-level decomposition statistics",
        ["term", "mean rho", "std", "min", "max"],
        rows,
    )


def figure_formatter() -> FuncFormatter:
    return FuncFormatter(lambda value, _position: f"{float(value):.3f}")


def style_axes(ax) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3)
    ax.xaxis.set_major_formatter(figure_formatter())
    ax.yaxis.set_major_formatter(figure_formatter())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def clean_svg_text(svg_text: str) -> str:
    return re.sub(r"<metadata>.*?</metadata>\s*", "", svg_text, flags=re.DOTALL)


def render_hoel_example_scatter_svg(system_row: dict[str, object]) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    target_values = np.array(
        [float(row[CORRELATION_TARGET_TERM]) for row in system_row["candidate_rows"]],
        dtype=float,
    )
    for axis, term in zip(axes.flat, CORRELATION_TERM_ORDER):
        x_values = np.array([term_value(row, term) for row in system_row["candidate_rows"]], dtype=float)
        fit = system_row["fit_summary"][term]
        axis.scatter(x_values, target_values, s=14, color="#6b7280", alpha=0.45, linewidths=0)
        x_min = float(x_values.min())
        x_max = float(x_values.max())
        line_x = np.array([x_min - 0.1, x_max + 0.1], dtype=float) if x_max - x_min <= 1e-12 else np.linspace(x_min, x_max, 120)
        line_y = float(fit["slope"]) * line_x + float(fit["intercept"])
        axis.plot(line_x, line_y, color="#111827", linewidth=1.4)
        axis.set_xlabel(TERM_LABELS[term], fontsize=10)
        axis.set_ylabel(TERM_LABELS[CORRELATION_TARGET_TERM], fontsize=10)
        style_axes(axis)
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    return clean_svg_text(buffer.getvalue())


def render_family_distribution_svg(family_rho_rows: list[dict[str, float]]) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    terms = CORRELATION_TERM_ORDER
    series = [
        np.array(
            [float(row[term]) for row in family_rho_rows],
            dtype=float,
        )
        for term in terms
    ]
    positions = np.arange(len(terms), dtype=float)
    box = ax.boxplot(
        series,
        positions=positions,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111827", "linewidth": 1.2},
        whiskerprops={"color": "#4b5563", "linewidth": 0.9},
        capprops={"color": "#4b5563", "linewidth": 0.9},
        boxprops={"facecolor": "#dbe4ea", "edgecolor": "none"},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.9)

    for position, term, values in zip(positions, terms, series):
        if len(values) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(
            position + jitter,
            values,
            s=16,
            color=FAMILY_TERM_COLORS[term],
            alpha=0.55,
            linewidths=0,
        )
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set_ylabel("Spearman rho with EI(Z->X+)", fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels([TERM_LABELS[term] for term in terms], fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3)
    ax.yaxis.set_major_formatter(figure_formatter())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    return clean_svg_text(buffer.getvalue())


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def load_cached_payload(cache_file: Path, cache_version: str) -> dict[str, object] | None:
    if not cache_file.exists():
        return None
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    if payload.get("cache_version") != cache_version:
        return None
    return payload


def build_summary_payload(family_rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "hoel_q_off": HOEL_Q_OFF,
        "family_q_off_values": list(FAMILY_Q_OFF_VALUES),
        "n_family_systems": len(family_rows),
        "n_family_rho_points": len(family_rows),
        "n_family_q_off_slices": len(FAMILY_Q_OFF_VALUES),
        "candidate_count_per_system": len(CANDIDATE_CATALOG),
        "pair_partition_count": len(PAIR_PARTITIONS),
        "mapping_count_by_block_size": {2: len(REPRESENTATIVE_MAPPINGS_BY_SIZE[2])},
    }


def write_cache_artifacts(
    cache_dir: Path,
    cache_version: str,
    summary: dict[str, object],
    hoel_example_row: dict[str, object],
    family_rows: list[dict[str, object]],
    family_rho_rows: list[dict[str, float]],
    hoel_candidate_rows: list[dict[str, object]],
) -> None:
    cache_file = cache_dir / "hoel_micro_mechanism_family_summary.json"
    family_csv_file = cache_dir / "family_system_metrics.csv"
    family_rho_csv_file = cache_dir / "family_system_rho_metrics.csv"
    hoel_candidate_csv_file = cache_dir / "hoel_candidate_metric_rows.csv"

    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": cache_version,
        "summary": summary,
        "hoel_example": hoel_example_row,
        "family_rows": family_rows,
        "family_rho_rows": family_rho_rows,
    }
    cache_file.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")

    with family_csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system_id",
                "q_off",
                "micro_ei",
                "candidate_count",
                "best_partition",
                "best_mapping_1",
                "best_mapping_2",
                "best_macro_ei",
                "best_macro_to_full_ei",
                "best_planted",
                "neg_syn_micro",
                "neg_loss_sum",
                "syn_macro",
                "mean_abs_rho",
            ],
        )
        writer.writeheader()
        for row in family_rows:
            writer.writerow({key: to_jsonable(value) for key, value in row.items()})

    with family_rho_csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system_id",
                "q_off",
                "candidate_count",
                "neg_syn_micro",
                "neg_loss_sum",
                "syn_macro",
                "mean_abs_rho",
            ],
        )
        writer.writeheader()
        for row in family_rho_rows:
            writer.writerow({key: to_jsonable(value) for key, value in row.items()})

    with hoel_candidate_csv_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system",
                "q_off",
                "shared_noise_gamma",
                "partition",
                "mapping_1",
                "mapping_2",
                "macro_ei",
                "macro_to_full_ei",
                "syn_micro",
                "loss_sum",
                "syn_macro",
                "decomposition_residual",
                "planted",
            ],
        )
        writer.writeheader()
        for row in hoel_candidate_rows:
            writer.writerow({key: to_jsonable(value) for key, value in row.items()})


def write_result_artifacts(
    results_dir: Path,
    hoel_example_scatter_svg: str,
    family_distribution_svg: str,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "hoel_fig2_toy_example_scatter.svg").write_text(
        hoel_example_scatter_svg,
        encoding="utf-8",
    )
    (results_dir / "family_system_rho_distribution.svg").write_text(
        family_distribution_svg,
        encoding="utf-8",
    )


def run_statistical_reverse_experiment(
    cache_dir: Path | None = None,
    results_dir: Path | None = None,
    cache_version: str = DEFAULT_CACHE_VERSION,
    refresh_cache: bool = False,
    write_artifacts: bool = True,
) -> dict[str, object]:
    cache_dir = DEFAULT_CACHE_DIR if cache_dir is None else Path(cache_dir)
    results_dir = DEFAULT_RESULTS_DIR if results_dir is None else Path(results_dir)
    cache_file = cache_dir / "hoel_micro_mechanism_family_summary.json"

    cached_payload = None if refresh_cache else load_cached_payload(cache_file, cache_version)
    if cached_payload is None:
        hoel_example_row, _fit_rows, hoel_candidate_rows = evaluate_candidate_space_system(
            name=HOEL_SYSTEM_ID,
            display_label="Hoel Figure 2",
            family="hoel_fig2",
            family_label="Hoel Figure 2 toy example",
            system_tpm=build_hoel_fig2_micro_tpm(),
            planted_groups=HOEL_GROUPS,
            planted_mappings=HOEL_MAPPINGS,
            q_off=HOEL_Q_OFF,
            shared_noise_gamma=HOEL_SHARED_NOISE_GAMMA,
        )
        family_rows = build_family_rows()
        family_rho_rows = compute_family_rho_rows(family_rows)
        summary = build_summary_payload(family_rows)
        if write_artifacts:
            write_cache_artifacts(
                cache_dir=cache_dir,
                cache_version=cache_version,
                summary=summary,
                hoel_example_row=hoel_example_row,
                family_rows=family_rows,
                family_rho_rows=family_rho_rows,
                hoel_candidate_rows=hoel_candidate_rows,
            )
    else:
        summary = dict(cached_payload["summary"])
        summary["mapping_count_by_block_size"] = {
            int(key): int(value)
            for key, value in dict(summary["mapping_count_by_block_size"]).items()
        }
        hoel_example_row = dict(cached_payload["hoel_example"])
        family_rows = list(cached_payload["family_rows"])
        family_rho_rows = list(cached_payload["family_rho_rows"])

    family_term_stats = compute_family_term_stats(family_rho_rows)
    summary_html = build_summary_html(summary)
    hoel_example_html = build_hoel_example_html(hoel_example_row)
    family_stats_html = build_family_stats_html(family_term_stats)
    hoel_example_scatter_svg = render_hoel_example_scatter_svg(hoel_example_row)
    family_distribution_svg = render_family_distribution_svg(family_rho_rows)

    if write_artifacts:
        write_result_artifacts(results_dir, hoel_example_scatter_svg, family_distribution_svg)

    return {
        "RQ3_RESULTS": summary,
        "HOEL_EXAMPLE_ROW": hoel_example_row,
        "FAMILY_SYSTEM_ROWS": family_rows,
        "FAMILY_RHO_ROWS": family_rho_rows,
        "FAMILY_TERM_STATS": family_term_stats,
        "hoel_example_scatter_svg": hoel_example_scatter_svg,
        "family_distribution_svg": family_distribution_svg,
        "summary_html": summary_html,
        "hoel_example_html": hoel_example_html,
        "family_stats_html": family_stats_html,
        "CACHE_DIR": cache_dir,
        "RESULTS_DIR": results_dir,
        "CACHE_FILE": cache_dir / "hoel_micro_mechanism_family_summary.json",
        "FAMILY_CSV_FILE": cache_dir / "family_system_metrics.csv",
        "FAMILY_RHO_CSV_FILE": cache_dir / "family_system_rho_metrics.csv",
        "HOEL_CANDIDATE_CSV_FILE": cache_dir / "hoel_candidate_metric_rows.csv",
    }
