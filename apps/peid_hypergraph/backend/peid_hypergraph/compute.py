from __future__ import annotations

import copy
import math

from .boolean import compute_boolean_graph
from .continuous import compute_continuous_graph


def compute_boolean_peid(payload: dict[str, object]) -> dict[str, object]:
    return compute_boolean_graph(payload)


def compute_continuous_peid(payload: dict[str, object]) -> dict[str, object]:
    return compute_continuous_graph(payload)


def _top_count(total: int, percent: float | None, top_k: int | None) -> int:
    if total <= 0:
        return 0
    if top_k is not None:
        return max(0, min(total, int(top_k)))
    if percent is None:
        return total
    bounded = min(100.0, max(0.0, float(percent)))
    return max(0, min(total, int(math.ceil(total * bounded / 100.0))))


def filter_visible_graph(
    result: dict[str, object],
    *,
    pairwise_top_percent: float | None = None,
    hyperedge_top_percent: float | None = None,
    pairwise_top_k: int | None = None,
    hyperedge_top_k: int | None = None,
) -> dict[str, object]:
    visible = copy.deepcopy(result)
    pairwise = sorted(
        list(result.get("pairwise_edges", [])),
        key=lambda row: float(row.get("ei", 0.0)),
        reverse=True,
    )
    hyperedges = sorted(
        list(result.get("hyperedges", [])),
        key=lambda row: float(row.get("display_value", row.get("synergy", 0.0))),
        reverse=True,
    )
    visible["pairwise_edges"] = pairwise[: _top_count(len(pairwise), pairwise_top_percent, pairwise_top_k)]
    visible["hyperedges"] = hyperedges[: _top_count(len(hyperedges), hyperedge_top_percent, hyperedge_top_k)]
    return visible
