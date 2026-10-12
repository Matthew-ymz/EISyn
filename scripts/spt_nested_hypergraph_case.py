"""Nine-node exact nested 3/4/5-hyperedge benchmark; no continuous-model refit.

Four primitive blocks have sizes 3,2,2,2. Three parent matchings place the triad
inside a different five-node hyperedge; the remaining pairs form a four-node
hyperedge. A nine-node root couples both. Each node has four state bits:
driver, primitive readout, parent readout/share, root readout/share.
All 36 current bits receive independent uniform interventions, target is the
complete future 36-bit state, horizon=1. Primitive noise=.25, parent/root=.1.
Copied and secret-shared readouts use the same seven noisy latent parities.
Shares use 4+3+8=15 independent uniform masks. Unused current readouts, fresh
future drivers and masks integrate out exactly. EI is unchanged by this lossless
target reduction; it is not estimated against a different target.

Canonical unconstrained exact SPT is reused, without supplying planted splits.
Grade the six planted nonroot supports. Every method's proper refinements of the
primitive triad are ignored because they have no identifiable binary truth.
Retain raw Syn, audit all 9,330 subset splits at tolerance=1e-10 bits.

Authority rechecked 2026-10-12: Zotero P6UJCVG8, DXGC7JEA (19 pages), MWIWKSVG
(28 pages); Methods (5)-(12), SI S5. No explicit manuscript version/date.
This is a synthetic development case, not evidence of real-world prevalence.
"""
from __future__ import annotations

import itertools
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.spt_multilevel_parity_case import MATCHINGS, TOLERANCE, analyze

SOURCES = tuple(range(1, 10))
BOTTOM = ((1, 2, 3), (4, 5), (6, 7), (8, 9))


def run():
    records = [analyze(m, mode, bottom=BOTTOM, sources=SOURCES)
               for mode in ("copied", "secret_shared") for m in MATCHINGS]
    for r in records:
        r["planted_bottom"] = BOTTOM
        r["evaluation"] = "all nonroot clades except proper refinements of primitive parity blocks"
    differences = {kind: max(abs(r[kind][i][j] - records[0][kind][i][j])
                            for r in records for i, j in itertools.product(range(9), repeat=2))
                   for kind in ("pair_phi_r", "pair_peid")}
    secret = records[3:]
    future_difference = max(abs(r["pair_future_mi"][i][j] - secret[0]["pair_future_mi"][i][j])
                            for r in secret for i, j in itertools.product(range(9), repeat=2))
    if max(*differences.values(), future_difference) > TOLERANCE:
        raise ArithmeticError("Pair matrix invariance failed")
    return {"protocol": {"nodes": 9, "bits_per_node": 4,
        "source": "independent uniform interventions on all 36 current state bits",
        "target": "complete future 36-bit state including independent fresh drivers", "horizon": 1,
        "primitive_supports": BOTTOM, "parent_orders": [5, 4], "root_order": 9,
        "primitive_noise": .25, "parent_noise": .1, "root_noise": .1,
        "estimator": "exact rational latent joint; uniform gauge/mask elimination is lossless",
        "sharing_masks": 15, "syn_tolerance_bits": TOLERANCE,
        "clade_scoring": "ignore only proper refinements of primitive parity blocks, for every method",
        "query_policy": "free canonical exact SPT; planted supports are not split constraints",
        "scope": "synthetic development example; not a held-out or finite-sample comparison"},
        "case_count": 6, "recovered_root_count": sum(r["correct_root"] for r in records),
        "pair_matrix_max_difference_bits": differences,
        "secret_shared_future_pair_max_difference_bits": future_difference,
        "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, help="Generate the full comparison figure")
    args = parser.parse_args()
    result = run()
    if args.figure:
        from scripts.plot_spt_hierarchy_comparison import plot
        plot(result["records"], args.figure)
    print(json.dumps(result, indent=2))
