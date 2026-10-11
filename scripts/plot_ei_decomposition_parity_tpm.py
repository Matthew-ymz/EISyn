#!/usr/bin/env python3
"""Exact finite-TPM examples of one-source/multiple-output synergy.

Fresh read, 2026-10-11: Zotero parent P6UJCVG8, title verified; main
DXGC7JEA (19/19 indexed pages), supplement MWIWKSVG (28/28 pages).
Neither supplies an explicit manuscript date/version; attachment metadata does
not resolve draft order. Relevant locations: main Methods pp.15--16 Eqs.(5)--(8),
supplement S1.1 Eqs.(S1)--(S3), S2 Eqs.(S20)--(S27). Target-side atoms use the
repository note Eqs.(1-6), (1-8), (1-25), (4-7)--(4-8), not a claimed full PID
lattice or an implementation of the manuscript's full-LC axiom package.

Rosas et al. (2020), DOI 10.1371/journal.pcbi.1008289, Example 1:
Zotero 7LWNIPLK / B3E4LB65, indexed arXiv v1 (2020-04-17), pp.2--3,
and the published full-text Example 1. Here the parity-preservation probability
is its limiting value gamma=1; the microscopic transition remains stochastic.

All cases share one uniform intervention over the complete old binary state,
one-step horizon and bits. Only marginalization changes the designated source.
No external noise variable is added. The deterministic internal-mask example
uses old state coordinates outside the designated source as internal context.

Fractions plus exact integer logarithms suffice for these uniform dyadic tables.
Constant/identity source channels attain analytic redundancy bounds, including
all stochastic channels, so no numerical optimizer/estimator/cache is required.
Syn tolerance=0 bit; report all negative and adjusted counts; never clip Syn.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from plot_ei_decomposition_xor_copy import (
    CHINESE, INK, MUTED, PURPLE, RULE, TEAL,
    conditional_information, entropy, mutual_information,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fig" / "ei_decomposition_parity_tpm.png"
SYN_TOLERANCE = Fraction(0)  # native units: bit; exact, not estimated


def parity(state):
    return sum(state) % 2


def make_tpm(n, mechanism):
    """Return binary-ordered states and row-stochastic exact probabilities."""
    states = tuple(product((0, 1), repeat=n))
    rows = []
    for old in states:
        if mechanism == "parity":
            row = tuple(Fraction(int(parity(old) == parity(new)), 2 ** (n - 1))
                        for new in states)
        elif mechanism == "internal_mask":
            future = old[1:] + (parity(old),)
            row = tuple(Fraction(int(new == future)) for new in states)
        elif mechanism == "rotation":
            future = old[1:] + old[:1]
            row = tuple(Fraction(int(new == future)) for new in states)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        if sum(row) != 1:
            raise ValueError("Invalid TPM row.")
        rows.append(row)
    return states, tuple(rows)


def intervened_table(states, tpm):
    """Joint table ordered (x_1,...,x_n,y_1,...,y_n), uniform old-state prior."""
    prior = Fraction(1, len(states))
    return {old + new: prior * tpm[i][j]
            for i, old in enumerate(states) for j, new in enumerate(states)
            if tpm[i][j]}


def exact_log2(value):
    a, b = value.numerator, value.denominator
    if a & (a - 1) or b & (b - 1):
        raise ValueError("Exact logarithm requires a power-of-two ratio.")
    return a.bit_length() - b.bit_length()


def tpm_ei(tpm):
    """Independent check using supplement Eq.(S3), base-2 logarithms."""
    m = len(tpm)
    column_sums = [sum(row[j] for row in tpm) for j in range(m)]
    return sum((p / m) * exact_log2(m * p / column_sums[j])
               for row in tpm for j, p in enumerate(row) if p)


def decompose(table, source, left, right):
    """Certify exact C_ext using attaining channels for these special cases.

    This is deliberately not a general-purpose C_ext optimizer.
    """
    pair = mutual_information(table, left, right)
    i_left = mutual_information(table, source, left)
    i_right = mutual_information(table, source, right)
    total = mutual_information(table, source, left + right)
    original_cmi = conditional_information(table, left, right, source)
    upper = min(pair, i_left, i_right)
    if pair == 0:
        redundancy = Fraction(0)  # Z=constant gives CoI=0, attaining upper=0.
        witness = "constant"
    elif original_cmi == 0:
        redundancy = pair  # Z=W gives CoI=I(left;right), attaining the bound.
        witness = "identity"
    else:
        raise ValueError("These witnesses do not certify this distribution.")
    assert redundancy == upper
    atoms = (redundancy, i_left - redundancy, i_right - redundancy,
             total - i_left - i_right + redundancy)
    if atoms[-1] < -SYN_TOLERANCE:
        raise ValueError(f"Syn minimum={atoms[-1]} bit; threshold={-SYN_TOLERANCE} bit; "
                         "affected count=1")
    assert atoms[0] + atoms[1] == i_left
    assert atoms[0] + atoms[2] == i_right
    assert sum(atoms) == total
    return {"atoms": atoms, "total": total, "witness": witness}


def conditioned(table, position, value):
    """Read a context from the same joint table; do not re-intervene."""
    selected = {state: mass for state, mass in table.items() if state[position] == value}
    normalizer = sum(selected.values())
    return {state: mass / normalizer for state, mass in selected.items()}


def verify_exact():
    checked, synergies = {}, []
    for n in (2, 3, 4):
        full_source, target = tuple(range(n)), tuple(range(n, 2 * n))
        for mechanism in ("parity", "internal_mask", "rotation"):
            states, tpm = make_tpm(n, mechanism)
            table = intervened_table(states, tpm)
            assert sum(table.values()) == 1
            # All three kernels preserve the uniform prior; deterministic ones
            # are permutations. This check is separate from row normalization.
            assert all(sum(row[j] for row in tpm) == 1 for j in range(len(states)))
            expected_full = 1 if mechanism == "parity" else n
            assert tpm_ei(tpm) == expected_full
            assert mutual_information(table, full_source, target) == expected_full
            assert entropy(table, target) == n
            single_source_ei = [mutual_information(table, (i,), target) for i in full_source]
            assert single_source_ei == ([0] * n if mechanism == "parity" else [1] * n)
            assert expected_full - sum(single_source_ei) == (1 if mechanism == "parity" else 0)
            source = (0,) if mechanism == "internal_mask" else full_source
            readout = {}
            for k in range(1, n + 1):
                for subset in combinations(target, k):
                    value = mutual_information(table, source, subset)
                    expected = k if mechanism == "rotation" else int(k == n)
                    assert value == expected
                    readout[subset] = value
            if mechanism == "parity":
                for k in range(1, n):
                    for subset in combinations(full_source, k):
                        assert mutual_information(table, subset, target) == 0
                for old in states:
                    for j, new in enumerate(states):
                        if tpm[states.index(old)][j]:
                            assert parity(old) == parity(new)
            elif mechanism == "internal_mask":
                for state in table:
                    assert parity(state[n:]) == state[0]
            else:
                assert all(parity(state[:n]) == parity(state[n:]) for state in table)

            # Enumerate every unordered, nontrivial target bipartition. They are
            # alternative decompositions, never summed as separate atoms.
            splits = []
            for mask in range(2 ** (n - 1) - 1):
                left = (target[0],) + tuple(target[i] for i in range(1, n)
                                            if mask & (1 << (i - 1)))
                right = tuple(i for i in target if i not in left)
                splits.append((left, right))
            assert len(splits) == 2 ** (n - 1) - 1
            for left, right in splits:
                result = decompose(table, source, left, right)
                expected = (0, len(left), len(right), 0) if mechanism == "rotation" else (0, 0, 0, 1)
                assert result["atoms"] == expected
                synergies.append(result["atoms"][-1])
                # With the complete deterministic old state as source, all
                # output-block synergy is zero, including the internal-mask map.
                if mechanism != "parity":
                    full = decompose(table, full_source, left, right)
                    assert full["atoms"] == (0, len(left), len(right), 0)
                    synergies.append(full["atoms"][-1])

            # The note's conditional target-tree path for three outputs.
            if n == 3 and mechanism != "rotation":
                assert mutual_information(table, source, (target[0],)) == 0
                assert conditional_information(table, source, target[1:], (target[0],)) == 1
                for bit in (0, 1):
                    result = decompose(conditioned(table, target[0], bit), source,
                                       (target[1],), (target[2],))
                    assert result["atoms"] == (0, 0, 0, 1)
                    synergies.append(result["atoms"][-1])
            checked[(n, mechanism)] = {"states": states, "tpm": tpm,
                                        "table": table, "source": source,
                                        "readout": readout}

    minimum = min(synergies)
    significant = sum(s < -SYN_TOLERANCE for s in synergies)
    tiny = sum(-SYN_TOLERANCE <= s < 0 for s in synergies)
    if significant:
        raise ValueError(f"Syn minimum={minimum} bit; threshold={-SYN_TOLERANCE} bit; "
                         f"affected count={significant}")
    assert tiny == 0
    print(f"Exact validation: n=2,3,4; three TPM families; {len(synergies)} Syn values.")
    print(f"Syn tolerance={SYN_TOLERANCE} bit; minimum={minimum}; "
          f"significant/tiny/adjusted counts={significant}/{tiny}/0.")
    print("Parity TPM: grouped source -> every proper target subset = 0 bit; all targets = 1 bit.")
    print("Internal-mask TPM: x1 -> every proper target subset = 0 bit; all targets = 1 bit.")
    print("Deterministic complete-old-state source: every target split Syn = 0 bit.")
    return checked


def draw(checked):
    style = {"font.family": "DejaVu Sans", "mathtext.fontset": "stixsans",
             "text.color": INK, "axes.labelcolor": INK,
             "xtick.color": MUTED, "ytick.color": INK, "font.size": 11}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(12.6, 8.9), facecolor="white")
        texts = []

        def text(x, y, content, size=12, color=INK, ha="left", chinese=False, weight="normal"):
            kwargs = {"fontsize": size, "color": color, "ha": ha, "va": "center",
                      "fontweight": weight}
            if chinese:
                kwargs["fontproperties"] = CHINESE
            artist = fig.text(x, y, content, **kwargs)
            texts.append(artist)
            return artist

        text(.045, .958, "一对多协同：两种无外源噪声变量的 TPM", 20, chinese=True)
        text(.955, .958, "共同均匀干预 · 一步 · bit", 10, color=MUTED, ha="right", chinese=True)
        text(.045, .902, "a", 15, weight="bold", color=PURPLE)
        text(.075, .902, "三比特奇偶保持（随机转移）", 14, chinese=True)
        text(.56, .902, "b", 15, weight="bold", color=TEAL)
        text(.59, .902, "两比特内部掩码（确定性转移）", 14, chinese=True)
        text(.075, .863, r"源 $W=\langle X_1,X_2,X_3\rangle$（完整旧状态）", 12, chinese=True)
        text(.59, .863, r"源 $W=X_1$；$Y_1=X_2,\quad Y_2=X_1\oplus X_2$", 12, chinese=True)

        def matrix_panel(rect, n, mechanism, color, grouped=False):
            case = checked[(n, mechanism)]
            states, tpm = case["states"], case["tpm"]
            order = sorted(range(len(states)), key=lambda i: (parity(states[i]), states[i])) if grouped else list(range(len(states)))
            matrix = np.array([[float(tpm[i][j]) for j in order] for i in order])
            ax = fig.add_axes(rect)
            # Color encodes allowed transitions only; exact probabilities are
            # written in cells, avoiding two incompatible probability scales.
            ax.imshow(matrix > 0, cmap=ListedColormap(["#FAFBFC", color]), vmin=0, vmax=1)
            names = ["".join(map(str, states[i])) for i in order]
            ax.set_xticks(range(len(states)), names, fontsize=10)
            ax.set_yticks(range(len(states)), names, fontsize=10)
            ax.set_xlabel(r"下一时刻 $\boldsymbol{y}$", fontproperties=CHINESE, fontsize=11, labelpad=7)
            ax.set_ylabel(r"上一时刻 $\boldsymbol{x}$", fontproperties=CHINESE, fontsize=11, labelpad=7)
            ax.set_xticks(np.arange(-.5, len(states), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(states), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=.8)
            ax.tick_params(which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for r, i in enumerate(order):
                for c, j in enumerate(order):
                    p = tpm[i][j]
                    label = (str(p.numerator) if p.denominator == 1 else f"{p.numerator}/{p.denominator}") if p else "0"
                    artist = ax.text(c, r, label, ha="center", va="center",
                                     fontsize=9 if n == 3 else 13,
                                     color="white" if p else "#8C96A0")
                    texts.append(artist)
            if grouped:
                ax.axhline(3.5, color=INK, lw=1)
                ax.axvline(3.5, color=INK, lw=1)
            return ax

        matrix_panel([.102, .433, .342, .400], 3, "parity", PURPLE, grouped=True)
        matrix_panel([.617, .433, .342, .400], 2, "internal_mask", TEAL)
        text(.273, .347, r"$Y_1\oplus Y_2\oplus Y_3=X_1\oplus X_2\oplus X_3$", 12, color=PURPLE, ha="center")
        text(.788, .347, r"$Y_1\oplus Y_2=X_1$", 13, color=TEAL, ha="center")
        text(.273, .315, "行列按偶 / 奇分组；同奇偶性后继各以 1/4 出现", 10.5, color=MUTED, ha="center", chinese=True)
        text(.788, .315, r"内部旧状态 $X_2$ 提供掩码；每行只有一个后继", 10.5, color=MUTED, ha="center", chinese=True)

        def readout_panel(rect, values, labels, color):
            ax = fig.add_axes(rect)
            ax.barh(range(3), values, height=.46, color=color)
            ax.invert_yaxis()
            ax.set_yticks(range(3), labels, fontproperties=CHINESE, fontsize=11)
            ax.set_xlim(0, 1.18)
            ax.set_xticks([0, .5, 1], ["0", "0.5", "1"], fontsize=10)
            ax.set_xlabel("EI (bit)", fontsize=11, labelpad=3)
            ax.tick_params(axis="both", length=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color(RULE)
            for i, value in enumerate(values):
                if value == 0:
                    ax.plot([0], [i], "o", color=color, ms=4, clip_on=False)
                artist = ax.text(float(value) + .04, i, str(value), va="center",
                                 color=color, fontsize=12)
                texts.append(artist)

        parity_case, mask_case = checked[(3, "parity")], checked[(2, "internal_mask")]
        readout_panel([.22, .149, .244, .127],
                      [parity_case["readout"][(3,)], parity_case["readout"][(3, 4)],
                       parity_case["readout"][(3, 4, 5)]],
                      ["任意单输出", "任意双输出", "完整三输出"], PURPLE)
        readout_panel([.735, .149, .244, .127],
                      [mask_case["readout"][(2,)], mask_case["readout"][(3,)],
                       mask_case["readout"][(2, 3)]],
                      [r"只读 $Y_1$", r"只读 $Y_2$", r"联合 $(Y_1,Y_2)$"], TEAL)
        text(.273, .069, "三输出协同 = 1 bit", 14, color=PURPLE, ha="center", chinese=True)
        text(.788, .069, "二输出协同 = 1 bit", 14, color=TEAL, ha="center", chinese=True)
        text(.5, .025, "有色格为允许的转移，格内为概率。右图若把完整旧状态作为源，输出协同为 0。",
             10, color=MUTED, ha="center", chinese=True)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in texts:
            bounds = artist.get_window_extent(renderer)
            if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
                raise RuntimeError(f"Text exceeds canvas: {artist.get_text()}")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=220, facecolor="white", bbox_inches=None)
        plt.close(fig)
    print(OUTPUT)


def main():
    draw(verify_exact())


if __name__ == "__main__":
    main()
