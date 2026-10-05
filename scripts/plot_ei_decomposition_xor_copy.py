#!/usr/bin/env python3
"""Draw the exact XOR/copy examples of the current symmetric EI candidate.

Definition source (read 2026-10-05):
  docs/ref/1对n的EI分解_最小耦合方案.md, Eqs. (1-1)--(1-4), (2-1).
Fresh Zotero check:
  parent P6UJCVG8, verified title: Emergent hierarchical organization of
  causal interactions in complex systems; children DXGC7JEA (main, 19 pages)
  and MWIWKSVG (supplement, 28 pages). Read main Methods pp. 15--16,
  Eqs. (5)--(8), supplement S1.1, S2 and S3.1, Eqs. (S20)--(S36).
  Neither attachment supplies an explicit manuscript date/version. Metadata
  dates alone do not resolve manuscript version order. This figure uses the
  Markdown's new three-position extractable-co-information candidate; it does
  not assert that the manuscript's full-LC axiom package defines this candidate.

One-step binary mechanisms, uniform intervention, all information in bits.
No EI estimator or numerical channel optimizer: rational tables and integer
base-2 logarithms give exact values. A constant channel attains the pair-MI
upper bound in XOR; identity channels attain it in copy. These bounds certify
optimality over *all* local stochastic channels, not just deterministic ones.
Syn nonnegative tolerance = 0 bit (exact arithmetic); affected count = 0.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fig" / "ei_decomposition_xor_copy"
INK = "#202832"
MUTED = "#63707D"
RULE = "#DDE2E7"
PURPLE = "#705096"
TEAL = "#167D82"
CHINESE = font_manager.FontProperties(fname="/System/Library/Fonts/STHeiti Light.ttc")


def entropy(table: dict[tuple[int, ...], Fraction], positions: tuple[int, ...]) -> Fraction:
    """Exact entropy for these dyadic tables (every nonzero mass is 2**-k)."""
    marginal: dict[tuple[int, ...], Fraction] = defaultdict(Fraction)
    for state, mass in table.items():
        marginal[tuple(state[i] for i in positions)] += mass
    result = Fraction()
    for mass in marginal.values():
        numerator, denominator = mass.numerator, mass.denominator
        if numerator != 1 or denominator & (denominator - 1):
            raise ValueError("This exact helper requires dyadic uniform marginal masses.")
        result += mass * (denominator.bit_length() - 1)
    return result


def mutual_information(table, a, b):
    return entropy(table, a) + entropy(table, b) - entropy(table, a + b)


def conditional_information(table, a, b, z):
    return (entropy(table, a + z) + entropy(table, b + z)
            - entropy(table, z) - entropy(table, a + b + z))


def exact_examples():
    tables = {
        "xor": {(u, v, u ^ v): Fraction(1, 4) for u in (0, 1) for v in (0, 1)},
        "copy": {(w, w, w): Fraction(1, 2) for w in (0, 1)},
    }
    checked = {}
    for name, table in tables.items():
        pair = [mutual_information(table, (a,), (b,)) for a, b in ((0, 1), (0, 2), (1, 2))]
        joint = mutual_information(table, (0, 1), (2,))
        original_cmi = conditional_information(table, (0,), (1,), (2,))
        extractable = []
        for a, b, t in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            # Z=constant for XOR, Z=T for copy.
            processed = {state + ((0 if name == "xor" else state[t]),): p for state, p in table.items()}
            cmi = conditional_information(processed, (a,), (b,), (3,))
            upper = mutual_information(table, (a,), (b,))
            attained = upper - cmi
            if attained != upper:
                raise AssertionError("Witness did not attain the global pair-MI bound.")
            extractable.append(attained)
        redundancy = max(extractable)
        un_u, un_v = pair[1] - redundancy, pair[2] - redundancy
        syn = joint - pair[1] - pair[2] + redundancy
        if syn < 0:
            raise ValueError(f"Syn minimum={syn} bit; threshold=0 bit; affected count=1")
        expected = (0, 0, 0, 1) if name == "xor" else (1, 0, 0, 0)
        assert (redundancy, un_u, un_v, syn) == expected
        assert joint == redundancy + un_u + un_v + syn == 1
        checked[name] = dict(pair=pair, joint=joint, cmi=original_cmi,
                             extractable=extractable, atoms=expected)
    return checked


def main():
    checked = exact_examples()
    style = {
        "font.family": "DejaVu Sans", "mathtext.fontset": "stixsans",
        "font.size": 13, "text.color": INK, "pdf.fonttype": 42,
        "svg.fonttype": "path", "axes.unicode_minus": False,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(16, 10), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 1600), ylim=(0, 1000))
        ax.set_aspect("equal")
        ax.axis("off")
        texts = []

        def text(x, y, content, size=13.5, color=INK, chinese=False, weight="normal", ha="left"):
            kw = dict(fontsize=size, color=color, ha=ha, va="center", fontweight=weight)
            if chinese:
                kw["fontproperties"] = CHINESE
            artist = ax.text(x, y, content, **kw)
            texts.append(artist)
            return artist

        def line(x0, x1, y, color=RULE, width=0.8):
            ax.plot((x0, x1), (y, y), color=color, lw=width)

        def node(x, y, name):
            ax.add_patch(Circle((x, y), 30, facecolor=INK, edgecolor="none"))
            text(x, y, f"${name}$", size=21, color="white", ha="center")

        def arrow(start, end, color):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                        lw=1.7, color=color, shrinkA=32, shrinkB=34))

        text(64, 946, "XOR 与复制的信息分解", size=25, chinese=True)
        text(1536, 946, "均匀干预 · 单位 bit", size=12.5,
             color=MUTED, chinese=True, ha="right")
        line(64, 1536, 907)
        ax.plot((800, 800), (42, 886), color=RULE, lw=0.8)

        # The two mechanisms retain the geometry of the user's sketch.
        for x, label, title, accent in ((64, "a", "XOR：联合输入", PURPLE),
                                        (850, "b", "复制：广播输出", TEAL)):
            text(x, 867, label, size=17, weight="bold", color=accent)
            text(x + 35, 867, title, size=17, chinese=True)
        text(64, 825, r"$U,V\overset{\mathrm{iid}}{\sim}\mathrm{Bern}(1/2),\qquad W=U\oplus V$", size=14)
        text(850, 825, r"$W\sim\mathrm{Bern}(1/2),\qquad U=V=W$", size=14)
        for start in ((215, 754), (215, 600)):
            arrow(start, (550, 677), PURPLE)
        for end in ((1435, 754), (1435, 600)):
            arrow((1100, 677), end, TEAL)
        for x, y, name in ((215, 754, "U"), (215, 600, "V"), (550, 677, "W"),
                            (1100, 677, "W"), (1435, 754, "U"), (1435, 600, "V")):
            node(x, y, name)
        text(389, 677, "XOR", size=17, color=PURPLE, ha="center")
        text(1260, 741, "copy", size=15, color=TEAL, ha="center")
        text(1260, 611, "copy", size=15, color=TEAL, ha="center")

        # Only the MI inputs, an attaining local channel, and the atom arithmetic
        # appear on the plate. General definitions and optimality proofs remain
        # in the source document and the exact verification above.
        for x, value in ((64, 0), (850, 1)):
            line(x, x + 686, 555)
            text(x, 525, "互信息", size=12.5, chinese=True, color=MUTED)
            text(x, 487, rf"$I(U;V)=I(U;W)=I(V;W)={value}$", size=15)
            text(x, 450, rf"$I(U,V;W)=1,\qquad\mathrm{{CoI}}={value}+{value}-1={2*value-1}$", size=15)
            text(x, 400, "单端提取", size=12.5, chinese=True, color=MUTED)
            if value == 0:
                text(x, 363, "取常量处理：$Z_W^*=Z_V^*=Z_U^*=z_0$", size=14.5, chinese=True)
            else:
                text(x, 363, "取恒等处理：$Z_W^*=W,\ Z_V^*=V,\ Z_U^*=U$", size=14.5, chinese=True)
            text(x, 325, rf"$C_{{\mathrm{{ext}}}}(U,V;W)=I(U;V)-I(U;V\mid Z_W^*)={value}-0={value}$", size=13.8)
            text(x, 288, rf"$C_{{\mathrm{{ext}}}}(U,W;V)=C_{{\mathrm{{ext}}}}(V,W;U)={value}$", size=14.5)
            text(x, 238, "冗余、特有与协同", size=12.5, chinese=True, color=MUTED)
            text(x, 199, rf"$R=\max\{{{value},{value},{value}\}}={value}$", size=15)
            text(x, 162, rf"$\mathrm{{Un}}_U=\mathrm{{Un}}_V=I(U;W)-R={value}-{value}=0$", size=15)
            coi = "(-1)" if value == 0 else "1"
            text(x, 125, rf"$\mathrm{{Syn}}=R-\mathrm{{CoI}}={value}-{coi}={1-value}$", size=15, color=PURPLE if value == 0 else TEAL)
            line(x, x + 686, 88)
        text(407, 54, "纯协同 · 1 bit", size=19, chinese=True, color=PURPLE, ha="center")
        text(1193, 54, "纯冗余 · 1 bit", size=19, chinese=True, color=TEAL, ha="center")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in texts:
            extent = artist.get_window_extent(renderer=renderer)
            if not fig.bbox.contains(extent.x0, extent.y0) or not fig.bbox.contains(extent.x1, extent.y1):
                raise RuntimeError(f"Text extends beyond the canvas: {artist.get_text()}")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT.with_suffix(".png"), dpi=200, facecolor="white", bbox_inches=None)
        fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white", bbox_inches=None)
        plt.close(fig)
    print(f"Exact results (R, Un_U, Un_V, Syn): {checked['xor']['atoms']}, {checked['copy']['atoms']}")
    print("Syn tolerance: 0 bit; negative/adjusted counts: 0/0; global optima certified analytically.")
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".svg"))


if __name__ == "__main__":
    main()
