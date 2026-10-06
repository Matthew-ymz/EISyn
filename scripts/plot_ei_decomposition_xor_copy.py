#!/usr/bin/env python3
"""Draw exact XOR/copy examples using the designated single-end EI definition.

Definition source (read 2026-10-06):
  docs/ref/1对n的EI分解_最小耦合方案.md, Eqs. (1-1)--(1-8), Sections 2.1--2.2.
Fresh Zotero check:
  parent P6UJCVG8, verified title: Emergent hierarchical organization of
  causal interactions in complex systems; children DXGC7JEA (main, 19 pages)
  and MWIWKSVG (supplement, 28 pages). Read main Methods pp. 15--16,
  Eqs. (5)--(8), supplement S1.1, S2 and S3.1, Eqs. (S20)--(S36).
  Neither attachment supplies an explicit manuscript date/version. Metadata
  dates alone do not resolve manuscript version order. The figure directly
  uses R(U,V;W)=C_ext(U,V;W), with W as the only processed endpoint. The
  manuscript's full-LC axiom package is not asserted for this information quantity.

One-step binary mechanisms, uniform intervention, all information in bits.
No EI estimator or numerical channel optimizer: rational tables and integer
base-2 logarithms give exact values. A constant channel attains the pair-MI
upper bound in XOR; an identity channel attains it in copy. These bounds certify
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
        # The third position is the sole processing endpoint:
        # Both use (U,V;W); W is the target in XOR and the source in copy.
        processed = {state + ((0 if name == "xor" else state[2]),): p
                     for state, p in table.items()}
        cmi = conditional_information(processed, (0,), (1,), (3,))
        upper = mutual_information(table, (0,), (1,))
        redundancy = upper - cmi
        if redundancy != upper:
            raise AssertionError("Single-end witness did not attain the pair-MI upper bound.")
        un_u, un_v = pair[1] - redundancy, pair[2] - redundancy
        syn = joint - pair[1] - pair[2] + redundancy
        if syn < 0:
            raise ValueError(f"Syn minimum={syn} bit; threshold=0 bit; affected count=1")
        expected = (0, 0, 0, 1) if name == "xor" else (1, 0, 0, 0)
        assert (redundancy, un_u, un_v, syn) == expected
        assert joint == redundancy + un_u + un_v + syn == 1
        checked[name] = dict(pair=pair, joint=joint, cmi=original_cmi,
                             extractable=redundancy, atoms=expected)
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
        for x, label, title, accent in ((64, "a", "2→1：XOR 联合输入", PURPLE),
                                        (850, "b", "1→2：复制广播输出", TEAL)):
            text(x, 867, label, size=17, weight="bold", color=accent)
            text(x + 35, 867, title, size=17, chinese=True)
        text(64, 825, r"$U,V\overset{\mathrm{iid}}{\sim}\mathrm{Bern}(1/2),\quad W=U\oplus V$", size=15)
        text(850, 825, r"$W\sim\mathrm{Bern}(1/2),\qquad U=V=W$", size=15)
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
            if value == 0:
                text(x, 487, r"$I(U;W)=I(V;W)=I(U;V)=0$", size=16)
                text(x, 450, r"$I(U,V;W)=1,\qquad\mathrm{CoI}=-1$", size=16)
                text(x, 400, "单端提取：只处理目标 W", size=14, chinese=True, color=MUTED)
                text(x, 363, "可取常量通道：$Z^*=z_0$", size=16, chinese=True)
                text(x, 325, r"$I(U;V\mid Z^*)=0$", size=16)
                text(x, 288, "达到上界 $I(U;V)=0$", size=15, chinese=True, color=MUTED)
                r_formula = r"$R=C_{\mathrm{ext}}(U,V;W)=0$"
                u_formula = r"$\mathrm{Un}_{U}=\mathrm{Un}_{V}=0-0=0$"
                s_formula = r"$S=R-\mathrm{CoI}=0-(-1)=1$"
            else:
                text(x, 487, r"$I(W;U)=I(W;V)=I(U;V)=1$", size=16)
                text(x, 450, r"$I(W;U,V)=1,\qquad\mathrm{CoI}=1$", size=16)
                text(x, 400, "单端提取：只处理源 W", size=14, chinese=True, color=MUTED)
                text(x, 363, "可取恒等通道：$Z^*=W$", size=16, chinese=True)
                text(x, 325, r"$I(U;V\mid Z^*)=0$", size=16)
                text(x, 288, "达到上界 $I(U;V)=1$", size=15, chinese=True, color=MUTED)
                r_formula = r"$R=C_{\mathrm{ext}}(U,V;W)=1$"
                u_formula = r"$\mathrm{Un}_{U}=\mathrm{Un}_{V}=1-1=0$"
                s_formula = r"$S=R-\mathrm{CoI}=1-1=0$"
            text(x, 238, "冗余、特有与协同", size=12.5, chinese=True, color=MUTED)
            text(x, 199, r_formula, size=16)
            text(x, 162, u_formula, size=16)
            text(x, 125, s_formula, size=16, color=PURPLE if value == 0 else TEAL)
            line(x, x + 686, 88)
        text(407, 54, "纯协同 · 1 bit", size=19, chinese=True, color=PURPLE, ha="center")
        text(1193, 54, "纯冗余 · 1 bit", size=19, chinese=True, color=TEAL, ha="center")
        text(800, 15, "局部通道只用于提取冗余；EI 仍来自原联合分布。", size=11.5,
             color=MUTED, chinese=True, ha="center")

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
