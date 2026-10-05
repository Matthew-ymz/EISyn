#!/usr/bin/env python3
"""One-source/two-output synergy, in the concise XOR/copy figure style.

Checked 2026-10-05 against docs/ref/1对n的EI分解_最小耦合方案.md,
Eqs. (1-1)--(1-4), (2-1), and the secret-sharing example in Section 3.1.
Fresh Zotero title/children/metadata/full-text retrieval: parent P6UJCVG8,
main DXGC7JEA (19 pages), supplementary MWIWKSVG (28 pages); main Methods
p.15 Eq.(5), supplement S2 Eqs.(S20)--(S27). No explicit manuscript version
or date is supplied by either attachment. The Markdown's symmetric candidate
is used here; the manuscript's full-LC PID assumptions are not asserted.

Only W is intervened on, uniformly over {0,1}. N is an independent, uniform
background bit retained within the stochastic output kernel. One-step outputs
U=N, V=N xor W. The graph includes W->XOR->V and N->U, N->XOR, not W->U.
All information quantities below use this same intervention-induced table.

Exact rational probabilities and integer base-2 logarithms; no estimator,
channel search, clipping, or cache. Syn tolerance=0 bit, adjusted count=0.
Constant channels reach the upper bound C_ext<=I(A;B)=0 in all three positions,
which proves optimality over all stochastic local channels.
"""

from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from plot_ei_decomposition_xor_copy import (
    CHINESE, INK, MUTED, OUTPUT as PREVIOUS_OUTPUT, PURPLE, RULE,
    conditional_information, entropy, mutual_information,
)

OUTPUT = PREVIOUS_OUTPUT.parent / "ei_decomposition_random_mask"


def verify_exact():
    table = {(n, n ^ w, w): Fraction(1, 4) for w in (0, 1) for n in (0, 1)}
    assert sum(table.values()) == 1
    assert all(mutual_information(table, (a,), (b,)) == 0
               for a, b in ((0, 1), (0, 2), (1, 2)))
    assert mutual_information(table, (2,), (0, 1)) == 1
    assert conditional_information(table, (0,), (1,), (2,)) == 1
    assert entropy(table, (0, 1, 2)) - entropy(table, (0, 1)) == 0
    extractable = []
    for a, b, _ in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        witness = {state + (0,): mass for state, mass in table.items()}
        pair = mutual_information(table, (a,), (b,))
        cmi = conditional_information(witness, (a,), (b,), (3,))
        assert pair == cmi == 0
        extractable.append(pair - cmi)
    r = max(extractable)
    un_u = mutual_information(table, (2,), (0,)) - r
    un_v = mutual_information(table, (2,), (1,)) - r
    syn = mutual_information(table, (2,), (0, 1)) - un_u - un_v - r
    if syn < 0:
        raise ValueError(f"Syn minimum={syn} bit; threshold=0 bit; affected count=1")
    assert (r, un_u, un_v, syn) == (0, 0, 0, 1)
    for u, v, w in table:
        assert u ^ v == w
    return r, un_u, un_v, syn


def main():
    atoms = verify_exact()
    with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "stixsans",
                         "text.color": INK, "svg.fonttype": "path"}):
        fig = plt.figure(figsize=(16, 10), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 1600), ylim=(0, 1000), aspect="equal")
        ax.axis("off")
        texts = []

        def text(x, y, content, size=14, color=INK, chinese=False, ha="left", weight="normal"):
            kwargs = dict(fontsize=size, color=color, ha=ha, va="center", fontweight=weight)
            if chinese:
                kwargs["fontproperties"] = CHINESE
            artist = ax.text(x, y, content, **kwargs)
            texts.append(artist)
            return artist

        def line(x0, x1, y):
            ax.plot((x0, x1), (y, y), color=RULE, lw=0.8)

        def node(x, y, label, radius=30, background=False, operation=False):
            face = "white" if background or operation else INK
            edge = PURPLE if operation else MUTED if background else "none"
            ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor=edge, lw=1.3))
            color = PURPLE if operation else MUTED if background else "white"
            text(x, y, label, size=23 if operation else 21, color=color, ha="center")

        def arrow(start, end, color, radius=30):
            # shrink is measured in points; this canvas uses 100 coordinates/inch.
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", color=color,
                                        lw=1.7, mutation_scale=14,
                                        shrinkA=radius * 0.72 + 3, shrinkB=radius * 0.72 + 5))

        text(64, 946, "一对二的纯协同：随机掩码编码", size=25, chinese=True)
        text(1536, 946, "均匀干预 W · 单位 bit", size=12.5, color=MUTED, chinese=True, ha="right")
        line(64, 1536, 907)
        text(64, 865, r"$W,N\overset{\mathrm{iid}}{\sim}\mathrm{Bern}(1/2),\qquad U=N,\quad V=N\oplus W$", size=16)
        ax.plot((800, 800), (184, 816), color=RULE, lw=0.8)

        # Actual causal construction: N is background randomness, not a second
        # intervention source. Gray arrows are background paths, purple is input.
        text(64, 805, "生成机制", size=12.5, color=MUTED, chinese=True)
        arrow((210, 726), (660, 726), MUTED)
        arrow((210, 726), (480, 536), MUTED)
        arrow((210, 536), (480, 536), PURPLE)
        arrow((480, 536), (660, 536), PURPLE)
        node(210, 726, "$N$", background=True)
        node(210, 536, "$W$")
        node(480, 536, r"$\oplus$", operation=True)
        node(660, 726, "$U$")
        node(660, 536, "$V$")
        text(210, 777, "随机背景", size=13, color=MUTED, chinese=True, ha="center")
        text(210, 485, "输入", size=13, color=PURPLE, chinese=True, ha="center")
        text(434, 760, "copy", size=15, color=MUTED, ha="center")
        text(480, 485, "XOR", size=14, color=PURPLE, ha="center")
        text(660, 673, "$U=N$", size=14.5, ha="center")
        text(660, 485, "$V=N\oplus W$", size=14.5, ha="center")

        line(64, 750, 437)
        text(64, 405, "输入决定两个输出的关系", size=15, chinese=True)

        def output_pair(x, y, u, v):
            for dx, bit in ((0, u), (58, v)):
                ax.add_patch(Circle((x + dx, y), 20, facecolor="white", edgecolor=INK, lw=1.0))
                text(x + dx, y, str(bit), size=15, ha="center")
            ax.plot((x + 21, x + 37), (y, y), color=MUTED, lw=1.0)

        text(64, 349, "$W=0$：相同", size=15, chinese=True)
        text(64, 263, "$W=1$：相反", size=15, chinese=True)
        for y, pairs in ((349, ((0, 0), (1, 1))), (263, ((0, 1), (1, 0)))):
            output_pair(365, y, *pairs[0])
            output_pair(640, y, *pairs[1])
            text(539, y, "或", size=12.5, color=MUTED, chinese=True, ha="center")
        text(394, 386, "$(U,V)$", size=12, color=MUTED, ha="center")
        text(669, 386, "$(U,V)$", size=12, color=MUTED, ha="center")
        text(64, 207, "给定 W，两种组合各以 1/2 出现。", size=12.5, color=MUTED, chinese=True)

        # Keep only the same three compact calculation groups as the last figure.
        x = 850
        text(x, 805, "互信息", size=12.5, color=MUTED, chinese=True)
        text(x, 763, r"$I(W;U)=I(W;V)=I(U;V)=0$", size=15)
        text(x, 721, r"$I(W;U,V)=H(W)-H(W\mid U,V)=1-0=1$", size=14)
        text(x, 680, r"$\mathrm{CoI}=0+0-1=-1$", size=15)

        text(x, 622, "单端提取", size=12.5, color=MUTED, chinese=True)
        text(x, 580, "取常量处理：$Z_W^*=Z_V^*=Z_U^*=z_0$", size=14.5, chinese=True)
        text(x, 538, r"$C_{\mathrm{ext}}(U,V;W)=I(U;V)-I(U;V\mid Z_W^*)=0-0=0$", size=13.8)
        text(x, 497, r"$C_{\mathrm{ext}}(U,W;V)=C_{\mathrm{ext}}(V,W;U)=0$", size=14.5)

        text(x, 439, "冗余、特有与协同", size=12.5, color=MUTED, chinese=True)
        text(x, 397, r"$R=\max\{0,0,0\}=0$", size=15)
        text(x, 355, r"$\mathrm{Un}_U=\mathrm{Un}_V=0-0=0$", size=15)
        text(x, 313, r"$\mathrm{Syn}=R-\mathrm{CoI}=0-(-1)=1$", size=16, color=PURPLE)
        line(x, 1536, 265)
        text(1193, 217, "纯协同 · 1 bit", size=20, color=PURPLE, chinese=True, ha="center")

        line(64, 1536, 157)
        text(800, 112, r"$W=U\oplus V$", size=23, color=PURPLE, ha="center")
        text(800, 62, "单独看任一输出都不知道 W，合起来即可恢复 W。", size=15,
             chinese=True, ha="center")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in texts:
            extent = artist.get_window_extent(renderer)
            if not fig.bbox.contains(extent.x0, extent.y0) or not fig.bbox.contains(extent.x1, extent.y1):
                raise RuntimeError(f"Text exceeds the canvas: {artist.get_text()}")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT.with_suffix(".png"), dpi=200, facecolor="white", bbox_inches=None)
        fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white", bbox_inches=None)
        plt.close(fig)
    print(f"Exact (R, Un_U, Un_V, Syn): {atoms}")
    print("Syn tolerance=0 bit; negative/adjusted counts=0/0.")
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".svg"))


if __name__ == "__main__":
    main()
